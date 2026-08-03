//! Resolver for a model's KV-cache dimensions, read from its GGUF **header** over
//! the network (never the weights), so the catalog fit badge can be *exact* before
//! a model is downloaded.
//!
//! This is a dedicated module rather than ad-hoc background spawns because the
//! caching/fetch behavior is where a naive version goes wrong. The invariants it
//! enforces:
//!
//! - **De-duplicated & rate-limited.** At most one in-flight fetch per model, and
//!   at most [`MAX_CONCURRENT_FETCHES`] fetches across the process. A page render
//!   can never fan out an unbounded burst of subprocesses.
//! - **No hot-path fetches.** [`lookup`](DimsResolver::lookup) is a pure, sync map
//!   read for the badge. Fetches are *scheduled* explicitly ([`ensure`],
//!   [`warm`]) — never as a side-effect of serving a page.
//! - **Write-behind, bounded, TTL'd disk cache.** Results persist so a restart
//!   doesn't re-fetch, but writes are debounced (N fetches → O(1) writes by a
//!   single flusher), the cache is size-capped with oldest-eviction, and entries
//!   expire so a re-upload eventually re-resolves.
//! - **Honest failure modes.** A header that parsed but is not a dense LLaMA-family
//!   model is cached as a *negative* (never re-fetched); a transient fetch error
//!   backs off (retried next process). Either way the caller falls back to the
//!   coarse size estimate — nothing here can block a load or a page.
//!
//! HTTP egress uses the same `curl` subprocess pattern as `hf_browse` / `catalog`
//! (the project deliberately avoids a heavy HTTP-client dependency); the fetch runs
//! on the blocking pool, never the async executor.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::fit::ModelDims;

/// Max cache entries kept on disk; the oldest are evicted past this.
const MAX_ENTRIES: usize = 512;
/// Entries older than this are dropped on load (a re-quant/re-upload re-resolves).
const ENTRY_TTL_SECS: u64 = 60 * 60 * 24 * 30;
/// After a *transient* fetch error, don't retry the same model for this long
/// (process-local; cleared on restart so genuine transients recover).
const FETCH_BACKOFF_SECS: u64 = 60 * 30;
/// Max concurrent header fetches — bounds bandwidth and subprocess count.
const MAX_CONCURRENT_FETCHES: usize = 4;
/// curl connect timeout — a stalled TCP/TLS handshake must not hold a permit.
const CONNECT_TIMEOUT_SECS: u64 = 10;
/// curl overall timeout — a slow/hung transfer must release its permit and in-flight
/// slot in bounded time (the outcome is a retriable transient error).
const FETCH_MAX_TIME_SECS: u64 = 60;
/// Bytes of the file head to fetch — covers metadata (incl. 128k-vocab tokenizer
/// arrays) plus tensor-info for current catalog models; a too-small fetch simply
/// fails to parse and the caller falls back to the estimate.
const HEADER_BYTES: u64 = 12 * 1024 * 1024;
/// Persisted-cache schema version; a mismatch is ignored (treated as empty).
const CACHE_SCHEMA_VERSION: u32 = 1;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Outcome of a single header fetch+parse.
#[derive(Clone, Copy)]
enum FetchOutcome {
    /// Parsed real dense dims.
    Resolved(ModelDims),
    /// Header fetched and parsed, but not a dense LLaMA-family model — caching this
    /// as a negative avoids re-fetching a model we can never resolve.
    Unparseable,
    /// The fetch itself failed (offline, HTTP error, unsafe input). Retried later.
    FetchError,
}

/// The answer to an explicit "resolve this model's dimensions" request.
///
/// Distinguishing the two no-dims cases is load-bearing for any caller that offers
/// the user a "check this" action. Collapsing them to `Option<ModelDims>` means a
/// settled negative and a failed attempt are indistinguishable, so the action can
/// never be retired and pressing it looks like a no-op — which is exactly what a
/// flattened first version of this API produced in the Models tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimsResolution {
    /// Real dimensions, from a fresh fetch or the cache.
    Resolved(ModelDims),
    /// Settled negative: the header was fetched and parsed, and this is not a model
    /// whose KV dimensions can be derived. Asking again cannot change the answer.
    Unresolvable,
    /// The attempt did not settle — offline, an HTTP error, rate-limited, remote
    /// dims disabled, or another resolution for the same model already running.
    /// A later attempt may succeed.
    Retryable,
}

impl DimsResolution {
    /// Whether the question is answered for good. `true` for a resolution **and**
    /// for a settled negative; `false` only when retrying could still help.
    pub fn is_settled(self) -> bool {
        match self {
            DimsResolution::Resolved(_) | DimsResolution::Unresolvable => true,
            DimsResolution::Retryable => false,
        }
    }

    /// The dimensions, if any were resolved.
    pub fn dims(self) -> Option<ModelDims> {
        match self {
            DimsResolution::Resolved(dims) => Some(dims),
            DimsResolution::Unresolvable | DimsResolution::Retryable => None,
        }
    }
}

/// A persisted cache entry. `dims: None` is a *negative* (fetched, parsed, but not a
/// dense model). `Some` is resolved. `fetched_at` drives TTL and LRU eviction.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dims: Option<ModelDims>,
    fetched_at: u64,
}

/// On-disk cache document (versioned for forward-compatibility).
#[derive(serde::Serialize, serde::Deserialize)]
struct DiskCache {
    version: u32,
    entries: HashMap<String, CacheEntry>,
}

struct Inner {
    /// Persisted: resolved dims and known-unparseable negatives.
    entries: HashMap<String, CacheEntry>,
    /// Memory-only transient-error backoff: key → unix secs of last failure.
    backoff: HashMap<String, u64>,
    /// De-dup: models with a fetch in flight right now.
    in_flight: HashSet<String>,
    /// Set when `entries` changed since the last flush.
    dirty: bool,
}

/// Process-wide resolver. Access via [`global`].
pub struct DimsResolver {
    inner: Mutex<Inner>,
    permits: Arc<tokio::sync::Semaphore>,
    disabled: bool,
    cache_path: PathBuf,
}

fn key_of(repo_id: &str, filename: &str) -> String {
    format!("{repo_id}/{filename}")
}

/// The process-wide resolver, seeded from the on-disk cache on first use.
pub fn global() -> &'static DimsResolver {
    static RESOLVER: OnceLock<DimsResolver> = OnceLock::new();
    RESOLVER.get_or_init(DimsResolver::new)
}

impl DimsResolver {
    fn new() -> Self {
        let disabled = std::env::var("CAMELID_NO_REMOTE_DIMS")
            .ok()
            .as_deref()
            .map(str::trim)
            == Some("1");
        let cache_path = cache_path();
        // Seed from disk (dropping expired), unless under test — tests keep the
        // global resolver hermetic and never touch the real user cache.
        let entries = if cfg!(test) {
            HashMap::new()
        } else {
            load_disk_cache(&cache_path)
        };
        DimsResolver {
            inner: Mutex::new(Inner {
                entries,
                backoff: HashMap::new(),
                in_flight: HashSet::new(),
                dirty: false,
            }),
            permits: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_FETCHES)),
            disabled,
            cache_path,
        }
    }

    /// Sync, non-blocking lookup for the badge: the resolved dims, or `None` if not
    /// resolved (never fetched, in flight, a negative, or expired).
    pub fn lookup(&self, repo_id: &str, filename: &str) -> Option<ModelDims> {
        let key = key_of(repo_id, filename);
        let guard = self.inner.lock().ok()?;
        let entry = guard.entries.get(&key)?;
        if now_secs().saturating_sub(entry.fetched_at) > ENTRY_TTL_SECS {
            return None;
        }
        entry.dims
    }

    /// Directly seed a resolved entry. Test-only: lets tests exercise the exact-fit
    /// path deterministically without a network fetch.
    #[cfg(test)]
    pub fn insert_for_test(&self, repo_id: &str, filename: &str, dims: ModelDims) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.entries.insert(
                key_of(repo_id, filename),
                CacheEntry {
                    dims: Some(dims),
                    fetched_at: now_secs(),
                },
            );
        }
    }

    /// Whether a fetch should be scheduled now: not disabled, not already resolved
    /// (or a fresh negative), not in flight, not inside the transient-error backoff.
    fn should_fetch(&self, key: &str) -> bool {
        if self.disabled {
            return false;
        }
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        if guard.in_flight.contains(key) {
            return false;
        }
        if let Some(entry) = guard.entries.get(key) {
            // A fresh entry (resolved OR a known-unparseable negative) is final.
            if now_secs().saturating_sub(entry.fetched_at) <= ENTRY_TTL_SECS {
                return false;
            }
        }
        if let Some(&at) = guard.backoff.get(key) {
            if now_secs().saturating_sub(at) < FETCH_BACKOFF_SECS {
                return false;
            }
        }
        true
    }

    /// Schedule a header fetch for one model if warranted (de-duplicated, rate
    /// limited) and return **immediately** — the fetch runs in the background and the
    /// badge upgrades on a later render. Never blocks the caller (the permit is
    /// awaited inside the spawned task, not here).
    pub fn schedule(&'static self, repo_id: String, filename: String, size: u64) {
        let key = key_of(&repo_id, &filename);
        if !self.should_fetch(&key) {
            return;
        }
        // Claim the in-flight slot; bail if another task raced us to it.
        {
            let mut guard = match self.inner.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            if !guard.in_flight.insert(key.clone()) {
                return;
            }
        }
        let permits = Arc::clone(&self.permits);
        tokio::spawn(async move {
            // Release the in-flight slot no matter how this task ends — a permit
            // error, or a panic anywhere in the fetch/parse/record path. Without
            // this, a panicking fetch would pin the key and `should_fetch` would
            // return false for it for the whole process lifetime. The happy path
            // also clears it in `record` (atomically with the cache write); this is
            // the panic/early-return safety net.
            let slot = InFlightGuard {
                resolver: self,
                key: key.clone(),
            };
            let permit = match permits.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return,
            };
            tokio::task::spawn_blocking(move || {
                let _slot = slot;
                let outcome = fetch_header_dims(&repo_id, &filename, size);
                self.record(&key, outcome);
                drop(permit);
            });
        });
    }

    /// Schedule fetches for a set of models (the startup curated warm).
    pub fn warm(&'static self, models: Vec<(String, String, u64)>) {
        for (repo, file, size) in models {
            self.schedule(repo, file, size);
        }
    }

    /// Resolve one model's dims **now**, awaiting the header fetch.
    ///
    /// This is the explicit, user-initiated counterpart to [`schedule`](Self::schedule):
    /// the background warm is bounded (a page can only ever warm a handful of rows),
    /// so a user looking at a specific model needs a way to ask about *that* one.
    /// It deliberately goes through the same [`should_fetch`](Self::should_fetch)
    /// gate, permit pool, cache and backoff — an interactive check must not be able
    /// to bypass the rate limit or re-fetch a known negative.
    ///
    /// The three-way [`DimsResolution`] is the point: a caller that only learns
    /// "no dims" cannot tell a settled answer from a failed attempt, so a UI built
    /// on it offers "check this" forever and the check appears to do nothing. See
    /// [`DimsResolution::is_settled`].
    pub async fn resolve_now(
        &'static self,
        repo_id: String,
        filename: String,
        size: u64,
    ) -> DimsResolution {
        let key = key_of(&repo_id, &filename);
        // Already resolved, a fresh negative, backed off, in flight, or disabled:
        // answer from the cache instead of issuing a duplicate fetch.
        if !self.should_fetch(&key) {
            return self.cached_resolution(&key);
        }
        {
            let Ok(mut guard) = self.inner.lock() else {
                return DimsResolution::Retryable;
            };
            if !guard.in_flight.insert(key.clone()) {
                // Another resolution for this exact model is already running; its
                // result will be cached, so this caller should simply ask again.
                return DimsResolution::Retryable;
            }
        }
        // Held across the await and the record below, so the slot is released
        // however this returns — including a cancelled request future.
        let _slot = InFlightGuard {
            resolver: self,
            key: key.clone(),
        };
        let Ok(permit) = Arc::clone(&self.permits).acquire_owned().await else {
            return DimsResolution::Retryable;
        };
        let fetched = tokio::task::spawn_blocking(move || {
            let outcome = fetch_header_dims(&repo_id, &filename, size);
            drop(permit);
            outcome
        })
        .await;
        let Ok(outcome) = fetched else {
            return DimsResolution::Retryable;
        };
        self.record(&key, outcome);
        match outcome {
            FetchOutcome::Resolved(dims) => DimsResolution::Resolved(dims),
            // Fetched and parsed, but not a model we can size. Re-asking cannot help.
            FetchOutcome::Unparseable => DimsResolution::Unresolvable,
            FetchOutcome::FetchError => DimsResolution::Retryable,
        }
    }

    /// The resolution implied by the cache alone, for the paths that must not
    /// fetch. A fresh entry is final either way (dims, or a known negative);
    /// anything else — expired, absent, in flight, backed off, disabled — is a
    /// state a later attempt could change.
    fn cached_resolution(&self, key: &str) -> DimsResolution {
        let Ok(guard) = self.inner.lock() else {
            return DimsResolution::Retryable;
        };
        match guard.entries.get(key) {
            Some(entry) if now_secs().saturating_sub(entry.fetched_at) <= ENTRY_TTL_SECS => {
                match entry.dims {
                    Some(dims) => DimsResolution::Resolved(dims),
                    None => DimsResolution::Unresolvable,
                }
            }
            _ => DimsResolution::Retryable,
        }
    }

    fn clear_in_flight(&self, key: &str) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.in_flight.remove(key);
        }
    }

    /// Store a fetch outcome, updating the caches and eviction, then mark dirty.
    fn record(&self, key: &str, outcome: FetchOutcome) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        guard.in_flight.remove(key);
        match outcome {
            FetchOutcome::Resolved(dims) => {
                guard.backoff.remove(key);
                guard.entries.insert(
                    key.to_string(),
                    CacheEntry {
                        dims: Some(dims),
                        fetched_at: now_secs(),
                    },
                );
                evict_if_needed(&mut guard.entries);
                guard.dirty = true;
            }
            FetchOutcome::Unparseable => {
                guard.backoff.remove(key);
                guard.entries.insert(
                    key.to_string(),
                    CacheEntry {
                        dims: None,
                        fetched_at: now_secs(),
                    },
                );
                evict_if_needed(&mut guard.entries);
                guard.dirty = true;
            }
            FetchOutcome::FetchError => {
                // Transient: don't persist; just back off so we retry later.
                guard.backoff.insert(key.to_string(), now_secs());
            }
        }
    }

    /// Debounced single-writer flush loop. Spawned once by `serve`; coalesces bursts
    /// of stores into one write. Never runs under test (keeps tests off disk).
    pub async fn flush_loop(&'static self) {
        if self.disabled {
            return;
        }
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
        loop {
            interval.tick().await;
            let snapshot = {
                let mut guard = match self.inner.lock() {
                    Ok(g) => g,
                    Err(_) => continue,
                };
                if !guard.dirty {
                    continue;
                }
                guard.dirty = false;
                guard.entries.clone()
            };
            let path = self.cache_path.clone();
            let _ = tokio::task::spawn_blocking(move || write_disk_cache(&path, &snapshot)).await;
        }
    }
}

/// Kick off the background lifecycle from `serve`: seed is lazy on first `global()`;
/// here we start the debounced flusher and warm the curated rows once. Both are
/// best-effort background tasks and never block startup.
pub fn start_background(curated: Vec<(String, String, u64)>) {
    let resolver = global();
    if resolver.disabled {
        return;
    }
    tokio::spawn(resolver.flush_loop());
    resolver.warm(curated);
}

/// RAII marker that clears a key's in-flight slot on drop, so an early return or a
/// panic in the spawned fetch task can never pin the key for the process lifetime.
struct InFlightGuard {
    resolver: &'static DimsResolver,
    key: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.resolver.clear_in_flight(&self.key);
    }
}

// --- GGUF header fetch + parse ---------------------------------------------

/// Read a GGUF's KV dimensions from a local file WITHOUT loading weights (header
/// only, the same cheap path `/api/models/inspect` uses). `None` for a non-GGUF, an
/// unreadable header, a non-dense/unknown architecture, or implausible dims.
pub fn dims_from_gguf_file(path: &Path) -> Option<ModelDims> {
    dims_from_gguf(&crate::gguf::read_metadata(path).ok()?)
}

/// Map a parsed GGUF's metadata to dense-Llama KV dims, or `None` for a
/// non-dense/unknown architecture or implausible values.
pub(crate) fn dims_from_gguf(gguf: &crate::gguf::GgufFile) -> Option<ModelDims> {
    let config = crate::model::LlamaModelConfig::from_gguf(gguf).ok()?;
    let dims = crate::model::DenseLlamaDims::from_config(&config).ok()?;
    let dims = ModelDims {
        layers: dims.block_count as u64,
        kv_heads: dims.attention_head_count_kv as u64,
        head_dim: dims.head_dim as u64,
    };
    dims.is_plausible().then_some(dims)
}

/// A GGUF filename that is one shard of a split export (e.g.
/// `model.shard-00001-of-00005.gguf` or `model-00001-of-00003.gguf`) is not a
/// standalone loadable model, so we never fetch or make a fit claim for it.
pub fn is_gguf_shard(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    let Some(idx) = lower.find("-of-") else {
        return false;
    };
    let left_is_num = lower[..idx]
        .rsplit('-')
        .next()
        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()));
    let right_is_num = lower[idx + 4..]
        .split(['-', '.'])
        .next()
        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()));
    left_is_num && right_is_num
}

/// A Hugging Face repo id / filename safe to interpolate into a resolve URL:
/// non-empty, no path traversal, only expected URL-path characters.
pub fn is_safe_hf_component(s: &str) -> bool {
    !s.is_empty()
        && !s.contains("..")
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
}

/// Fetch a model's KV dims from its GGUF header over the network. GGUF stores all
/// metadata + tensor-info at the file start, so we range-fetch a header slice and
/// parse it directly, telling the trusted parser the model's true full length so
/// its tensor-offset bounds checks still hold. We do NOT extend the temp file to
/// the full size: on NTFS that allocates the whole (multi-GB) length on disk even
/// though only the header is fetched. Blocking; call off the executor.
fn fetch_header_dims(repo_id: &str, filename: &str, full_size: u64) -> FetchOutcome {
    // Skip inputs we shouldn't fetch: unknown size, split-model shards, and any
    // unsafe repo/filename we'd interpolate into a URL.
    if full_size == 0 || is_gguf_shard(filename) {
        return FetchOutcome::Unparseable;
    }
    if !is_safe_hf_component(repo_id) || filename.contains('/') || !is_safe_hf_component(filename) {
        return FetchOutcome::Unparseable;
    }
    let safe: String = filename
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    // The temp path must be unique per (repo, filename): different repos routinely
    // ship the same filename, and `in_flight` dedup is keyed by the full pair, so
    // those fetch concurrently. Filename alone would collide into one temp file and
    // race the write/parse/remove. Hash the full key for a collision-free name.
    let token = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(repo_id, &mut hasher);
        std::hash::Hash::hash(filename, &mut hasher);
        std::hash::Hasher::finish(&hasher)
    };
    let tmp = std::env::temp_dir().join(format!(
        "camelid-hdr-{}-{token:016x}-{safe}",
        std::process::id()
    ));
    let range_end = HEADER_BYTES.min(full_size).saturating_sub(1);
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}");
    let fetched = std::process::Command::new("curl")
        .args([
            "-fsSL",
            "--connect-timeout",
            &CONNECT_TIMEOUT_SECS.to_string(),
            "--max-time",
            &FETCH_MAX_TIME_SECS.to_string(),
            "-r",
            &format!("0-{range_end}"),
            "-o",
        ])
        .arg(&tmp)
        .arg(&url)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !fetched {
        let _ = std::fs::remove_file(&tmp);
        return FetchOutcome::FetchError;
    }
    // Parse the header prefix directly, validating tensor offsets against the real
    // full length. No file extension, so no disk is allocated for the unfetched
    // body.
    let parsed = crate::gguf::read_metadata_with_len(&tmp, full_size);
    let _ = std::fs::remove_file(&tmp);
    match parsed.ok().as_ref().and_then(dims_from_gguf) {
        Some(d) => FetchOutcome::Resolved(d),
        // Fetched fine but not dense-parseable → negative (don't re-fetch).
        None => FetchOutcome::Unparseable,
    }
}

// --- disk cache -------------------------------------------------------------

/// Where the persisted dims cache lives. Honors `CAMELID_FIT_DIMS_CACHE` (tests use
/// it for isolation); otherwise the per-user OS cache dir, else the temp dir.
fn cache_path() -> PathBuf {
    if let Some(p) = std::env::var_os("CAMELID_FIT_DIMS_CACHE") {
        return PathBuf::from(p);
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    };
    base.unwrap_or_else(std::env::temp_dir)
        .join("camelid")
        .join("fit-dims-cache.json")
}

/// Read + validate the persisted cache, dropping expired entries. Missing, corrupt,
/// or wrong-version → empty (never panics).
fn load_disk_cache(path: &Path) -> HashMap<String, CacheEntry> {
    let Some(doc) = std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<DiskCache>(&s).ok())
    else {
        return HashMap::new();
    };
    if doc.version != CACHE_SCHEMA_VERSION {
        return HashMap::new();
    }
    let now = now_secs();
    doc.entries
        .into_iter()
        .filter(|(_, e)| now.saturating_sub(e.fetched_at) <= ENTRY_TTL_SECS)
        .collect()
}

/// Persist the cache (best-effort; creates the parent dir).
fn write_disk_cache(path: &Path, entries: &HashMap<String, CacheEntry>) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let doc = DiskCache {
        version: CACHE_SCHEMA_VERSION,
        entries: entries.clone(),
    };
    if let Ok(json) = serde_json::to_string(&doc) {
        let _ = std::fs::write(path, json);
    }
}

/// Evict the oldest entries (by `fetched_at`) until at most [`MAX_ENTRIES`] remain.
fn evict_if_needed(entries: &mut HashMap<String, CacheEntry>) {
    if entries.len() <= MAX_ENTRIES {
        return;
    }
    let mut by_age: Vec<(u64, String)> = entries
        .iter()
        .map(|(k, e)| (e.fetched_at, k.clone()))
        .collect();
    by_age.sort_unstable();
    let excess = entries.len() - MAX_ENTRIES;
    for (_, key) in by_age.into_iter().take(excess) {
        entries.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(l: u64) -> ModelDims {
        ModelDims {
            layers: l,
            kv_heads: 8,
            head_dim: 128,
        }
    }

    #[test]
    fn shard_filenames_are_detected() {
        assert!(is_gguf_shard("model.shard-00001-of-00005.gguf"));
        assert!(is_gguf_shard("Meta-Llama-70B-00001-of-00003.gguf"));
        assert!(!is_gguf_shard("Qwen3-0.6B-Q8_0.gguf"));
        assert!(!is_gguf_shard("something-of-value.gguf")); // "-of-" but not numeric
    }

    #[test]
    fn unsafe_hf_components_are_rejected() {
        assert!(is_safe_hf_component("Qwen/Qwen3-0.6B-GGUF"));
        assert!(is_safe_hf_component("Model-Q8_0.gguf"));
        assert!(!is_safe_hf_component(""));
        assert!(!is_safe_hf_component("../../etc/passwd"));
        assert!(!is_safe_hf_component("repo/../x"));
        assert!(!is_safe_hf_component("bad name.gguf?x=1"));
    }

    #[test]
    fn disk_cache_round_trips_and_tolerates_missing_corrupt_and_version() {
        let dir = std::env::temp_dir().join(format!("camelid-fitdims-test-{}", std::process::id()));
        let path = dir.join("dims.json");
        assert!(load_disk_cache(&path).is_empty()); // missing → empty
        let mut map = HashMap::new();
        map.insert(
            "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf".to_string(),
            CacheEntry {
                dims: Some(dims(28)),
                fetched_at: now_secs(),
            },
        );
        write_disk_cache(&path, &map);
        let back = load_disk_cache(&path);
        assert_eq!(back.len(), 1);
        assert_eq!(back.values().next().unwrap().dims, Some(dims(28)));
        // Corrupt → empty.
        std::fs::write(&path, b"{ not json").unwrap();
        assert!(load_disk_cache(&path).is_empty());
        // Wrong version → empty.
        std::fs::write(&path, br#"{"version":999,"entries":{}}"#).unwrap();
        assert!(load_disk_cache(&path).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn expired_entries_are_dropped_on_load() {
        let dir = std::env::temp_dir().join(format!("camelid-fitdims-ttl-{}", std::process::id()));
        let path = dir.join("dims.json");
        let mut map = HashMap::new();
        map.insert(
            "fresh".to_string(),
            CacheEntry {
                dims: Some(dims(1)),
                fetched_at: now_secs(),
            },
        );
        map.insert(
            "stale".to_string(),
            CacheEntry {
                dims: Some(dims(2)),
                fetched_at: now_secs().saturating_sub(ENTRY_TTL_SECS + 1),
            },
        );
        write_disk_cache(&path, &map);
        let back = load_disk_cache(&path);
        assert!(back.contains_key("fresh"));
        assert!(!back.contains_key("stale"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_keeps_the_newest_entries() {
        let mut entries = HashMap::new();
        for i in 0..(MAX_ENTRIES as u64 + 10) {
            entries.insert(
                format!("k{i}"),
                CacheEntry {
                    dims: Some(dims(i.max(1))),
                    fetched_at: 1000 + i, // higher i = newer
                },
            );
        }
        evict_if_needed(&mut entries);
        assert_eq!(entries.len(), MAX_ENTRIES);
        assert!(entries.contains_key(&format!("k{}", MAX_ENTRIES as u64 + 9))); // newest kept
        assert!(!entries.contains_key("k0")); // oldest evicted
    }

    #[test]
    fn negative_entries_serialize_compactly_and_round_trip() {
        let dir = std::env::temp_dir().join(format!("camelid-fitdims-neg-{}", std::process::id()));
        let path = dir.join("dims.json");
        let mut map = HashMap::new();
        map.insert(
            "some/moe-model.gguf".to_string(),
            CacheEntry {
                dims: None, // negative: fetched but not dense-parseable
                fetched_at: now_secs(),
            },
        );
        write_disk_cache(&path, &map);
        let back = load_disk_cache(&path);
        assert_eq!(back.get("some/moe-model.gguf").unwrap().dims, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- minimal GGUF fixture: proves `dims_from_gguf_file` end to end in CI,
    // with no network and no committed binary blob. Byte layout mirrors the GGUF v3
    // reader (magic, version, i64 counts, typed metadata KVs, tensor directory).
    fn push_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_i32(b: &mut Vec<u8>, v: i32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_i64(b: &mut Vec<u8>, v: i64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_u64(b: &mut Vec<u8>, v: u64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_str(b: &mut Vec<u8>, s: &str) {
        push_u64(b, s.len() as u64);
        b.extend_from_slice(s.as_bytes());
    }
    fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
        push_str(b, k);
        push_u32(b, 8); // string type id
        push_str(b, v);
    }
    fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
        push_str(b, k);
        push_u32(b, 4); // u32 type id
        push_u32(b, v);
    }

    #[test]
    fn dims_from_gguf_file_parses_a_minimal_llama_fixture() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        push_u32(&mut b, 3); // version
        push_i64(&mut b, 1); // tensor_count
        push_i64(&mut b, 8); // metadata_count
        kv_str(&mut b, "general.architecture", "llama");
        kv_u32(&mut b, "llama.context_length", 4096);
        kv_u32(&mut b, "llama.embedding_length", 64);
        kv_u32(&mut b, "llama.block_count", 4);
        kv_u32(&mut b, "llama.feed_forward_length", 128);
        kv_u32(&mut b, "llama.attention.head_count", 8);
        kv_u32(&mut b, "llama.attention.head_count_kv", 8);
        kv_u32(&mut b, "llama.vocab_size", 1000);
        // One tensor so the directory + offset checks exercise the real parser.
        push_str(&mut b, "token_embd.weight");
        push_u32(&mut b, 2); // n_dims
        push_i64(&mut b, 4);
        push_i64(&mut b, 2);
        push_i32(&mut b, 0); // f32
        push_u64(&mut b, 0); // relative offset
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        b.extend_from_slice(&[0u8; 4 * 2 * 4]); // tensor data

        let dir =
            std::env::temp_dir().join(format!("camelid-fitdims-fixture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("minimal-llama.gguf");
        std::fs::write(&path, &b).unwrap();

        // embedding 64 / head_count 8 → head_dim 8.
        let dims = dims_from_gguf_file(&path).expect("minimal llama fixture should parse");
        assert_eq!(
            dims,
            ModelDims {
                layers: 4,
                kv_heads: 8,
                head_dim: 8,
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_metadata_with_len_parses_a_header_prefix_without_the_body() {
        // The same minimal llama header, but we persist ONLY the header prefix (no
        // tensor data) and declare the true full length — exactly what the header
        // range-fetch does, and without extending the file to the full size on disk.
        let mut hdr = Vec::new();
        hdr.extend_from_slice(b"GGUF");
        push_u32(&mut hdr, 3);
        push_i64(&mut hdr, 1);
        push_i64(&mut hdr, 8);
        kv_str(&mut hdr, "general.architecture", "llama");
        kv_u32(&mut hdr, "llama.context_length", 4096);
        kv_u32(&mut hdr, "llama.embedding_length", 64);
        kv_u32(&mut hdr, "llama.block_count", 4);
        kv_u32(&mut hdr, "llama.feed_forward_length", 128);
        kv_u32(&mut hdr, "llama.attention.head_count", 8);
        kv_u32(&mut hdr, "llama.attention.head_count_kv", 8);
        kv_u32(&mut hdr, "llama.vocab_size", 1000);
        push_str(&mut hdr, "token_embd.weight");
        push_u32(&mut hdr, 2);
        push_i64(&mut hdr, 4);
        push_i64(&mut hdr, 2);
        push_i32(&mut hdr, 0);
        push_u64(&mut hdr, 0);
        while !hdr.len().is_multiple_of(32) {
            hdr.push(0);
        }
        // The tensor body we deliberately never write nor fetch.
        let full_len = hdr.len() as u64 + 4 * 2 * 4;

        let dir =
            std::env::temp_dir().join(format!("camelid-fitdims-prefix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prefix-llama.gguf");
        std::fs::write(&path, &hdr).unwrap(); // header prefix only

        // The ordinary stat-based parser rejects it — the tensor data runs past the
        // truncated EOF...
        assert!(crate::gguf::read_metadata(&path).is_err());
        // ...but validating against the declared full length parses the prefix, with
        // no disk allocated for the absent body.
        let gguf = crate::gguf::read_metadata_with_len(&path, full_len)
            .expect("header prefix should parse against the declared full length");
        assert_eq!(
            dims_from_gguf(&gguf).unwrap(),
            ModelDims {
                layers: 4,
                kv_heads: 8,
                head_dim: 8,
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn in_flight_guard_releases_the_slot_on_drop() {
        // A panic or early return in the spawned fetch task must not pin the key: the
        // RAII guard clears the in-flight marker regardless of how the task ends.
        let resolver = global();
        let key = "test-fitdims/in-flight-guard.gguf".to_string();
        resolver.inner.lock().unwrap().in_flight.insert(key.clone());
        assert!(resolver.inner.lock().unwrap().in_flight.contains(&key));
        {
            let _slot = InFlightGuard {
                resolver,
                key: key.clone(),
            };
        } // drop here
        assert!(!resolver.inner.lock().unwrap().in_flight.contains(&key));
    }

    #[test]
    fn lookup_round_trips_via_the_test_injector() {
        // Uses a unique key so it can't collide with other tests on the shared global.
        let (repo, file) = ("test-fitdims/lookup-repo", "lookup.gguf");
        assert!(global().lookup(repo, file).is_none());
        global().insert_for_test(
            repo,
            file,
            ModelDims {
                layers: 12,
                kv_heads: 2,
                head_dim: 64,
            },
        );
        assert_eq!(
            global().lookup(repo, file),
            Some(ModelDims {
                layers: 12,
                kv_heads: 2,
                head_dim: 64
            })
        );
    }

    #[test]
    fn a_settled_negative_is_distinguishable_from_a_failed_attempt() {
        // This distinction is the whole reason `resolve_now` does not return
        // `Option<ModelDims>`. A caller that only sees "no dims" cannot retire a
        // "check this" affordance, so pressing it looks like a no-op forever.
        let resolver = global();
        let dims = ModelDims {
            layers: 8,
            kv_heads: 2,
            head_dim: 64,
        };

        // Nothing cached: a later attempt could still succeed.
        let unknown = key_of("test-fitdims/resolution", "never-seen.gguf");
        assert_eq!(
            resolver.cached_resolution(&unknown),
            DimsResolution::Retryable
        );
        assert!(!DimsResolution::Retryable.is_settled());
        assert_eq!(DimsResolution::Retryable.dims(), None);

        // A cached NEGATIVE (fetched, parsed, not a model we can size) is final.
        let negative = key_of("test-fitdims/resolution", "negative.gguf");
        resolver.record(&negative, FetchOutcome::Unparseable);
        assert_eq!(
            resolver.cached_resolution(&negative),
            DimsResolution::Unresolvable
        );
        assert!(DimsResolution::Unresolvable.is_settled());
        assert_eq!(DimsResolution::Unresolvable.dims(), None);

        // A cached POSITIVE is final too, and carries the dims.
        let positive = key_of("test-fitdims/resolution", "positive.gguf");
        resolver.record(&positive, FetchOutcome::Resolved(dims));
        assert_eq!(
            resolver.cached_resolution(&positive),
            DimsResolution::Resolved(dims)
        );
        assert!(DimsResolution::Resolved(dims).is_settled());
        assert_eq!(DimsResolution::Resolved(dims).dims(), Some(dims));

        // A transient fetch error must NOT be cached as a negative: it only backs
        // off, so the question stays open and the caller may retry.
        let transient = key_of("test-fitdims/resolution", "transient.gguf");
        resolver.record(&transient, FetchOutcome::FetchError);
        assert_eq!(
            resolver.cached_resolution(&transient),
            DimsResolution::Retryable
        );
    }

    #[test]
    fn header_fetch_reads_a_real_gguf_when_enabled() {
        // Network-gated: self-skips offline / in normal CI. Enable with
        // CAMELID_TEST_REMOTE_DIMS=1 to verify the header range-fetch end to end.
        if std::env::var("CAMELID_TEST_REMOTE_DIMS").ok().as_deref() != Some("1") {
            return;
        }
        // Qwen3-0.6B (~639 MB) and a 128k-vocab Llama (biggest metadata) — header only.
        for (repo, file, size) in [
            (
                "Qwen/Qwen3-0.6B-GGUF",
                "Qwen3-0.6B-Q8_0.gguf",
                639_446_688u64,
            ),
            (
                "unsloth/Llama-3.2-1B-Instruct-GGUF",
                "Llama-3.2-1B-Instruct-Q8_0.gguf",
                1_321_082_528u64,
            ),
        ] {
            match fetch_header_dims(repo, file, size) {
                FetchOutcome::Resolved(d) => {
                    assert!(d.is_plausible(), "{file}: implausible dims {d:?}");
                    eprintln!("{file} header dims: {d:?}");
                }
                _ => panic!("{file}: expected resolved dims from a real header"),
            }
        }
    }
}
