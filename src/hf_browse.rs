//! Live Hugging Face GGUF browse for the experimental lane.
//!
//! This module is a **network concern only** — it discovers GGUF files on the Hub
//! so the Models tab can offer to download them. It makes NO claim about whether a
//! file will load or run: the `architecture`/`quant` it returns are best-effort
//! guesses from the filename (advisory, never authoritative). The authoritative
//! architecture is read from real GGUF metadata at load time, never here. Browse
//! results therefore always land in the experimental group with a permanent
//! "unverified, no parity claim" marker — discovering a file is not endorsing it.
//!
//! HTTP egress reuses the same `curl` subprocess pattern as `catalog.rs` (no new
//! dependency); every call tolerates offline by returning a typed error rather than
//! panicking, so the catalog endpoint can degrade to curated-only.
//!
//! Latency is the design constraint here: one search is 1 repo-search round-trip
//! plus one file-tree round-trip *per repo*. Serially that is
//! `1 + HF_SEARCH_LIMIT` sequential requests (measured ~5.3 s for 15 repos), which
//! reads to the user as a hung UI. Two bounded mitigations, both dependency-free:
//! the per-repo trees are fetched by a small scoped thread pool
//! ([`REPO_FETCH_CONCURRENCY`]), and identical `(query, cursor)` pages are served
//! from a short-lived in-process cache ([`SEARCH_CACHE_TTL`]) so retyping,
//! back-navigation, and two clients on the same box do not re-pay the round-trips.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Number of repo file-tree requests in flight at once. Each is an independent
/// `curl` subprocess blocked on network IO, so this trades a handful of short-lived
/// threads for a near-linear latency reduction. Kept small deliberately: the Hub
/// rate-limits, and a page only ever inspects `HF_SEARCH_LIMIT` repos.
const REPO_FETCH_CONCURRENCY: usize = 8;

/// curl connect timeout — a stalled TCP/TLS handshake must not pin a worker thread.
const CONNECT_TIMEOUT_SECS: u64 = 10;

/// curl overall timeout — a slow or hung transfer must release its worker in bounded
/// time. One repo timing out costs that repo's files, never the whole page.
const REQUEST_MAX_TIME_SECS: u64 = 30;

/// How long a fetched page stays servable. Short on purpose: the Hub is live data
/// and a stale catalog would be a correctness problem, but a human refining a query
/// (or re-opening the tab) acts well inside this window.
const SEARCH_CACHE_TTL: Duration = Duration::from_secs(60);

/// Max cached pages. Bounds memory for a pathological typist; oldest evicted first.
const SEARCH_CACHE_MAX_ENTRIES: usize = 32;

/// One GGUF file discovered on the Hub. `size_bytes` is the real content size
/// (LFS-aware). `architecture`/`quant` are filename guesses — advisory only, empty
/// when nothing matched.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HfGgufFile {
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
    pub downloads: u64,
    pub likes: u64,
    /// Filename-guessed architecture (advisory, NOT authoritative). Empty if unknown.
    pub architecture: String,
    /// Filename-guessed quant (advisory, NOT authoritative). Empty if unknown.
    pub quant: String,
}

/// A page of search results plus the opaque cursor for the next page (`None` when
/// the Hub advertised no further pages).
#[derive(Debug, Clone)]
pub struct HfSearchPage {
    pub files: Vec<HfGgufFile>,
    pub next_cursor: Option<String>,
}

/// Search the Hugging Face Hub for GGUF repos matching `query` and enumerate their
/// `*.gguf` files with real sizes. `limit` bounds the number of repos inspected;
/// `cursor` resumes a previous page. Network failure → typed `Err` (never panic),
/// so callers can fall back to curated-only.
///
/// Identical `(query, cursor)` pages are served from the in-process cache for
/// [`SEARCH_CACHE_TTL`]. Only successes are cached — an offline blip must not pin a
/// failure for a minute.
pub async fn search_gguf(
    query: &str,
    limit: usize,
    cursor: Option<&str>,
) -> anyhow::Result<HfSearchPage> {
    let key = cache_key(query, limit, cursor);
    if let Some(hit) = search_cache().get(&key, Instant::now()) {
        return Ok(hit);
    }
    let query = query.to_string();
    let cursor = cursor.map(|c| c.to_string());
    // curl is blocking; keep it off the async executor.
    let page =
        tokio::task::spawn_blocking(move || search_gguf_blocking(&query, limit, cursor.as_deref()))
            .await
            .map_err(|err| anyhow::anyhow!("hugging face search task panicked: {err}"))??;
    search_cache().insert(key, page.clone(), Instant::now());
    Ok(page)
}

/// Cache identity for a page. `limit` is part of it because it bounds how many
/// repos the page inspects: serving a 5-repo page to a caller that asked for 15
/// would silently truncate results. The NUL separators cannot appear in a URL-safe
/// cursor or a user query, so no two distinct triples can collide.
fn cache_key(query: &str, limit: usize, cursor: Option<&str>) -> String {
    format!("{query}\u{0}{limit}\u{0}{}", cursor.unwrap_or_default())
}

/// Bounded, TTL'd page cache. Insertion-ordered `Vec` rather than a `HashMap` +
/// separate ordering structure: at [`SEARCH_CACHE_MAX_ENTRIES`] a linear scan is
/// cheaper than the bookkeeping, and eviction is a single `remove(0)`.
#[derive(Default)]
struct SearchCache {
    entries: Mutex<Vec<CacheEntry>>,
}

struct CacheEntry {
    key: String,
    stored_at: Instant,
    page: HfSearchPage,
}

impl SearchCache {
    /// The cached page for `key` if it has not expired as of `now`. Expired entries
    /// are dropped on the way past, so the cache self-trims without a timer.
    ///
    /// `now` is a parameter rather than read from the clock so expiry and eviction
    /// are deterministically testable.
    fn get(&self, key: &str, now: Instant) -> Option<HfSearchPage> {
        let mut entries = self.lock();
        entries.retain(|e| now.duration_since(e.stored_at) < SEARCH_CACHE_TTL);
        entries
            .iter()
            .find(|e| e.key == key)
            .map(|e| e.page.clone())
    }

    /// Store `page` under `key`, replacing any previous entry for the same key and
    /// evicting the oldest entries past [`SEARCH_CACHE_MAX_ENTRIES`].
    fn insert(&self, key: String, page: HfSearchPage, now: Instant) {
        let mut entries = self.lock();
        entries.retain(|e| e.key != key && now.duration_since(e.stored_at) < SEARCH_CACHE_TTL);
        entries.push(CacheEntry {
            key,
            stored_at: now,
            page,
        });
        let overflow = entries.len().saturating_sub(SEARCH_CACHE_MAX_ENTRIES);
        entries.drain(..overflow);
    }

    /// A poisoned cache mutex must not take down a search: recover the guard and
    /// carry on (the worst case is a stale-but-bounded entry).
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<CacheEntry>> {
        self.entries.lock().unwrap_or_else(|err| err.into_inner())
    }
}

fn search_cache() -> &'static SearchCache {
    static CACHE: OnceLock<SearchCache> = OnceLock::new();
    CACHE.get_or_init(SearchCache::default)
}

struct RepoMeta {
    id: String,
    downloads: u64,
    likes: u64,
}

fn search_gguf_blocking(
    query: &str,
    limit: usize,
    cursor: Option<&str>,
) -> anyhow::Result<HfSearchPage> {
    let limit = limit.clamp(1, 50);

    // 1. Ask the Hub for repos matching the query (GGUF filter, full record so we
    //    pick up download/like counts). `cursor` paginates.
    let mut url = format!(
        "https://huggingface.co/api/models?search={}&filter=gguf&limit={}&full=true",
        urlencode(query),
        limit
    );
    if let Some(c) = cursor {
        url.push_str("&cursor=");
        url.push_str(&urlencode(c));
    }

    let (body, headers) = curl_get_with_headers(&url)?;
    let models: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|err| anyhow::anyhow!("could not parse hugging face search response: {err}"))?;
    let next_cursor = parse_next_cursor(&headers);

    let repos: Vec<RepoMeta> = models
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?.to_string();
                    Some(RepoMeta {
                        downloads: m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                        likes: m.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                        id,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // 2. For each repo, read its file tree for the real `*.gguf` sizes.
    let files = fetch_repo_files_parallel(&repos);

    Ok(HfSearchPage { files, next_cursor })
}

/// Fetch every repo's file tree with at most [`REPO_FETCH_CONCURRENCY`] requests in
/// flight, preserving the Hub's repo ordering in the output.
///
/// Ordering matters: the Hub returns repos in relevance order and the Models tab
/// renders them in the order received, so this is the ranking. Workers therefore
/// tag their results with the repo index and the page is re-sorted before flatten;
/// completion order never leaks into the UI.
///
/// One repo failing (private, renamed, rate-limited, timed out) drops only that
/// repo's files, never the page — same tolerance the sequential version had.
fn fetch_repo_files_parallel(repos: &[RepoMeta]) -> Vec<HfGgufFile> {
    if repos.is_empty() {
        return Vec::new();
    }
    let worker_count = REPO_FETCH_CONCURRENCY.min(repos.len());
    let next = AtomicUsize::new(0);
    let collected: Mutex<Vec<(usize, Vec<HfGgufFile>)>> =
        Mutex::new(Vec::with_capacity(repos.len()));

    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(repo) = repos.get(index) else {
                    break;
                };
                let Ok(files) = repo_gguf_files(repo) else {
                    continue;
                };
                if let Ok(mut out) = collected.lock() {
                    out.push((index, files));
                }
            });
        }
    });

    let mut collected = collected
        .into_inner()
        .unwrap_or_else(|err| err.into_inner());
    collected.sort_unstable_by_key(|(index, _)| *index);
    collected.into_iter().flat_map(|(_, files)| files).collect()
}

/// Enumerate the top-level `*.gguf` files in a repo with LFS-aware sizes, mirroring
/// the `remote_size()` logic in `catalog.rs`. Only top-level files are returned:
/// the downloader writes to `models/<filename>` and the local scan globs
/// `models/*.gguf`, so a nested path would download but never surface as a local
/// model. Sharded/subdir GGUFs are therefore skipped at browse time.
fn repo_gguf_files(repo: &RepoMeta) -> anyhow::Result<Vec<HfGgufFile>> {
    let url = format!(
        "https://huggingface.co/api/models/{}/tree/main?recursive=1",
        repo.id
    );
    let (body, _) = curl_get_with_headers(&url)?;
    let tree: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|err| anyhow::anyhow!("could not parse hugging face tree response: {err}"))?;

    let mut out = Vec::new();
    let Some(entries) = tree.as_array() else {
        return Ok(out);
    };
    for entry in entries {
        let Some(path) = entry.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        if !path.to_lowercase().ends_with(".gguf") || path.contains('/') {
            continue;
        }
        // LFS/xet-backed files report the real content size under `lfs.size`; the
        // top-level `size` for those is just the pointer's byte count.
        let size = entry
            .get("lfs")
            .and_then(|lfs| lfs.get("size"))
            .and_then(|s| s.as_u64())
            .or_else(|| entry.get("size").and_then(|s| s.as_u64()))
            .unwrap_or(0);

        out.push(HfGgufFile {
            repo_id: repo.id.clone(),
            filename: path.to_string(),
            size_bytes: size,
            downloads: repo.downloads,
            likes: repo.likes,
            architecture: guess_architecture(path, &repo.id),
            quant: guess_quant(path).unwrap_or_default(),
        });
    }
    Ok(out)
}

/// Run `curl -fsSL <url>` returning `(body, raw_response_headers)`. Headers are
/// dumped to a temp file (`-D`) so the body stays clean JSON regardless of
/// redirects. Offline / HTTP error / timeout → typed `Err`.
///
/// The timeouts are load-bearing, not defensive dressing: these calls run on a
/// fixed-size worker pool, so an untimed hung connection would retire a worker for
/// the life of the process and eventually starve every search.
fn curl_get_with_headers(url: &str) -> anyhow::Result<(Vec<u8>, String)> {
    let header_path = unique_temp_path("camelid-hf-hdr");
    let output = std::process::Command::new("curl")
        .args([
            "-fsSL",
            "--connect-timeout",
            &CONNECT_TIMEOUT_SECS.to_string(),
            "--max-time",
            &REQUEST_MAX_TIME_SECS.to_string(),
            "-D",
        ])
        .arg(&header_path)
        .arg(url)
        .output()
        .map_err(|err| anyhow::anyhow!("could not run curl (is it installed?): {err}"))?;

    let headers = std::fs::read_to_string(&header_path).unwrap_or_default();
    let _ = std::fs::remove_file(&header_path);

    if !output.status.success() {
        anyhow::bail!(
            "hugging face request failed (curl exited {})",
            output.status
        );
    }
    Ok((output.stdout, headers))
}

/// A process-unique temp path. Avoids time/random (unavailable in some sandboxes)
/// by combining the pid with a monotonic counter.
fn unique_temp_path(prefix: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{}-{n}.tmp", std::process::id()))
}

/// Pull the `cursor` query param out of the `rel="next"` entry of an HTTP `Link`
/// header, if any. Returns the decoded cursor (callers re-encode before reuse).
fn parse_next_cursor(headers: &str) -> Option<String> {
    for line in headers.lines() {
        if !line.to_lowercase().starts_with("link:") {
            continue;
        }
        let value = &line[line.find(':')? + 1..];
        for part in value.split(',') {
            if !part.to_lowercase().contains("rel=\"next\"") {
                continue;
            }
            let start = part.find('<')?;
            let end = part.find('>')?;
            return extract_query_param(&part[start + 1..end], "cursor");
        }
    }
    None
}

/// Decoded value of query parameter `key` in `url`, if present.
fn extract_query_param(url: &str, key: &str) -> Option<String> {
    let query = url.split('?').nth(1)?;
    for pair in query.split('&') {
        let mut it = pair.splitn(2, '=');
        if it.next()? == key {
            return Some(urldecode(it.next().unwrap_or("")));
        }
    }
    None
}

/// Percent-encode a query component (RFC 3986 unreserved set kept verbatim).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Reverse `urlencode` (also tolerates `+` left as-is; HF cursors are %-encoded).
fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 3 <= bytes.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Best-effort quant guess from a filename (advisory only). Longest tokens first so
/// `Q4_K_M` isn't shadowed by `Q4_K`.
fn guess_quant(filename: &str) -> Option<String> {
    let upper = filename.to_uppercase();
    const PATTERNS: &[&str] = &[
        "IQ2_XXS", "IQ3_XXS", "IQ2_XS", "IQ3_XS", "IQ4_XS", "IQ4_NL", "IQ1_S", "IQ1_M", "IQ2_S",
        "IQ2_M", "IQ3_S", "IQ3_M", "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M",
        "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_K", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q4_0", "Q4_1", "Q5_0",
        "Q5_1", "Q8_0", "BF16", "F16", "F32",
    ];
    PATTERNS
        .iter()
        .find(|p| upper.contains(**p))
        .map(|p| (*p).to_string())
}

/// Best-effort architecture guess from repo/filename tokens (advisory only). The
/// real architecture comes from GGUF metadata at load time; this never gates a
/// lane. More specific tokens are matched first.
fn guess_architecture(filename: &str, repo_id: &str) -> String {
    let hay = format!("{repo_id} {filename}").to_lowercase();
    const TABLE: &[(&str, &str)] = &[
        ("qwen3", "qwen3"),
        ("qwen2", "qwen2"),
        ("smollm3", "smollm3"),
        ("smollm", "smollm3"),
        ("gemma-4", "gemma4"),
        ("gemma4", "gemma4"),
        ("gemma-3", "gemma3"),
        ("gemma3", "gemma3"),
        ("phi-3", "phi3"),
        ("phi3", "phi3"),
        ("lfm2", "lfm2"),
        ("mixtral", "mixtral"),
        ("mistral", "mistral"),
        ("tinyllama", "llama"),
        ("llama", "llama"),
    ];
    for (needle, arch) in TABLE {
        if hay.contains(needle) {
            return (*arch).to_string();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guesses_common_quants_longest_first() {
        assert_eq!(guess_quant("model-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(guess_quant("model-Q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(
            guess_quant("model-IQ3_XXS.gguf").as_deref(),
            Some("IQ3_XXS")
        );
        assert_eq!(guess_quant("model-f16.gguf").as_deref(), Some("F16"));
        assert_eq!(guess_quant("model.gguf"), None);
    }

    #[test]
    fn guesses_architecture_advisory() {
        assert_eq!(
            guess_architecture("Qwen3-4B-Q8_0.gguf", "Qwen/Qwen3-4B-GGUF"),
            "qwen3"
        );
        assert_eq!(
            guess_architecture("model.gguf", "TheBloke/TinyLlama-1.1B-GGUF"),
            "llama"
        );
        assert_eq!(
            guess_architecture("gemma-3-1b-it-Q8_0.gguf", "ggml-org/x"),
            "gemma3"
        );
        assert_eq!(guess_architecture("mystery.gguf", "someone/mystery"), "");
    }

    #[test]
    fn urlencode_urldecode_roundtrip() {
        let raw = "abc/123+def=ghi?x&y";
        assert_eq!(urldecode(&urlencode(raw)), raw);
    }

    #[test]
    fn extracts_cursor_from_query() {
        assert_eq!(
            extract_query_param(
                "https://huggingface.co/api/models?search=q&cursor=AbC%3D%3D",
                "cursor"
            )
            .as_deref(),
            Some("AbC==")
        );
        assert_eq!(extract_query_param("https://x/api?foo=1", "cursor"), None);
    }

    #[test]
    fn parses_next_cursor_from_link_header() {
        let headers = "HTTP/2 200\r\nlink: <https://huggingface.co/api/models?search=q&cursor=NEXT%2B1>; rel=\"next\"\r\ncontent-type: application/json\r\n";
        assert_eq!(parse_next_cursor(headers).as_deref(), Some("NEXT+1"));
        assert_eq!(
            parse_next_cursor("HTTP/2 200\r\ncontent-type: application/json\r\n"),
            None
        );
    }

    /// A one-file page tagged by `marker`, so tests can tell cached pages apart.
    fn page(marker: &str) -> HfSearchPage {
        HfSearchPage {
            files: vec![HfGgufFile {
                repo_id: format!("owner/{marker}"),
                filename: format!("{marker}.gguf"),
                size_bytes: 1,
                downloads: 0,
                likes: 0,
                architecture: String::new(),
                quant: String::new(),
            }],
            next_cursor: None,
        }
    }

    fn only_repo(page: &HfSearchPage) -> &str {
        &page.files[0].repo_id
    }

    #[test]
    fn cache_key_separates_query_limit_and_cursor() {
        // Without separators, ("a\0b", None) and ("a", Some("b")) would collide.
        assert_ne!(
            cache_key("a\u{0}b", 15, None),
            cache_key("a", 15, Some("b"))
        );
        assert_eq!(cache_key("phi", 15, None), cache_key("phi", 15, Some("")));
        // A page sized for 15 repos must not be served to a caller asking for 5.
        assert_ne!(cache_key("phi", 15, None), cache_key("phi", 5, None));
    }

    #[test]
    fn cache_serves_a_hit_within_the_ttl_and_expires_after_it() {
        let cache = SearchCache::default();
        let t0 = Instant::now();
        cache.insert("phi".to_string(), page("a"), t0);

        let fresh = cache.get("phi", t0 + SEARCH_CACHE_TTL - Duration::from_millis(1));
        assert_eq!(only_repo(&fresh.expect("within ttl")), "owner/a");

        assert!(cache.get("phi", t0 + SEARCH_CACHE_TTL).is_none());
        assert!(cache.get("qwen", t0).is_none());
    }

    #[test]
    fn cache_replaces_rather_than_duplicates_the_same_key() {
        let cache = SearchCache::default();
        let t0 = Instant::now();
        cache.insert("phi".to_string(), page("old"), t0);
        cache.insert("phi".to_string(), page("new"), t0 + Duration::from_secs(1));

        assert_eq!(cache.lock().len(), 1);
        let hit = cache.get("phi", t0 + Duration::from_secs(1)).expect("hit");
        assert_eq!(only_repo(&hit), "owner/new");
    }

    #[test]
    fn cache_evicts_the_oldest_entries_past_the_cap() {
        let cache = SearchCache::default();
        let t0 = Instant::now();
        for i in 0..SEARCH_CACHE_MAX_ENTRIES + 3 {
            cache.insert(format!("q{i}"), page(&format!("m{i}")), t0);
        }
        assert_eq!(cache.lock().len(), SEARCH_CACHE_MAX_ENTRIES);
        // The three oldest keys are gone; the newest survive.
        assert!(cache.get("q0", t0).is_none());
        assert!(cache.get("q2", t0).is_none());
        assert!(cache.get("q3", t0).is_some());
        assert!(cache
            .get(&format!("q{}", SEARCH_CACHE_MAX_ENTRIES + 2), t0)
            .is_some());
    }
}
