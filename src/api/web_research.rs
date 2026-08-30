//! Bounded server-side web research for the embedded Chat UI.
//!
//! This is deliberately a pre-generation data path, not a model tool. Gemma 4
//! rows do not have a certified tool-call renderer, while the Web UI still
//! needs to ground a prompt that contains links or explicitly asks for current
//! information. The browser asks this endpoint first and injects the returned,
//! explicitly-untrusted excerpts into the ordinary chat request.
//!
//! Network safety is owned here. Only public HTTP(S) targets on the default
//! ports are accepted. An optional operator GitHub token is scoped to exact
//! api.github.com HTTPS hops and private repositories remain excluded. Every redirect is parsed, checked, and
//! DNS-resolved again before curl sees it; validated DNS answers are pinned with
//! `--resolve`, proxy use and curlrc loading are disabled, and response/time/
//! redirect counts are bounded. Incoming Camelid authentication headers never
//! enter this module and therefore cannot be forwarded to the public web.

use std::{
    collections::{HashMap, HashSet},
    io::{Read, Write},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs},
    process::{Command, Stdio},
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::{Duration, Instant},
};

use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use url::{Host, Url};

use super::{api_error, AppState};

const MAX_PROMPT_BYTES: usize = 64 * 1024;
const MAX_DIRECT_URLS: usize = 4;
const MAX_TOTAL_SOURCES: usize = 6;
const MAX_SEARCH_RESULTS: usize = 8;
const MAX_SEARCH_FETCHES: usize = 3;
const MAX_REDIRECTS: usize = 4;
const MAX_HTTP_BODY_BYTES: usize = 512 * 1024;
const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;
const MAX_SOURCE_EXCERPT_CHARS: usize = 12_000;
const MAX_README_CHARS: usize = 4_000;
const MAX_CODE_FILE_CHARS: usize = 2_400;
const MAX_GITHUB_SOURCE_FILES: usize = 3;
const MAX_GITHUB_REF_SEGMENTS: usize = 16;
const MAX_GITHUB_API_TREE_PAGES: usize = 12;
const MAX_GITHUB_API_TREE_DEPTH: usize = 6;
const MAX_GITHUB_API_TREE_ENTRIES: usize = 1_024;
const MAX_GITHUB_HTML_PAGES: usize = 6;
const MAX_GITHUB_HTML_DEPTH: usize = 4;
const MAX_SEARCH_QUERY_CHARS: usize = 600;
const MAX_CONCURRENT_RESEARCH: usize = 2;
const MAX_CONCURRENT_DNS_LOOKUPS: usize = 4;
const FETCH_CACHE_CAPACITY: usize = 96;
const FETCH_CACHE_TTL: Duration = Duration::from_secs(90);
const RESEARCH_ADMISSION_TIMEOUT: Duration = Duration::from_millis(750);
const RESEARCH_TOTAL_DEADLINE: Duration = Duration::from_secs(30);
const GITHUB_TOKEN_ENV: &str = "CAMELID_WEB_GITHUB_TOKEN";
const BRAVE_SEARCH_ENDPOINT: &str = "https://search.brave.com/search";
const DDG_SEARCH_ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";
const BING_SEARCH_ENDPOINT: &str = "https://www.bing.com/search";
const USER_AGENT: &str = "Camelid-WebResearch/0.6 (+https://github.com/timtoole02/Camelid)";

#[derive(Debug, Deserialize)]
pub(super) struct WebResearchRequest {
    prompt: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct WebResearchChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct WebResearchSource {
    pub id: usize,
    pub title: String,
    pub url: String,
    pub excerpt: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<WebResearchChunk>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct WebResearchResponse {
    /// complete | partial | skipped | failed
    pub status: &'static str,
    /// True whenever the prompt requested research, even if providers failed.
    pub triggered: bool,
    /// embedded_urls | explicit_search | current_info | not_needed
    pub reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    pub sources: Vec<WebResearchSource>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub(super) struct FetchResponse {
    status: u16,
    content_type: String,
    body: Vec<u8>,
    location: Option<String>,
    etag: Option<String>,
    final_url: Option<Url>,
}

/// A credential-free response for other crate-local callers that need the same
/// public-network boundary as Web Auto.  Deliberately exposes no redirect or
/// cache internals and never enables provider authentication.
#[derive(Debug)]
pub(crate) struct PublicHttpResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
    pub final_url: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PublicHttpMethod {
    Get,
    Head,
}

impl PublicHttpMethod {
    fn parse(method: &str) -> Result<Self, String> {
        match method.trim().to_ascii_uppercase().as_str() {
            "GET" => Ok(Self::Get),
            "HEAD" => Ok(Self::Head),
            other => Err(format!(
                "unsupported HTTP method `{other}` (only GET and HEAD are allowed)"
            )),
        }
    }

    fn head_only(self) -> bool {
        self == Self::Head
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProviderAuthIntent {
    None,
    GithubRepositoryMetadata,
    GithubVerifiedPublicApi,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GithubAuthScope {
    owner: String,
    repo: String,
}

impl FetchResponse {
    #[cfg(test)]
    fn text(status: u16, content_type: &str, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            content_type: content_type.to_string(),
            body: body.into(),
            location: None,
            etag: None,
            final_url: None,
        }
    }
}

/// Small injection seam: production uses curl, tests provide in-memory pages.
/// No request headers are accepted, which makes forwarding a Camelid API key
/// structurally impossible.
pub(super) trait WebTransport: Send + Sync {
    fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String>;

    fn fetch_github_repository_api(
        &self,
        url: &Url,
        accept: &str,
    ) -> Result<FetchResponse, String> {
        self.fetch(url, accept)
    }

    fn fetch_github_verified_public_api(
        &self,
        url: &Url,
        accept: &str,
    ) -> Result<FetchResponse, String> {
        self.fetch(url, accept)
    }

    fn fetch_conditional(
        &self,
        url: &Url,
        accept: &str,
        _etag: Option<&str>,
    ) -> Result<FetchResponse, String> {
        self.fetch(url, accept)
    }

    fn fetch_cancellable(
        &self,
        url: &Url,
        accept: &str,
        etag: Option<&str>,
        cancel: Option<&tokio_util::sync::CancellationToken>,
        deadline: Option<Instant>,
        auth: ProviderAuthIntent,
    ) -> Result<FetchResponse, String> {
        if cancel.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
            return Err("web research was cancelled".to_string());
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(research_deadline_error());
        }
        match auth {
            ProviderAuthIntent::None => self.fetch_conditional(url, accept, etag),
            ProviderAuthIntent::GithubRepositoryMetadata => {
                self.fetch_github_repository_api(url, accept)
            }
            ProviderAuthIntent::GithubVerifiedPublicApi => {
                self.fetch_github_verified_public_api(url, accept)
            }
        }
    }
}

struct CurlWebTransport {
    github_token: Option<Arc<str>>,
}

impl Default for CurlWebTransport {
    fn default() -> Self {
        Self {
            github_token: github_token_from_env(),
        }
    }
}

#[derive(Clone)]
struct FetchCacheEntry {
    response: FetchResponse,
    stored_at: Instant,
    last_used: Instant,
}

struct CachedWebTransport {
    inner: Arc<dyn WebTransport>,
    entries: Mutex<HashMap<String, FetchCacheEntry>>,
    ttl: Duration,
    capacity: usize,
}

impl CachedWebTransport {
    fn new(inner: Arc<dyn WebTransport>, ttl: Duration, capacity: usize) -> Self {
        Self {
            inner,
            entries: Mutex::new(HashMap::new()),
            ttl,
            capacity: capacity.max(1),
        }
    }
}

struct DeadlineTransport<'a> {
    inner: &'a dyn WebTransport,
    deadline: Instant,
    cancel: tokio_util::sync::CancellationToken,
}

impl WebTransport for DeadlineTransport<'_> {
    fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String> {
        if self.cancel.is_cancelled() {
            return Err("web research was cancelled".to_string());
        }
        if Instant::now() >= self.deadline {
            return Err(format!(
                "the {}-second web research deadline was reached",
                RESEARCH_TOTAL_DEADLINE.as_secs()
            ));
        }
        self.inner.fetch_cancellable(
            url,
            accept,
            None,
            Some(&self.cancel),
            Some(self.deadline),
            ProviderAuthIntent::None,
        )
    }

    fn fetch_github_repository_api(
        &self,
        url: &Url,
        accept: &str,
    ) -> Result<FetchResponse, String> {
        if self.cancel.is_cancelled() {
            return Err("web research was cancelled".to_string());
        }
        if Instant::now() >= self.deadline {
            return Err(format!(
                "the {}-second web research deadline was reached",
                RESEARCH_TOTAL_DEADLINE.as_secs()
            ));
        }
        self.inner.fetch_cancellable(
            url,
            accept,
            None,
            Some(&self.cancel),
            Some(self.deadline),
            ProviderAuthIntent::GithubRepositoryMetadata,
        )
    }

    fn fetch_github_verified_public_api(
        &self,
        url: &Url,
        accept: &str,
    ) -> Result<FetchResponse, String> {
        if self.cancel.is_cancelled() {
            return Err("web research was cancelled".to_string());
        }
        if Instant::now() >= self.deadline {
            return Err(format!(
                "the {}-second web research deadline was reached",
                RESEARCH_TOTAL_DEADLINE.as_secs()
            ));
        }
        self.inner.fetch_cancellable(
            url,
            accept,
            None,
            Some(&self.cancel),
            Some(self.deadline),
            ProviderAuthIntent::GithubVerifiedPublicApi,
        )
    }
}

struct ResearchCancelOnDrop(tokio_util::sync::CancellationToken);

impl Drop for ResearchCancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

struct DnsLookupGate {
    active: std::sync::atomic::AtomicUsize,
    capacity: usize,
}

impl DnsLookupGate {
    const fn new(capacity: usize) -> Self {
        Self {
            active: std::sync::atomic::AtomicUsize::new(0),
            capacity,
        }
    }

    fn try_acquire(&self) -> Option<DnsLookupPermit<'_>> {
        self.active
            .fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |active| (active < self.capacity).then_some(active + 1),
            )
            .ok()
            .map(|_| DnsLookupPermit { gate: self })
    }
}

struct DnsLookupPermit<'a> {
    gate: &'a DnsLookupGate,
}

impl Drop for DnsLookupPermit<'_> {
    fn drop(&mut self) {
        self.gate
            .active
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}

fn dns_lookup_gate() -> &'static DnsLookupGate {
    static GATE: DnsLookupGate = DnsLookupGate::new(MAX_CONCURRENT_DNS_LOOKUPS);
    &GATE
}

fn research_semaphore() -> Arc<tokio::sync::Semaphore> {
    static SEMAPHORE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    SEMAPHORE
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_RESEARCH)))
        .clone()
}

pub(super) fn default_transport() -> Arc<dyn WebTransport> {
    Arc::new(CachedWebTransport::new(
        Arc::new(CurlWebTransport::default()),
        FETCH_CACHE_TTL,
        FETCH_CACHE_CAPACITY,
    ))
}

/// Fetch one public HTTP(S) resource through the hardened Web Auto transport.
///
/// This entry point is intentionally small: callers cannot provide headers,
/// credentials, proxies, ports, redirect policy, or an unbounded timeout/body
/// size.  GET and HEAD are the only accepted methods.  Each redirect is
/// revalidated and re-resolved, and the validated DNS answers remain pinned for
/// the corresponding curl hop.
pub(crate) fn fetch_public_http(method: &str, raw_url: &str) -> Result<PublicHttpResponse, String> {
    let method = PublicHttpMethod::parse(method)?;
    let url = Url::parse(raw_url.trim()).map_err(|error| format!("invalid URL: {error}"))?;
    validate_url_shape(&url)?;
    let deadline = Instant::now() + RESEARCH_TOTAL_DEADLINE;
    let admission_deadline = Instant::now() + RESEARCH_ADMISSION_TIMEOUT;
    let _permit = loop {
        match research_semaphore().try_acquire_owned() {
            Ok(permit) => break permit,
            Err(_) if Instant::now() < admission_deadline && Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => {
                return Err("public web fetch capacity is busy; retry shortly".to_string());
            }
        }
    };
    // Do not use the provider-token-bearing default transport here.  Keeping the
    // token absent in the transport object makes forwarding Camelid/GitHub
    // credentials structurally impossible for native agent fetches.
    let transport = CurlWebTransport { github_token: None };
    let response = transport.fetch_public_method(
        &url,
        "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.1",
        method,
        None,
        Some(deadline),
    )?;
    Ok(PublicHttpResponse {
        status: response.status,
        content_type: response.content_type,
        body: response.body,
        final_url: response.final_url.unwrap_or(url).as_str().to_string(),
    })
}

impl CurlWebTransport {
    fn fetch_public_method(
        &self,
        url: &Url,
        accept: &str,
        method: PublicHttpMethod,
        cancel: Option<&tokio_util::sync::CancellationToken>,
        deadline: Option<Instant>,
    ) -> Result<FetchResponse, String> {
        self.fetch_method_cancellable(
            url,
            accept,
            None,
            cancel,
            deadline,
            ProviderAuthIntent::None,
            method,
        )
    }

    // Keep the security-relevant request attributes explicit at this boundary:
    // provider auth, cancellation, deadline, and method must not be inferred.
    #[expect(clippy::too_many_arguments)]
    fn fetch_method_cancellable(
        &self,
        url: &Url,
        accept: &str,
        etag: Option<&str>,
        cancel: Option<&tokio_util::sync::CancellationToken>,
        deadline: Option<Instant>,
        auth: ProviderAuthIntent,
        method: PublicHttpMethod,
    ) -> Result<FetchResponse, String> {
        let deadline = deadline.unwrap_or_else(|| Instant::now() + RESEARCH_TOTAL_DEADLINE);
        let github_scope = github_auth_scope_for_request(url, accept, auth);
        if auth != ProviderAuthIntent::None && github_scope.is_none() {
            return Err(
                "GitHub provider-auth request was outside its managed repository scope".to_string(),
            );
        }
        let mut current = url.clone();
        for redirects in 0..=MAX_REDIRECTS {
            if cancel.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
                return Err("web research was cancelled".to_string());
            }
            if Instant::now() >= deadline {
                return Err(research_deadline_error());
            }
            validate_url_shape(&current)?;
            validate_github_redirect_scope(&current, github_scope.as_ref())?;
            let addresses = resolve_public_addresses(&current, cancel, Some(deadline))?;
            let conditional = (redirects == 0).then_some(etag).flatten();
            let github_token = github_token_for_request(
                &current,
                accept,
                auth,
                github_scope.as_ref(),
                self.github_token.as_deref(),
            );
            let mut response = curl_single_hop(
                &current,
                accept,
                &addresses,
                conditional,
                github_token,
                cancel,
                Some(deadline),
                method.head_only(),
            )?;
            if response.status == 304 || !(300..400).contains(&response.status) {
                response.final_url = Some(current);
                return Ok(response);
            }
            if redirects == MAX_REDIRECTS {
                return Err(format!("redirect limit ({MAX_REDIRECTS}) exceeded"));
            }
            let location = response
                .location
                .as_deref()
                .ok_or_else(|| format!("HTTP {} redirect omitted Location", response.status))?;
            current = validated_redirect_target(&current, location)?;
        }
        unreachable!("bounded redirect loop always returns")
    }
}

impl WebTransport for CurlWebTransport {
    fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String> {
        self.fetch_conditional(url, accept, None)
    }

    fn fetch_conditional(
        &self,
        url: &Url,
        accept: &str,
        etag: Option<&str>,
    ) -> Result<FetchResponse, String> {
        self.fetch_cancellable(url, accept, etag, None, None, ProviderAuthIntent::None)
    }

    fn fetch_github_repository_api(
        &self,
        url: &Url,
        accept: &str,
    ) -> Result<FetchResponse, String> {
        self.fetch_cancellable(
            url,
            accept,
            None,
            None,
            None,
            ProviderAuthIntent::GithubRepositoryMetadata,
        )
    }

    fn fetch_github_verified_public_api(
        &self,
        url: &Url,
        accept: &str,
    ) -> Result<FetchResponse, String> {
        self.fetch_cancellable(
            url,
            accept,
            None,
            None,
            None,
            ProviderAuthIntent::GithubVerifiedPublicApi,
        )
    }

    fn fetch_cancellable(
        &self,
        url: &Url,
        accept: &str,
        etag: Option<&str>,
        cancel: Option<&tokio_util::sync::CancellationToken>,
        deadline: Option<Instant>,
        auth: ProviderAuthIntent,
    ) -> Result<FetchResponse, String> {
        self.fetch_method_cancellable(
            url,
            accept,
            etag,
            cancel,
            deadline,
            auth,
            PublicHttpMethod::Get,
        )
    }
}

impl WebTransport for CachedWebTransport {
    fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String> {
        self.fetch_cancellable(url, accept, None, None, None, ProviderAuthIntent::None)
    }

    fn fetch_github_repository_api(
        &self,
        url: &Url,
        accept: &str,
    ) -> Result<FetchResponse, String> {
        self.fetch_cancellable(
            url,
            accept,
            None,
            None,
            None,
            ProviderAuthIntent::GithubRepositoryMetadata,
        )
    }

    fn fetch_github_verified_public_api(
        &self,
        url: &Url,
        accept: &str,
    ) -> Result<FetchResponse, String> {
        self.fetch_cancellable(
            url,
            accept,
            None,
            None,
            None,
            ProviderAuthIntent::GithubVerifiedPublicApi,
        )
    }

    fn fetch_cancellable(
        &self,
        url: &Url,
        accept: &str,
        _etag: Option<&str>,
        cancel: Option<&tokio_util::sync::CancellationToken>,
        deadline: Option<Instant>,
        auth: ProviderAuthIntent,
    ) -> Result<FetchResponse, String> {
        if cancel.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
            return Err("web research was cancelled".to_string());
        }
        validate_url_shape(url)?;
        // Metadata can reveal a private repository before the enrichment layer
        // rejects it, so it is never retained. API tree/README bodies enter the
        // cache only through the distinct intent issued after explicit
        // `private:false` verification. Arbitrary linked API URLs stay
        // unauthenticated and bypass this cache.
        if auth == ProviderAuthIntent::GithubRepositoryMetadata
            || (auth == ProviderAuthIntent::None
                && url
                    .host_str()
                    .is_some_and(|host| host.eq_ignore_ascii_case("api.github.com")))
        {
            return self
                .inner
                .fetch_cancellable(url, accept, None, cancel, deadline, auth);
        }
        let key = format!("{auth:?}\n{}\n{accept}", url.as_str());
        let now = Instant::now();
        let cached = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| "web fetch cache lock was poisoned".to_string())?;
            entries.get_mut(&key).map(|entry| {
                entry.last_used = now;
                entry.clone()
            })
        };
        if let Some(entry) = cached.as_ref() {
            if now.duration_since(entry.stored_at) < self.ttl {
                return Ok(entry.response.clone());
            }
        }

        let response = self.inner.fetch_cancellable(
            url,
            accept,
            cached.as_ref().and_then(|entry| {
                entry
                    .response
                    .final_url
                    .as_ref()
                    .is_none_or(|final_url| final_url == url)
                    .then_some(entry.response.etag.as_deref())
                    .flatten()
            }),
            cancel,
            deadline,
            auth,
        )?;
        if response.status == 304 {
            if let Some(mut entry) = cached {
                entry.stored_at = now;
                entry.last_used = now;
                self.entries
                    .lock()
                    .map_err(|_| "web fetch cache lock was poisoned".to_string())?
                    .insert(key, entry.clone());
                return Ok(entry.response);
            }
            return Err("web provider returned HTTP 304 without a cached source".to_string());
        }

        if (200..300).contains(&response.status) {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| "web fetch cache lock was poisoned".to_string())?;
            if entries.len() >= self.capacity && !entries.contains_key(&key) {
                if let Some(oldest) = entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(cache_key, _)| cache_key.clone())
                {
                    entries.remove(&oldest);
                }
            }
            entries.insert(
                key,
                FetchCacheEntry {
                    response: response.clone(),
                    stored_at: now,
                    last_used: now,
                },
            );
        }
        Ok(response)
    }
}

fn github_token_from_env() -> Option<Arc<str>> {
    let token = std::env::var(GITHUB_TOKEN_ENV).ok()?;
    let token = token.trim();
    if token.is_empty()
        || token.len() > 1_024
        || !token
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return None;
    }
    Some(Arc::from(token))
}

fn github_token_for_request<'a>(
    url: &Url,
    accept: &str,
    auth: ProviderAuthIntent,
    scope: Option<&GithubAuthScope>,
    token: Option<&'a str>,
) -> Option<&'a str> {
    let scope = scope?;
    (github_auth_scope_for_request(url, accept, auth).as_ref() == Some(scope))
        .then_some(token)
        .flatten()
}

fn github_auth_scope_for_request(
    url: &Url,
    accept: &str,
    auth: ProviderAuthIntent,
) -> Option<GithubAuthScope> {
    if url.scheme() != "https"
        || !url
            .host_str()
            .is_some_and(|host| host.eq_ignore_ascii_case("api.github.com"))
    {
        return None;
    }
    let segments = url.path_segments()?.collect::<Vec<_>>();
    let managed = match auth {
        ProviderAuthIntent::None => false,
        ProviderAuthIntent::GithubRepositoryMetadata => matches!(
            (segments.as_slice(), accept),
            (["repos", _, _], "application/vnd.github+json")
        ),
        ProviderAuthIntent::GithubVerifiedPublicApi => matches!(
            (segments.as_slice(), accept),
            (
                ["repos", _, _, "git", "trees", _, ..],
                "application/vnd.github+json"
            ) | (["repos", _, _, "readme"], "application/vnd.github.raw+json")
        ),
    };
    if !managed {
        return None;
    }
    let owner = *segments.get(1)?;
    let repo = *segments.get(2)?;
    (valid_github_owner(owner) && valid_github_repo(repo)).then(|| GithubAuthScope {
        owner: owner.to_ascii_lowercase(),
        repo: repo.to_ascii_lowercase(),
    })
}

fn validate_github_redirect_scope(
    url: &Url,
    scope: Option<&GithubAuthScope>,
) -> Result<(), String> {
    let Some(scope) = scope else { return Ok(()) };
    if !url
        .host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("api.github.com"))
    {
        return Ok(());
    }
    let segments = url
        .path_segments()
        .map(|segments| segments.collect::<Vec<_>>())
        .unwrap_or_default();
    let redirected = match segments.as_slice() {
        ["repos", owner, repo, ..] if valid_github_owner(owner) && valid_github_repo(repo) => {
            GithubAuthScope {
                owner: owner.to_ascii_lowercase(),
                repo: repo.to_ascii_lowercase(),
            }
        }
        _ => {
            return Err(
                "GitHub API redirect left the original managed repository scope".to_string(),
            )
        }
    };
    if redirected != *scope {
        return Err(format!(
            "GitHub API redirect changed repository scope from {}/{} to {}/{}",
            scope.owner, scope.repo, redirected.owner, redirected.repo
        ));
    }
    Ok(())
}

pub(super) async fn handler(
    State(state): State<AppState>,
    payload: Result<Json<WebResearchRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_web_research_json",
                format!("invalid /api/web/research JSON request: {error}"),
                Some("body"),
            )
        }
    };
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_web_research_prompt",
            "prompt must not be empty".to_string(),
            Some("prompt"),
        );
    }
    if prompt.len() > MAX_PROMPT_BYTES {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "web_research_prompt_too_large",
            format!("prompt exceeds the {MAX_PROMPT_BYTES}-byte web-research limit"),
            Some("prompt"),
        );
    }

    let decision = classify_prompt(prompt);
    if matches!(decision, ResearchDecision::Skip) {
        return Json(skipped_response()).into_response();
    }
    let reason = decision.reason();
    let permit = match tokio::time::timeout(
        RESEARCH_ADMISSION_TIMEOUT,
        research_semaphore().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) | Err(_) => {
            return Json(WebResearchResponse {
                status: "failed",
                triggered: true,
                reason,
                query: None,
                sources: Vec::new(),
                warnings: vec![
                    "Web research is busy; local chat can continue without web sources."
                        .to_string(),
                ],
            })
            .into_response()
        }
    };

    let prompt = prompt.to_string();
    let transport = state.web_research_transport.clone();
    let cancel = tokio_util::sync::CancellationToken::new();
    let cancel_worker = cancel.clone();
    // If Axum drops this handler because the browser aborts, the blocking
    // transport sees cancellation, terminates curl, and stops before the next
    // fetch. The semaphore permit remains owned by the worker until it exits.
    let _cancel_on_drop = ResearchCancelOnDrop(cancel);
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let bounded = DeadlineTransport {
            inner: transport.as_ref(),
            deadline: Instant::now() + RESEARCH_TOTAL_DEADLINE,
            cancel: cancel_worker,
        };
        research_prompt(&prompt, &bounded)
    })
    .await
    .unwrap_or_else(|error| WebResearchResponse {
        status: "failed",
        triggered: true,
        reason,
        query: None,
        sources: Vec::new(),
        warnings: vec![format!("web research worker stopped unexpectedly: {error}")],
    });
    Json(result).into_response()
}

#[derive(Debug)]
enum ResearchDecision {
    Research {
        reason: &'static str,
        urls: Vec<Url>,
        omitted: bool,
        query: Option<String>,
    },
    Skip,
}

impl ResearchDecision {
    fn reason(&self) -> &'static str {
        match self {
            Self::Research { reason, .. } => reason,
            Self::Skip => "not_needed",
        }
    }
}

fn skipped_response() -> WebResearchResponse {
    WebResearchResponse {
        status: "skipped",
        triggered: false,
        reason: "not_needed",
        query: None,
        sources: Vec::new(),
        warnings: Vec::new(),
    }
}

fn research_prompt(prompt: &str, transport: &dyn WebTransport) -> WebResearchResponse {
    match classify_prompt(prompt) {
        ResearchDecision::Skip => skipped_response(),
        ResearchDecision::Research {
            reason,
            urls,
            omitted,
            query,
        } => {
            let mut warnings = Vec::new();
            if omitted {
                warnings.push(format!(
                    "Only the first {MAX_DIRECT_URLS} distinct URLs were researched."
                ));
            }
            let mut sources = Vec::new();
            for url in urls {
                fetch_source(transport, &url, None, prompt, &mut sources, &mut warnings);
            }
            if let Some(search_query) = query.as_deref() {
                let supplemental = search_and_fetch(transport, reason, search_query.to_string());
                warnings.extend(supplemental.warnings);
                let mut seen = sources
                    .iter()
                    .map(|source| source.url.clone())
                    .collect::<HashSet<_>>();
                sources.extend(
                    supplemental
                        .sources
                        .into_iter()
                        .filter(|source| seen.insert(source.url.clone())),
                );
            }
            finish_response(reason, query, sources, warnings)
        }
    }
}

fn finish_response(
    reason: &'static str,
    query: Option<String>,
    mut sources: Vec<WebResearchSource>,
    mut warnings: Vec<String>,
) -> WebResearchResponse {
    let mut seen = HashSet::new();
    sources.retain(|source| seen.insert(source.url.clone()));
    if sources.len() > MAX_TOTAL_SOURCES {
        sources.truncate(MAX_TOTAL_SOURCES);
        warnings.push(format!(
            "Only the first {MAX_TOTAL_SOURCES} distinct sources were retained."
        ));
    }
    for (index, source) in sources.iter_mut().enumerate() {
        source.id = index + 1;
    }
    let status = if sources.is_empty() {
        "failed"
    } else if warnings.is_empty() {
        "complete"
    } else {
        "partial"
    };
    WebResearchResponse {
        status,
        triggered: true,
        reason,
        query,
        sources,
        warnings,
    }
}

fn classify_prompt(prompt: &str) -> ResearchDecision {
    let (mut urls, mut omitted) = extract_prompt_urls(prompt);
    let lower = prompt
        .to_lowercase()
        .replace('’', "'")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if has_global_web_veto(&lower) {
        return ResearchDecision::Skip;
    }
    let links_vetoed = has_link_read_veto(&lower);
    if links_vetoed {
        urls.clear();
        omitted = false;
    }
    let explicit = [
        "search the web",
        "search web",
        "web search",
        "search online",
        "look this up",
        "look it up",
        "look up",
        "look up online",
        "browse the web",
        "browse web",
        "browse the internet",
        "browse internet",
        "browse online",
        "use the internet",
        "find online",
        "find it online",
        "find this online",
        "find that online",
        "research online",
        "research this",
        "research that",
        "research on the web",
        "read the linked",
        "read linked",
        "read the website",
        "read website",
        "read the web page",
        "read web page",
        "read the github",
        "read github",
        "read the documentation",
        "read documentation",
        "read the docs",
        "read docs",
        "check the web",
        "check web",
        "check the internet",
        "check internet",
        "check the website",
        "check website",
        "check the github",
        "check github",
        "check the documentation",
        "check documentation",
        "check the docs",
        "check docs",
        "cite web sources",
        "cite online sources",
        "cite your sources",
        "cite your web sources",
        "cite your online sources",
        "cite sources",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    // With no URL, direct-read language still asks Camelid to locate a page.
    // With an explicit URL, only unmistakable broader-search language should
    // cause a second public-web query; "read/check/cite this link" is one leg.
    let supplemental = [
        "search the web",
        "search web",
        "web search",
        "search online",
        "browse the web",
        "browse web",
        "browse the internet",
        "browse internet",
        "browse online",
        "use the internet",
        "find online",
        "find it online",
        "find this online",
        "find that online",
        "research online",
        "research on the web",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
        || has_linked_broader_research_cue(&lower);
    let current = [
        "latest version",
        "latest versions",
        "latest release",
        "latest releases",
        "latest news",
        "latest price",
        "latest prices",
        "latest schedule",
        "latest schedules",
        "latest score",
        "latest scores",
        "latest documentation",
        "latest docs",
        "latest specification",
        "latest specifications",
        "latest status",
        "latest statuses",
        "newest version",
        "newest versions",
        "newest release",
        "newest releases",
        "newest news",
        "newest price",
        "newest prices",
        "newest schedule",
        "newest schedules",
        "newest score",
        "newest scores",
        "newest documentation",
        "newest docs",
        "newest specification",
        "newest specifications",
        "newest status",
        "newest statuses",
        "most recent",
        "current version",
        "current versions",
        "current release",
        "current releases",
        "current status",
        "current statuses",
        "current price",
        "current prices",
        "current weather",
        "current schedule",
        "current schedules",
        "current score",
        "current scores",
        "current documentation",
        "current docs",
        "current ceo",
        "current ceos",
        "currently available",
        "up-to-date",
        "up to date",
        "as of today",
        "as of now",
        "today's news",
        "today’s news",
        "today's weather",
        "today’s weather",
        "today's price",
        "today’s price",
        "today's schedule",
        "today’s schedule",
        "today's score",
        "today’s score",
        "recent news",
        "recent events",
        "recent developments",
        "recent changes",
        "recent updates",
        "recent releases",
        "news today",
        "weather today",
        "price today",
        "schedule today",
        "score today",
        "news right now",
        "weather right now",
        "price right now",
        "schedule right now",
        "score right now",
        "news now",
        "weather now",
        "price now",
        "schedule now",
        "score now",
        "current officeholder",
        "current officeholders",
        "who is the current",
        "who is current",
        "what's new in",
        "what's new with",
        "what is new in",
        "what is new with",
        "what is the latest",
        "what's the latest",
        "what is the current",
        "what's the current",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
        || contains_as_of_year(&lower);

    if !urls.is_empty() || explicit || current {
        let query_requested =
            current || supplemental || (!links_vetoed && urls.is_empty() && explicit);
        let query = query_requested.then(|| search_query_from_prompt(prompt));
        let reason = if !urls.is_empty() && query.is_some() {
            "embedded_urls_and_search"
        } else if !urls.is_empty() {
            "embedded_urls"
        } else if explicit {
            "explicit_search"
        } else {
            "current_info"
        };
        return ResearchDecision::Research {
            reason,
            urls,
            omitted,
            query,
        };
    }
    ResearchDecision::Skip
}

fn has_global_web_veto(lower: &str) -> bool {
    let no_source_phrases = [
        "no web search",
        "no web research",
        "no web access",
        "no web browsing",
        "no internet search",
        "no internet research",
        "no internet access",
        "no internet browsing",
        "no online sources",
        "no outside sources",
        "no external sources",
        "no network access",
    ];
    if no_source_phrases
        .iter()
        .any(|phrase| contains_no_source_directive(lower, phrase))
        || ["stay offline", "remain offline"]
            .iter()
            .any(|phrase| directive_clause_starts_with(lower, phrase))
        || [
            "without online sources",
            "without outside sources",
            "without external sources",
            "without network access",
        ]
        .iter()
        .any(|phrase| directive_clause_starts_with(lower, phrase))
    {
        return true;
    }

    // Keep these newer forms clause-directed. A product prompt can quote UI
    // copy such as "do not connect to the internet" or "answer without
    // external sources" while separately asking Camelid to research examples;
    // matching only an imperative clause avoids turning that descriptive copy
    // into a privacy instruction for the whole turn.
    if directive_action_targets(
        lower,
        &[
            "do not connect to ",
            "don't connect to ",
            "dont connect to ",
            "never connect to ",
            "without connecting to ",
        ],
        &[
            "web",
            "the web",
            "internet",
            "the internet",
            "network",
            "the network",
        ],
    ) || directive_action_targets(
        lower,
        &[
            "do not search ",
            "don't search ",
            "dont search ",
            "never search ",
            "do not browse ",
            "don't browse ",
            "dont browse ",
            "never browse ",
            "do not research ",
            "don't research ",
            "dont research ",
            "never research ",
            "without searching ",
            "without browsing ",
            "without researching ",
        ],
        &["online"],
    ) || directive_action_targets(
        lower,
        &["answer without ", "respond without "],
        &[
            "online sources",
            "outside sources",
            "external sources",
            "web sources",
            "online material",
            "outside material",
            "external material",
            "web material",
            "online content",
            "outside content",
            "external content",
            "web content",
        ],
    ) {
        return true;
    }

    let direct_prefixes = ["do not ", "don't ", "dont ", "never "];
    let direct_actions = ["use", "access", "browse", "search", "research", "consult"];
    let progressive_actions = [
        "using",
        "accessing",
        "browsing",
        "searching",
        "researching",
        "consulting",
    ];
    let targets = [
        "web",
        "the web",
        "internet",
        "the internet",
        "online sources",
        "outside sources",
        "external sources",
        "online sites",
        "outside sites",
        "external sites",
        "online websites",
        "outside websites",
        "external websites",
        "network",
        "the network",
    ];
    if direct_prefixes
        .iter()
        .any(|prefix| veto_action_phrase(lower, prefix, &direct_actions, &targets))
    {
        return true;
    }
    veto_action_phrase(lower, "without ", &progressive_actions, &targets)
}

fn directive_action_targets(lower: &str, prefixes: &[&str], targets: &[&str]) -> bool {
    lower.split([';', '.', '!', '?', '\n']).any(|clause| {
        let clause = clause
            .trim()
            .strip_prefix("please ")
            .unwrap_or_else(|| clause.trim());
        prefixes.iter().any(|prefix| {
            clause.strip_prefix(prefix).is_some_and(|after| {
                targets.iter().any(|target| {
                    after.strip_prefix(target).is_some_and(|tail| {
                        tail.chars()
                            .next()
                            .is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_')
                    })
                })
            })
        })
    })
}

fn has_linked_broader_research_cue(lower: &str) -> bool {
    const ACTIONS: &[&str] = &["look up", "research"];

    // Match an explicitly coordinated second research leg. Keeping the
    // coordinator attached to the action avoids treating a link-only request
    // such as "Look up https://example.com and summarize it" as a web search.
    for lead in ["also ", "additionally ", "in addition ", "in addition, "] {
        if ACTIONS
            .iter()
            .any(|action| contains_word_bounded(lower, &format!("{lead}{action}")))
        {
            return true;
        }
    }
    for coordinator in ["and ", "then "] {
        for courtesy in ["", "please "] {
            for additive in ["", "also "] {
                if ACTIONS.iter().any(|action| {
                    contains_word_bounded(
                        lower,
                        &format!("{coordinator}{courtesy}{additive}{action}"),
                    )
                }) {
                    return true;
                }
            }
        }
    }

    // Also accept a bounded trailing coordination marker, mirroring the Web
    // UI's 180-character, single-clause window.
    ACTIONS.iter().any(|action| {
        lower.match_indices(action).any(|(index, _)| {
            if !word_bounds_at(lower, index, action.len()) {
                return false;
            }
            let tail = &lower[index + action.len()..];
            let bounded = tail
                .chars()
                .take(180)
                .take_while(|ch| !matches!(ch, '.' | '!' | '?'))
                .collect::<String>();
            ["too", "also", "as well"]
                .iter()
                .any(|marker| contains_word_bounded(&bounded, marker))
        })
    })
}

fn contains_word_bounded(value: &str, phrase: &str) -> bool {
    value
        .match_indices(phrase)
        .any(|(index, _)| word_bounds_at(value, index, phrase.len()))
}

fn word_bounds_at(value: &str, index: usize, length: usize) -> bool {
    let is_word = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';
    value[..index]
        .chars()
        .next_back()
        .is_none_or(|ch| !is_word(ch))
        && value[index + length..]
            .chars()
            .next()
            .is_none_or(|ch| !is_word(ch))
}

fn has_link_read_veto(lower: &str) -> bool {
    const ACTIONS: &[&str] = &["open", "fetch", "follow", "visit", "read", "access", "use"];
    const TARGETS: &[&str] = &[
        "link",
        "links",
        "url",
        "urls",
        "this link",
        "this url",
        "that link",
        "that url",
        "the link",
        "the links",
        "the url",
        "the urls",
        "these links",
        "these urls",
        "those links",
        "those urls",
        "any link",
        "any links",
        "any url",
        "any urls",
        "web page",
        "the web page",
        "website",
        "the website",
        "linked page",
        "the linked page",
        "linked docs",
        "the linked docs",
    ];
    ["do not ", "don't ", "dont ", "never "]
        .iter()
        .any(|prefix| {
            veto_action_phrase(lower, prefix, ACTIONS, TARGETS)
                || veto_action_phrase(lower, prefix, ACTIONS, &["http://", "https://"])
        })
}

fn veto_action_phrase(lower: &str, prefix: &str, actions: &[&str], targets: &[&str]) -> bool {
    actions.iter().any(|first| {
        targets
            .iter()
            .any(|target| contains_veto_target(lower, &format!("{prefix}{first} {target}"), target))
            || ["or", "and"].iter().any(|conjunction| {
                actions.iter().any(|second| {
                    targets.iter().any(|target| {
                        contains_veto_target(
                            lower,
                            &format!("{prefix}{first} {conjunction} {second} {target}"),
                            target,
                        )
                    })
                })
            })
    })
}

fn contains_veto_target(lower: &str, phrase: &str, target: &str) -> bool {
    lower.match_indices(phrase).any(|(index, _)| {
        let after = &lower[index + phrase.len()..];
        if target.ends_with("://") {
            return true;
        }
        let boundary = after
            .chars()
            .next()
            .is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_');
        if !boundary {
            return false;
        }
        let link_specific_web_target = matches!(target, "web" | "the web")
            && [" page", " pages", " site", " sites"]
                .iter()
                .any(|suffix| after.starts_with(suffix));
        !link_specific_web_target
    })
}

fn contains_no_source_directive(lower: &str, phrase: &str) -> bool {
    lower.match_indices(phrase).any(|(index, _)| {
        let before = lower[..index].trim_end();
        if before.is_empty() {
            return true;
        }
        let clause = before
            .rsplit([';', '.', '!', '?', '\n'])
            .next()
            .unwrap_or_default()
            .trim()
            .trim_end_matches(',')
            .trim();
        matches!(
            clause,
            "use"
                | "using"
                | "with"
                | "please"
                | "please use"
                | "answer with"
                | "respond with"
                | "work with"
                | "continue with"
        )
    })
}

fn directive_clause_starts_with(lower: &str, phrase: &str) -> bool {
    lower.split([';', '.', '!', '?', '\n']).any(|clause| {
        let clause = clause
            .trim()
            .strip_prefix("please ")
            .unwrap_or_else(|| clause.trim());
        clause == phrase
            || clause
                .strip_prefix(phrase)
                .is_some_and(|after| after.starts_with([',', ':']))
    })
}

fn search_query_from_prompt(prompt: &str) -> String {
    let mut offsets = Vec::new();
    for scheme in ["https://", "http://"] {
        offsets.extend(prompt.match_indices(scheme).map(|(start, _)| start));
    }
    offsets.sort_unstable();
    offsets.dedup();
    let mut query = String::with_capacity(prompt.len());
    let mut cursor = 0usize;
    for start in offsets {
        if start < cursor {
            continue;
        }
        let tail = &prompt[start..];
        let end = start + prompt_url_candidate_end(tail);
        query.push_str(&prompt[cursor..start]);
        query.push(' ');
        cursor = end;
    }
    query.push_str(&prompt[cursor..]);
    let compact = query.split_whitespace().collect::<Vec<_>>().join(" ");
    clip_chars(compact.trim(), MAX_SEARCH_QUERY_CHARS, false)
}

fn contains_as_of_year(text: &str) -> bool {
    text.match_indices("as of ").any(|(index, marker)| {
        let tail = &text[index + marker.len()..];
        let year = tail.as_bytes().get(..4);
        year.is_some_and(|digits| digits.iter().all(u8::is_ascii_digit))
            && tail
                .as_bytes()
                .get(4)
                .is_none_or(|next| !next.is_ascii_digit())
    })
}

fn extract_prompt_urls(prompt: &str) -> (Vec<Url>, bool) {
    let mut offsets = Vec::new();
    for scheme in ["https://", "http://"] {
        offsets.extend(prompt.match_indices(scheme).map(|(offset, _)| offset));
    }
    offsets.sort_unstable();

    let mut seen = HashSet::new();
    let mut urls = Vec::new();
    let mut omitted = false;
    for offset in offsets {
        let tail = &prompt[offset..];
        let end = prompt_url_candidate_end(tail);
        let mut raw = tail[..end].trim_end_matches(['.', ',', ';', ':', '!', '?', '}']);
        // A closing parenthesis is syntax only when this URL began immediately
        // after an opening Markdown/prose wrapper. Legal GitHub refs and file
        // names may themselves end in `)` and must otherwise remain untouched.
        if prompt[..offset].ends_with('(') {
            raw = raw.strip_suffix(')').unwrap_or(raw);
        }
        // Markdown escapes `)` inside a link destination as `\)`. Decode only
        // the known malformed numbered-list paste shape at a GitHub repo root;
        // a general backslash rewrite could change legitimate URL paths.
        let repaired_raw = repair_markdown_escaped_github_repo_candidate(raw);
        let parse_raw = repaired_raw.as_deref().unwrap_or(raw);
        let Ok(parsed) = Url::parse(parse_raw) else {
            continue;
        };
        let normalized = canonical_prompt_url(parsed);
        let key = normalized.as_str().to_string();
        if !seen.insert(key) {
            continue;
        }
        if urls.len() == MAX_DIRECT_URLS {
            omitted = true;
            continue;
        }
        urls.push(normalized);
    }
    (urls, omitted)
}

fn repair_markdown_escaped_github_repo_candidate(raw: &str) -> Option<String> {
    let (scheme, authority_and_path) = raw
        .strip_prefix("https://")
        .map(|tail| ("https", tail))
        .or_else(|| raw.strip_prefix("http://").map(|tail| ("http", tail)))?;
    let (authority, path) = authority_and_path.split_once('/')?;
    if !authority.eq_ignore_ascii_case("github.com") {
        return None;
    }

    let mut segments = path.split('/');
    let owner = segments.next()?;
    let pasted_repo = segments.next()?;
    if segments.next().is_some() {
        return None;
    }
    let close = pasted_repo.rfind("\\)")?;
    let repo = &pasted_repo[..close];
    let suffix = &pasted_repo[close + 2..];
    if suffix.is_empty()
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
        || !valid_github_owner(owner)
        || !valid_github_repo(repo)
    {
        return None;
    }

    Some(format!("{scheme}://github.com/{owner}/{repo}){suffix}"))
}

fn prompt_url_candidate_end(candidate: &str) -> usize {
    let ipv6_host_close = ipv6_host_closing_bracket(candidate);
    candidate
        .char_indices()
        .skip(1)
        .find_map(|(index, ch)| {
            (ch.is_whitespace()
                || matches!(ch, '<' | '>' | '"' | '\'' | '`')
                || (ch == ']' && Some(index) != ipv6_host_close))
                .then_some(index)
        })
        .unwrap_or(candidate.len())
}

fn ipv6_host_closing_bracket(candidate: &str) -> Option<usize> {
    let authority = candidate
        .strip_prefix("https://")
        .or_else(|| candidate.strip_prefix("http://"))?;
    if !authority.starts_with('[') {
        return None;
    }
    let relative = authority.find(']')?;
    Some(candidate.len() - authority.len() + relative)
}

fn canonical_prompt_url(mut url: Url) -> Url {
    url.set_fragment(None);
    repair_numbered_github_repo_paste(&mut url);
    // Never normalize away a security-relevant component. It must survive to
    // `validate_url_shape`, where it is rejected before any transport call.
    if url.username().is_empty() && url.password().is_none() && url.port().is_none() {
        if let Some(repo) = github_repo(&url) {
            return repo.canonical_url;
        }
    }
    url
}

fn repair_numbered_github_repo_paste(url: &mut Url) {
    if !url
        .host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("github.com"))
    {
        return;
    }
    let segments = match url.path_segments() {
        Some(segments) => segments
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>(),
        None => return,
    };
    let [owner, pasted_repo] = segments.as_slice() else {
        return;
    };
    let Some(close) = pasted_repo.rfind(')') else {
        return;
    };
    let suffix = &pasted_repo[close + 1..];
    let repo = &pasted_repo[..close];
    if suffix.is_empty()
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
        || !valid_github_owner(owner)
        || !valid_github_repo(repo)
    {
        return;
    }
    let owner = owner.to_string();
    let repo = repo.to_string();
    if let Ok(mut path) = url.path_segments_mut() {
        path.clear().extend([owner.as_str(), repo.as_str()]);
    }
}

#[derive(Debug, Clone)]
struct GithubRepo {
    owner: String,
    repo: String,
    canonical_url: Url,
    view: Option<GithubView>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GithubViewKind {
    Tree,
    Blob,
}

#[derive(Debug, Clone)]
struct GithubView {
    kind: GithubViewKind,
    segments: Vec<String>,
}

fn github_repo(url: &Url) -> Option<GithubRepo> {
    if !url
        .host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("github.com"))
    {
        return None;
    }
    let mut segments = url.path_segments()?.filter(|part| !part.is_empty());
    let owner = segments.next()?;
    let raw_repo = segments.next()?;
    let repo = raw_repo.strip_suffix(".git").unwrap_or(raw_repo);
    if !valid_github_owner(owner) || !valid_github_repo(repo) {
        return None;
    }
    let action = segments.next();
    let remainder = segments
        .filter(|segment| !segment.is_empty() && *segment != "." && *segment != "..")
        .map(str::to_string)
        .collect::<Vec<_>>();
    let view = match action {
        Some("tree") if !remainder.is_empty() => Some(GithubView {
            kind: GithubViewKind::Tree,
            segments: remainder,
        }),
        Some("blob") if remainder.len() >= 2 => Some(GithubView {
            kind: GithubViewKind::Blob,
            segments: remainder,
        }),
        _ => None,
    };
    if action.is_some() && view.is_none() {
        // Issues, pulls, releases, actions, and other GitHub pages are ordinary
        // web pages. Only repository roots and explicit tree/blob views use the
        // repository enrichment path.
        return None;
    }
    let mut canonical_url = Url::parse("https://github.com/").ok()?;
    {
        let mut path = canonical_url.path_segments_mut().ok()?;
        path.extend([owner, repo]);
        if let Some(view) = view.as_ref() {
            path.push(match view.kind {
                GithubViewKind::Tree => "tree",
                GithubViewKind::Blob => "blob",
            });
            path.extend(view.segments.iter().map(String::as_str));
        }
    }
    Some(GithubRepo {
        owner: owner.to_string(),
        repo: repo.to_string(),
        canonical_url,
        view,
    })
}

fn valid_github_owner(value: &str) -> bool {
    (1..=39).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        && value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && value
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
        && !value.contains("--")
}

fn valid_github_repo(value: &str) -> bool {
    (1..=100).contains(&value.len())
        && !matches!(value, "." | "..")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn fetch_source(
    transport: &dyn WebTransport,
    url: &Url,
    title_hint: Option<&str>,
    query: &str,
    sources: &mut Vec<WebResearchSource>,
    warnings: &mut Vec<String>,
) {
    if let Err(error) = validate_url_shape(url) {
        warnings.push(format!("Blocked web URL: {error}"));
        return;
    }
    if let Some(repo) = github_repo(url) {
        if let Some(source) = fetch_github_repo(transport, &repo, query, warnings) {
            sources.push(source);
        }
        return;
    }
    let response = match transport.fetch(url, "text/html,text/plain,application/json;q=0.8") {
        Ok(response) => response,
        Err(error) => {
            warnings.push(format!("Could not fetch {}: {error}", display_url(url)));
            return;
        }
    };
    if !(200..300).contains(&response.status) {
        warnings.push(format!(
            "Could not fetch {}: HTTP {}",
            display_url(url),
            response.status
        ));
        return;
    }
    let Some(excerpt) = response_excerpt(&response) else {
        warnings.push(format!(
            "Could not use {}: unsupported or empty content",
            display_url(url)
        ));
        return;
    };
    let provenance_url = response.final_url.as_ref().unwrap_or(url);
    let text = String::from_utf8_lossy(&response.body);
    let title = title_hint
        .filter(|title| !title.trim().is_empty())
        .map(str::to_string)
        .or_else(|| extract_html_title(&text))
        .unwrap_or_else(|| display_url(provenance_url));
    sources.push(WebResearchSource {
        id: 0,
        title: clip_chars(&title, 240, false),
        url: provenance_url.as_str().to_string(),
        chunks: vec![WebResearchChunk {
            path: useful_url_path(provenance_url),
            url: Some(provenance_url.as_str().to_string()),
            text: excerpt.clone(),
        }],
        excerpt,
    });
}

fn useful_url_path(url: &Url) -> Option<String> {
    let path = url.path().trim();
    (!path.is_empty() && path != "/").then(|| path.to_string())
}

#[derive(Debug, Deserialize)]
struct GithubMetadata {
    default_branch: Option<String>,
    private: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct GithubTree {
    #[serde(default)]
    tree: Vec<GithubTreeEntry>,
    #[serde(default)]
    truncated: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct GithubTreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    sha: Option<String>,
}

#[derive(Debug, Clone)]
struct GithubReadme {
    path: String,
    url: Url,
    text: String,
}

enum GithubTreeCandidateResult {
    Found(GithubTree),
    ProviderUnavailable,
    NotFound,
}

struct GithubTreeFrontier {
    treeish: String,
    prefix: Option<String>,
    depth: usize,
}

fn fetch_github_tree_candidate(
    transport: &dyn WebTransport,
    repo: &GithubRepo,
    candidate: &GithubRefCandidate,
    query_terms: &[String],
    pages_remaining: &mut usize,
    warnings: &mut Vec<String>,
) -> GithubTreeCandidateResult {
    if *pages_remaining == 0 {
        return GithubTreeCandidateResult::NotFound;
    }
    *pages_remaining -= 1;
    let label = format!("{}/{}", repo.owner, repo.repo);
    let tree_url = github_api_url(
        &repo.owner,
        &repo.repo,
        &["git", "trees", &candidate.reference],
    );
    let mut truncated_tree = None;
    match transport.fetch_github_verified_public_api(&tree_url, "application/vnd.github+json") {
        Ok(response) if (200..300).contains(&response.status) => {
            match serde_json::from_slice::<GithubTree>(&response.body) {
                Ok(tree) if !tree.truncated => return GithubTreeCandidateResult::Found(tree),
                Ok(tree) => {
                    warnings.push(format!(
                        "{label}: GitHub returned a truncated recursive tree; using bounded tree traversal"
                    ));
                    truncated_tree = Some(tree);
                }
                Err(error) => warnings.push(format!(
                    "{label}: GitHub recursive file tree was not valid JSON ({error}); using bounded tree traversal"
                )),
            }
        }
        Ok(response) if matches!(response.status, 401 | 403 | 429) => {
            warnings.push(format!(
                "{label}: GitHub API file tree returned HTTP {}; trying public raw/HTML content",
                response.status
            ));
            return GithubTreeCandidateResult::ProviderUnavailable;
        }
        Ok(response) if matches!(response.status, 404 | 422) => {
            return GithubTreeCandidateResult::NotFound;
        }
        Ok(response) => warnings.push(format!(
            "{label}: GitHub recursive file tree returned HTTP {}; trying bounded tree traversal",
            response.status
        )),
        Err(error) => warnings.push(format!(
            "{label}: GitHub recursive file tree unavailable ({error}); trying bounded tree traversal"
        )),
    }

    match fetch_bounded_github_tree(
        transport,
        repo,
        &candidate.reference,
        candidate.path.as_deref(),
        query_terms,
        pages_remaining,
        warnings,
    ) {
        GithubTreeCandidateResult::NotFound => truncated_tree
            .map(GithubTreeCandidateResult::Found)
            .unwrap_or(GithubTreeCandidateResult::NotFound),
        result => result,
    }
}

fn fetch_bounded_github_tree(
    transport: &dyn WebTransport,
    repo: &GithubRepo,
    reference: &str,
    path_prefix: Option<&str>,
    query_terms: &[String],
    pages_remaining: &mut usize,
    warnings: &mut Vec<String>,
) -> GithubTreeCandidateResult {
    let label = format!("{}/{}", repo.owner, repo.repo);
    let target = path_prefix
        .map(|path| path.trim_matches('/'))
        .filter(|path| !path.is_empty());
    let mut pending = vec![GithubTreeFrontier {
        treeish: reference.to_string(),
        prefix: None,
        depth: 0,
    }];
    let mut seen_trees = HashSet::new();
    let mut entries = Vec::new();
    let mut successful_pages = 0usize;
    let mut inspected_entries = 0usize;
    let mut incomplete = false;
    let mut entry_limit_exceeded = false;

    while *pages_remaining > 0 && !pending.is_empty() {
        let next_index = pending
            .iter()
            .enumerate()
            .max_by_key(|(_, item)| {
                github_api_directory_score(item.prefix.as_deref(), target, query_terms)
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        let item = pending.swap_remove(next_index);
        if !seen_trees.insert((item.treeish.clone(), item.prefix.clone())) {
            continue;
        }
        *pages_remaining -= 1;
        let tree_url = github_nonrecursive_tree_api_url(&repo.owner, &repo.repo, &item.treeish);
        let response = match transport
            .fetch_github_verified_public_api(&tree_url, "application/vnd.github+json")
        {
            Ok(response) if (200..300).contains(&response.status) => response,
            Ok(response) if matches!(response.status, 401 | 403 | 429) => {
                warnings.push(format!(
                    "{label}: bounded GitHub tree traversal returned HTTP {}",
                    response.status
                ));
                if entries.is_empty() {
                    return GithubTreeCandidateResult::ProviderUnavailable;
                }
                incomplete = true;
                break;
            }
            Ok(response) => {
                incomplete = true;
                warnings.push(format!(
                    "{label}: bounded GitHub tree page returned HTTP {}",
                    response.status
                ));
                continue;
            }
            Err(error) => {
                incomplete = true;
                warnings.push(format!(
                    "{label}: bounded GitHub tree page was unavailable ({error})"
                ));
                continue;
            }
        };
        let page = match serde_json::from_slice::<GithubTree>(&response.body) {
            Ok(page) => page,
            Err(error) => {
                incomplete = true;
                warnings.push(format!(
                    "{label}: bounded GitHub tree page was not valid JSON ({error})"
                ));
                continue;
            }
        };
        successful_pages += 1;
        incomplete |= page.truncated;

        for entry in page.tree {
            if inspected_entries >= MAX_GITHUB_API_TREE_ENTRIES {
                incomplete = true;
                entry_limit_exceeded = true;
                break;
            }
            inspected_entries += 1;
            if !valid_github_tree_path(&entry.path) {
                continue;
            }
            let full_path = item
                .prefix
                .as_deref()
                .map(|prefix| format!("{prefix}/{}", entry.path))
                .unwrap_or_else(|| entry.path.clone());
            if !valid_github_tree_path(&full_path) {
                continue;
            }
            match entry.kind.as_str() {
                "blob" => entries.push(GithubTreeEntry {
                    path: full_path,
                    kind: entry.kind,
                    size: entry.size,
                    sha: entry.sha,
                }),
                "tree" if item.depth < MAX_GITHUB_API_TREE_DEPTH => {
                    if let Some(sha) = entry.sha.filter(|sha| is_github_commit_ref(sha)) {
                        pending.push(GithubTreeFrontier {
                            treeish: sha,
                            prefix: Some(full_path),
                            depth: item.depth + 1,
                        });
                    }
                }
                "tree" => incomplete = true,
                _ => {}
            }
        }
        if entry_limit_exceeded {
            break;
        }
    }

    if successful_pages == 0 {
        return GithubTreeCandidateResult::NotFound;
    }
    if !pending.is_empty() {
        incomplete = true;
    }
    if incomplete {
        warnings.push(format!(
            "{label}: bounded GitHub tree traversal stopped at its page, depth, or entry limit"
        ));
    }
    GithubTreeCandidateResult::Found(GithubTree {
        tree: entries,
        truncated: incomplete,
    })
}

fn github_api_directory_score(
    path: Option<&str>,
    target: Option<&str>,
    query_terms: &[String],
) -> i64 {
    let Some(path) = path else { return i64::MAX };
    let path_score = github_directory_score(Some(path), query_terms) as i64;
    let target_score = target.map_or(0, |target| {
        if path == target
            || target
                .strip_prefix(path)
                .is_some_and(|suffix| suffix.starts_with('/'))
        {
            1_000_000
        } else if path
            .strip_prefix(target)
            .is_some_and(|suffix| suffix.starts_with('/'))
        {
            500_000
        } else {
            -100_000
        }
    });
    target_score + path_score
}

fn valid_github_tree_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 4_096
        && !path.starts_with('/')
        && !path.chars().any(char::is_control)
        && path
            .split('/')
            .all(|segment| !segment.is_empty() && !matches!(segment, "." | ".."))
}

fn fetch_github_repo(
    transport: &dyn WebTransport,
    repo: &GithubRepo,
    query: &str,
    warnings: &mut Vec<String>,
) -> Option<WebResearchSource> {
    let warning_start = warnings.len();
    let label = format!("{}/{}", repo.owner, repo.repo);
    let query_terms = query_terms(query);
    let overview_only = repo.view.is_none() && is_github_repository_overview_request(query, repo);
    // Downstream authenticated reads are fail-closed until GitHub explicitly
    // confirms that this repository is public. A malformed/missing `private`
    // field must never turn the provider token into a private-content reader.
    let mut api_available = false;
    let mut default_branch = None;
    let metadata_url = github_api_url(&repo.owner, &repo.repo, &[]);
    match transport.fetch_github_repository_api(&metadata_url, "application/vnd.github+json") {
        Ok(response) if (200..300).contains(&response.status) => {
            match serde_json::from_slice::<GithubMetadata>(&response.body) {
                Ok(metadata) if metadata.private => {
                    warnings.push(format!(
                        "{label}: private GitHub repositories are outside public-web research"
                    ));
                    return None;
                }
                Ok(metadata) => {
                    api_available = true;
                    default_branch = metadata
                        .default_branch
                        .filter(|branch| !branch.trim().is_empty());
                }
                Err(error) => warnings.push(format!(
                    "{label}: GitHub metadata could not verify a public repository ({error}); using public raw/HTML only"
                )),
            }
        }
        Ok(response) => {
            warnings.push(format!(
                "{label}: GitHub API metadata returned HTTP {}; trying public raw content",
                response.status
            ));
        }
        Err(error) => warnings.push(format!(
            "{label}: GitHub API metadata unavailable ({error}); trying public raw content"
        )),
    }

    let candidates = github_ref_candidates(repo, default_branch.as_deref());
    // A repository-root overview is already answered best by the repository
    // metadata and README. Avoid spending GitHub API quota (and latency) on a
    // recursive tree that will not add useful evidence to that answer.
    let mut resolved = overview_only.then(|| ResolvedGithubTarget {
        reference: candidates
            .first()
            .map(|candidate| candidate.reference.clone())
            .unwrap_or_else(|| "HEAD".to_string()),
        path: None,
        tree: None,
    });
    let mut api_tree_pages_remaining = MAX_GITHUB_API_TREE_PAGES;
    if api_available && resolved.is_none() {
        for candidate in &candidates {
            match fetch_github_tree_candidate(
                transport,
                repo,
                candidate,
                &query_terms,
                &mut api_tree_pages_remaining,
                warnings,
            ) {
                GithubTreeCandidateResult::Found(tree) => {
                    resolved = Some(ResolvedGithubTarget {
                        reference: candidate.reference.clone(),
                        path: candidate.path.clone(),
                        tree: Some(tree),
                    });
                    break;
                }
                GithubTreeCandidateResult::ProviderUnavailable => {
                    api_available = false;
                    break;
                }
                GithubTreeCandidateResult::NotFound => {}
            }
            if api_tree_pages_remaining == 0 {
                warnings.push(format!(
                    "{label}: GitHub tree lookup reached its {MAX_GITHUB_API_TREE_PAGES}-page limit"
                ));
                break;
            }
        }
    }

    if resolved.is_none() {
        resolved = resolve_github_target_without_api(transport, repo, &candidates, warnings);
    }
    let resolved = resolved.unwrap_or_else(|| ResolvedGithubTarget {
        reference: candidates
            .first()
            .map(|candidate| candidate.reference.clone())
            .unwrap_or_else(|| "HEAD".to_string()),
        path: candidates
            .first()
            .and_then(|candidate| candidate.path.clone()),
        tree: None,
    });

    let mut readme = None;
    if api_available {
        let mut readme_url = github_api_url(&repo.owner, &repo.repo, &["readme"]);
        readme_url
            .query_pairs_mut()
            .append_pair("ref", &resolved.reference);
        match transport
            .fetch_github_verified_public_api(&readme_url, "application/vnd.github.raw+json")
        {
            Ok(response) if (200..300).contains(&response.status) => {
                readme = response_excerpt_with_limit(&response, MAX_README_CHARS).map(|text| {
                    GithubReadme {
                        path: "README".to_string(),
                        url: github_readme_page_url(&repo.owner, &repo.repo, &resolved.reference),
                        text,
                    }
                });
            }
            Ok(response) if matches!(response.status, 401 | 403 | 429) => warnings.push(format!(
                "{label}: GitHub API README returned HTTP {}; using public raw fallback",
                response.status
            )),
            Ok(_) => {}
            Err(error) => warnings.push(format!(
                "{label}: GitHub API README unavailable ({error}); using public raw fallback"
            )),
        }
    }
    if readme.is_none() {
        readme = fetch_raw_github_readme(transport, &repo.owner, &repo.repo, &resolved.reference);
    }

    let readme_only = overview_only && readme.is_some();
    // Specific implementation questions keep code ahead of the README.
    // Generic repository overviews use only the README so the model receives
    // focused evidence without a broad and quota-expensive file-tree scan.
    let mut chunks = Vec::new();
    if !readme_only {
        if repo
            .view
            .as_ref()
            .is_some_and(|view| view.kind == GithubViewKind::Blob)
        {
            if let Some(path) = resolved.path.as_deref() {
                fetch_github_code_chunk(
                    transport,
                    repo,
                    &resolved.reference,
                    path,
                    &query_terms,
                    &mut chunks,
                    warnings,
                );
            }
        } else {
            let mut entries = resolved
                .tree
                .as_ref()
                .map(|tree| tree.tree.clone())
                .unwrap_or_default();
            if entries.is_empty() {
                entries = fetch_github_html_tree(
                    transport,
                    repo,
                    &resolved.reference,
                    resolved.path.as_deref(),
                    &query_terms,
                    warnings,
                );
            }
            for path in select_github_source_paths(&entries, resolved.path.as_deref(), &query_terms)
            {
                fetch_github_code_chunk(
                    transport,
                    repo,
                    &resolved.reference,
                    &path,
                    &query_terms,
                    &mut chunks,
                    warnings,
                );
            }
        }
    }

    normalize_github_tree_warnings(
        warnings,
        warning_start,
        &label,
        resolved.tree.as_ref().map(|tree| tree.truncated),
    );

    let title = readme
        .as_ref()
        .and_then(|readme| markdown_title(&readme.text))
        .unwrap_or_else(|| label.clone());
    if let Some(readme) = readme {
        chunks.push(WebResearchChunk {
            path: Some(readme.path),
            url: Some(readme.url.to_string()),
            text: readme.text,
        });
    }
    if chunks.is_empty() {
        warnings.push(format!(
            "Could not retrieve readable repository content for {label}"
        ));
        return None;
    }
    let combined = chunks
        .iter()
        .map(|chunk| {
            let heading = chunk.path.as_deref().unwrap_or("Source");
            format!("## Source: {heading}\n{}", chunk.text)
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    Some(WebResearchSource {
        id: 0,
        title: clip_chars(&title, 240, false),
        url: repo.canonical_url.as_str().to_string(),
        excerpt: clip_chars(&combined, MAX_SOURCE_EXCERPT_CHARS, true),
        chunks,
    })
}

fn normalize_github_tree_warnings(
    warnings: &mut Vec<String>,
    warning_start: usize,
    label: &str,
    incomplete: Option<bool>,
) {
    let Some(incomplete) = incomplete else {
        return;
    };
    let tail = warnings.split_off(warning_start.min(warnings.len()));
    let mut saw_tree_warning = false;
    for warning in tail {
        let detail = warning
            .strip_prefix(label)
            .map(str::trim_start)
            .and_then(|detail| detail.strip_prefix(':'))
            .map(str::trim_start);
        let is_tree_warning = detail.is_some_and(|detail| {
            let lower = detail.to_ascii_lowercase();
            detail.starts_with("GitHub returned a truncated recursive tree")
                || detail.starts_with("bounded GitHub tree traversal stopped")
                || detail.starts_with("GitHub returned a truncated file tree")
                || detail.starts_with("GitHub tree lookup reached")
                || (detail.starts_with("GitHub recursive file tree unavailable")
                    && (lower.contains("maximum allowed file size")
                        || lower.contains("maximum file size exceeded")
                        || lower.contains("body limit")))
        });
        if is_tree_warning {
            saw_tree_warning = true;
        } else {
            warnings.push(warning);
        }
    }
    if incomplete && saw_tree_warning {
        warnings.push(format!(
            "{label}: repository file coverage was incomplete after bounded GitHub traversal; only ranked visible files were considered"
        ));
    }
}

#[derive(Debug, Clone)]
struct GithubRefCandidate {
    reference: String,
    path: Option<String>,
}

#[derive(Debug, Clone)]
struct ResolvedGithubTarget {
    reference: String,
    path: Option<String>,
    tree: Option<GithubTree>,
}

fn github_ref_candidates(
    repo: &GithubRepo,
    default_branch: Option<&str>,
) -> Vec<GithubRefCandidate> {
    let Some(view) = repo.view.as_ref() else {
        return vec![GithubRefCandidate {
            reference: default_branch.unwrap_or("HEAD").to_string(),
            path: None,
        }];
    };
    let max_ref_segments = match view.kind {
        GithubViewKind::Tree => view.segments.len(),
        GithubViewKind::Blob => view.segments.len().saturating_sub(1),
    };
    (1..=max_ref_segments.min(MAX_GITHUB_REF_SEGMENTS))
        .rev()
        .map(|ref_segments| GithubRefCandidate {
            reference: view.segments[..ref_segments].join("/"),
            path: (ref_segments < view.segments.len())
                .then(|| view.segments[ref_segments..].join("/")),
        })
        .collect()
}

fn resolve_github_target_without_api(
    transport: &dyn WebTransport,
    repo: &GithubRepo,
    candidates: &[GithubRefCandidate],
    warnings: &mut Vec<String>,
) -> Option<ResolvedGithubTarget> {
    for candidate in candidates {
        let probe = match repo.view.as_ref().map(|view| view.kind) {
            Some(GithubViewKind::Blob) => candidate
                .path
                .as_deref()
                .map(|path| raw_github_url(&repo.owner, &repo.repo, &candidate.reference, path)),
            _ => Some(github_tree_page_url(
                &repo.owner,
                &repo.repo,
                &candidate.reference,
                candidate.path.as_deref(),
            )),
        }?;
        match transport.fetch(&probe, "text/html,text/plain") {
            Ok(response) if (200..300).contains(&response.status) => {
                if repo
                    .view
                    .as_ref()
                    .is_some_and(|view| view.kind == GithubViewKind::Tree)
                {
                    let body = String::from_utf8_lossy(&response.body);
                    let Some(target) = parse_github_tree_route(&body, repo) else {
                        warnings.push(format!(
                            "{}/{}: GitHub HTML did not identify the requested branch/path split",
                            repo.owner, repo.repo
                        ));
                        continue;
                    };
                    return Some(ResolvedGithubTarget {
                        reference: target.reference,
                        path: target.path,
                        tree: None,
                    });
                }
                return Some(ResolvedGithubTarget {
                    reference: candidate.reference.clone(),
                    path: candidate.path.clone(),
                    tree: None,
                });
            }
            Ok(_) => {}
            Err(error) => warnings.push(format!(
                "{}/{}: public GitHub fallback unavailable: {error}",
                repo.owner, repo.repo
            )),
        }
    }
    None
}

fn parse_github_tree_route(html: &str, repo: &GithubRepo) -> Option<GithubRefCandidate> {
    let view = repo.view.as_ref()?;
    if view.kind != GithubViewKind::Tree {
        return None;
    }
    let marker = "data-target=\"react-app.embeddedData\"";
    let mut rest = html;
    while let Some(marker_index) = rest.find(marker) {
        rest = &rest[marker_index + marker.len()..];
        let Some(start) = rest.find('>').map(|index| index + 1) else {
            break;
        };
        let after_start = &rest[start..];
        let Some(end) = after_start.find("</script>") else {
            break;
        };
        let payload = &after_start[..end];
        let candidate = serde_json::from_str::<serde_json::Value>(payload)
            .ok()
            .and_then(|value| {
                let route = value.pointer("/payload/codeViewTreeRoute")?;
                let reference = route.pointer("/refInfo/name")?.as_str()?.trim_matches('/');
                let path = route
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .map(|value| value.trim_matches('/'))
                    .filter(|value| !value.is_empty());
                Some((reference.to_string(), path.map(str::to_string)))
            });
        if let Some((reference, path)) = candidate {
            if reference.is_empty()
                || reference.len() > 1_024
                || reference
                    .split('/')
                    .any(|part| part.is_empty() || matches!(part, "." | ".."))
                || path.as_deref().is_some_and(|value| {
                    value.len() > 4_096
                        || value
                            .split('/')
                            .any(|part| part.is_empty() || matches!(part, "." | ".."))
                })
            {
                rest = &after_start[end + "</script>".len()..];
                continue;
            }
            let reconstructed = reference
                .split('/')
                .chain(
                    path.as_deref()
                        .into_iter()
                        .flat_map(|value| value.split('/')),
                )
                .collect::<Vec<_>>();
            if reconstructed != view.segments.iter().map(String::as_str).collect::<Vec<_>>() {
                rest = &after_start[end + "</script>".len()..];
                continue;
            }
            return Some(GithubRefCandidate { reference, path });
        }
        rest = &after_start[end + "</script>".len()..];
    }
    None
}

fn fetch_raw_github_readme(
    transport: &dyn WebTransport,
    owner: &str,
    repo: &str,
    reference: &str,
) -> Option<GithubReadme> {
    ["README.md", "readme.md", "README", "README.txt"]
        .into_iter()
        .find_map(|path| {
            let url = raw_github_url(owner, repo, reference, path);
            transport
                .fetch(&url, "text/plain")
                .ok()
                .filter(|response| (200..300).contains(&response.status))
                .and_then(|response| {
                    let text = response_excerpt_with_limit(&response, MAX_README_CHARS)?;
                    let provenance = response
                        .final_url
                        .as_ref()
                        .filter(|final_url| *final_url != &url)
                        .cloned()
                        .unwrap_or_else(|| github_blob_page_url(owner, repo, reference, path));
                    Some(GithubReadme {
                        path: path.to_string(),
                        url: provenance,
                        text,
                    })
                })
        })
}

fn fetch_github_code_chunk(
    transport: &dyn WebTransport,
    repo: &GithubRepo,
    reference: &str,
    path: &str,
    query_terms: &[String],
    chunks: &mut Vec<WebResearchChunk>,
    warnings: &mut Vec<String>,
) {
    let raw_url = raw_github_url(&repo.owner, &repo.repo, reference, path);
    match transport.fetch(&raw_url, "text/plain") {
        Ok(response) if (200..300).contains(&response.status) => {
            let provenance = response
                .final_url
                .as_ref()
                .filter(|final_url| *final_url != &raw_url)
                .cloned()
                .unwrap_or_else(|| github_blob_page_url(&repo.owner, &repo.repo, reference, path));
            if let Some(text) = code_excerpt(&response, MAX_CODE_FILE_CHARS, query_terms) {
                chunks.push(WebResearchChunk {
                    path: Some(path.to_string()),
                    url: Some(provenance.to_string()),
                    text,
                });
            }
        }
        Ok(response) => warnings.push(format!(
            "{}/{}: {path} returned HTTP {}",
            repo.owner, repo.repo, response.status
        )),
        Err(error) => warnings.push(format!(
            "{}/{}: {path} unavailable: {error}",
            repo.owner, repo.repo
        )),
    }
}

fn github_tree_page_url(owner: &str, repo: &str, reference: &str, path: Option<&str>) -> Url {
    let mut url = Url::parse("https://github.com/").expect("static GitHub URL");
    let mut segments = url.path_segments_mut().expect("GitHub URL is a base");
    segments.extend([owner, repo, "tree", reference]);
    if let Some(path) = path {
        segments.extend(path.split('/'));
    }
    drop(segments);
    url
}

fn github_blob_page_url(owner: &str, repo: &str, reference: &str, path: &str) -> Url {
    let mut url = Url::parse("https://github.com/").expect("static GitHub URL");
    let mut segments = url.path_segments_mut().expect("GitHub URL is a base");
    segments.extend([owner, repo, "blob"]);
    segments.extend(reference.split('/'));
    segments.extend(path.split('/'));
    drop(segments);
    url
}

fn github_readme_page_url(owner: &str, repo: &str, reference: &str) -> Url {
    let mut url = Url::parse("https://github.com/").expect("static GitHub URL");
    let mut segments = url.path_segments_mut().expect("GitHub URL is a base");
    segments.extend([owner, repo, "tree"]);
    segments.extend(reference.split('/'));
    drop(segments);
    url.set_fragment(Some("readme"));
    url
}

fn fetch_github_html_tree(
    transport: &dyn WebTransport,
    repo: &GithubRepo,
    reference: &str,
    path: Option<&str>,
    query_terms: &[String],
    warnings: &mut Vec<String>,
) -> Vec<GithubTreeEntry> {
    let root = path
        .map(|value| value.trim_matches('/').to_string())
        .filter(|value| !value.is_empty());
    let root_depth = root
        .as_deref()
        .map(|value| value.split('/').count())
        .unwrap_or(0);
    let mut pending = vec![root.clone()];
    let mut seen_pages = HashSet::new();
    let mut blob_paths = HashSet::new();

    for _ in 0..MAX_GITHUB_HTML_PAGES {
        if pending.is_empty() {
            break;
        }
        let next_index = pending
            .iter()
            .enumerate()
            .max_by_key(|(_, candidate)| github_directory_score(candidate.as_deref(), query_terms))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let directory = pending.swap_remove(next_index);
        if !seen_pages.insert(directory.clone()) {
            continue;
        }
        let url = github_tree_page_url(&repo.owner, &repo.repo, reference, directory.as_deref());
        let response = match transport.fetch(&url, "text/html") {
            Ok(response) if (200..300).contains(&response.status) => response,
            Ok(response) => {
                warnings.push(format!(
                    "{}/{}: GitHub HTML tree returned HTTP {}",
                    repo.owner, repo.repo, response.status
                ));
                continue;
            }
            Err(error) => {
                warnings.push(format!(
                    "{}/{}: GitHub HTML tree unavailable: {error}",
                    repo.owner, repo.repo
                ));
                continue;
            }
        };
        let body = String::from_utf8_lossy(&response.body);
        blob_paths.extend(parse_github_page_paths(&body, repo, reference, "blob"));
        for child in parse_github_page_paths(&body, repo, reference, "tree") {
            let depth = child.split('/').count();
            let inside_root = root.as_deref().is_none_or(|prefix| {
                child == prefix
                    || child
                        .strip_prefix(prefix)
                        .is_some_and(|rest| rest.starts_with('/'))
            });
            if inside_root
                && depth.saturating_sub(root_depth) <= MAX_GITHUB_HTML_DEPTH
                && !seen_pages.contains(&Some(child.clone()))
            {
                pending.push(Some(child));
            }
        }
    }

    let mut entries = blob_paths
        .into_iter()
        .map(|path| GithubTreeEntry {
            path,
            kind: "blob".to_string(),
            size: None,
            sha: None,
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    entries
}

fn github_directory_score(path: Option<&str>, query_terms: &[String]) -> i32 {
    let Some(path) = path else { return i32::MAX };
    let lower = path.to_ascii_lowercase();
    let basename = lower.rsplit('/').next().unwrap_or(lower.as_str());
    let query_matches = query_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count() as i32;
    let conventional = match basename {
        "src" | "lib" | "app" => 400,
        "packages" | "examples" => 200,
        "docs" | "test" | "tests" => 100,
        _ => 0,
    };
    query_matches * 1_000 + conventional - path.matches('/').count().min(20) as i32
}

fn parse_github_page_paths(
    html: &str,
    repo: &GithubRepo,
    reference: &str,
    kind: &str,
) -> Vec<String> {
    let exact_prefix = format!("/{}/{}/{kind}/{reference}/", repo.owner, repo.repo);
    let route_prefix = format!("/{}/{}/{kind}/", repo.owner, repo.repo);
    let mut paths = HashSet::new();
    for marker in ["href=\"", "href='"] {
        let quote = marker.chars().last().unwrap_or('"');
        let mut rest = html;
        while let Some(index) = rest.find(marker) {
            rest = &rest[index + marker.len()..];
            let Some(end) = rest.find(quote) else { break };
            let href = decode_html_entities(&rest[..end]);
            let href = href.split(['?', '#']).next().unwrap_or_default();
            let path = href.strip_prefix(&exact_prefix).or_else(|| {
                let remainder = href.strip_prefix(&route_prefix)?;
                let (rendered_ref, path) = remainder.split_once('/')?;
                is_github_commit_ref(rendered_ref).then_some(path)
            });
            if let Some(path) = path {
                if !path.is_empty() {
                    paths.insert(path.to_string());
                }
            }
            rest = &rest[end + quote.len_utf8()..];
        }
    }
    let mut paths = paths.into_iter().collect::<Vec<_>>();
    paths.sort();
    paths
}

fn is_github_commit_ref(value: &str) -> bool {
    (7..=64).contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn github_api_url(owner: &str, repo: &str, suffix: &[&str]) -> Url {
    let mut url = Url::parse("https://api.github.com/").expect("static GitHub API URL");
    {
        let mut path = url.path_segments_mut().expect("GitHub API URL is a base");
        path.extend(["repos", owner, repo]);
        path.extend(suffix.iter().copied());
    }
    if suffix.first() == Some(&"trees") || suffix.get(1) == Some(&"trees") {
        url.query_pairs_mut().append_pair("recursive", "1");
    }
    url
}

fn github_nonrecursive_tree_api_url(owner: &str, repo: &str, treeish: &str) -> Url {
    let mut url = github_api_url(owner, repo, &["git", "trees", treeish]);
    url.set_query(None);
    url
}

fn raw_github_url(owner: &str, repo: &str, branch: &str, path: &str) -> Url {
    let mut url =
        Url::parse("https://raw.githubusercontent.com/").expect("static GitHub raw-content URL");
    {
        let mut segments = url
            .path_segments_mut()
            .expect("GitHub raw-content URL is a base");
        segments.extend([owner, repo, branch]);
        segments.extend(path.split('/'));
    }
    url
}

fn select_github_source_paths(
    entries: &[GithubTreeEntry],
    path_prefix: Option<&str>,
    query_terms: &[String],
) -> Vec<String> {
    let path_prefix = path_prefix
        .map(|prefix| prefix.trim_matches('/'))
        .filter(|prefix| !prefix.is_empty());
    let mut candidates: Vec<(i32, &str)> = entries
        .iter()
        .filter(|entry| entry.kind == "blob")
        .filter(|entry| entry.size.unwrap_or(0) <= MAX_HTTP_BODY_BYTES as u64)
        .filter(|entry| {
            path_prefix.is_none_or(|prefix| {
                entry.path == prefix
                    || entry
                        .path
                        .strip_prefix(prefix)
                        .is_some_and(|rest| rest.starts_with('/'))
            })
        })
        .filter_map(|entry| {
            source_path_score(&entry.path, query_terms).map(|score| (score, entry.path.as_str()))
        })
        .collect();
    candidates.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.to_lowercase().cmp(&right.1.to_lowercase()))
    });
    candidates
        .into_iter()
        .take(MAX_GITHUB_SOURCE_FILES)
        .map(|(_, path)| path.to_string())
        .collect()
}

fn source_path_score(path: &str, query_terms: &[String]) -> Option<i32> {
    let lower = path.to_ascii_lowercase();
    let basename = lower.rsplit('/').next()?;
    if lower.contains("/vendor/")
        || lower.contains("/node_modules/")
        || lower.contains("/dist/")
        || lower.contains("/target/")
        || basename.ends_with(".min.js")
        || matches!(basename, "package-lock.json" | "cargo.lock" | "license")
        || basename.starts_with("readme")
    {
        return None;
    }
    let extension = basename.rsplit_once('.').map(|(_, ext)| ext).unwrap_or("");
    if ![
        "c", "cc", "cpp", "css", "go", "h", "hpp", "html", "java", "js", "jsx", "kt", "m", "mm",
        "php", "py", "rb", "rs", "sh", "swift", "ts", "tsx", "xml", "yaml", "yml",
    ]
    .contains(&extension)
    {
        return None;
    }

    let query_matches = query_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count() as i32;
    let entrypoint = ["main.", "app.", "index.", "lib.", "mod."]
        .iter()
        .any(|prefix| basename.starts_with(prefix));
    let depth = path.matches('/').count().min(20) as i32;
    Some(query_matches * 1_000 + i32::from(entrypoint) * 250 + (40 - depth))
}

fn is_github_repository_overview_request(query: &str, repo: &GithubRepo) -> bool {
    const IMPLEMENTATION_CUES: &[&str] = &[
        "architecture",
        "backend",
        "bug",
        "build",
        "class",
        "code",
        "dependency",
        "dependencies",
        "endpoint",
        "feature",
        "file",
        "fix",
        "frontend",
        "function",
        "implement",
        "implementation",
        "modify",
        "module",
        "performance",
        "protocol",
        "security",
        "source",
        "test",
        "update",
        "vulnerability",
        "vulnerabilities",
    ];
    const GENERIC_OVERVIEW_WORDS: &[&str] = &[
        "a",
        "about",
        "an",
        "and",
        "can",
        "check",
        "could",
        "describe",
        "do",
        "does",
        "explain",
        "for",
        "give",
        "github",
        "inspect",
        "is",
        "it",
        "its",
        "me",
        "of",
        "overview",
        "please",
        "read",
        "repo",
        "repository",
        "review",
        "summarize",
        "summary",
        "tell",
        "the",
        "this",
        "us",
        "what",
        "would",
        "you",
    ];

    let without_urls = search_query_from_prompt(query).to_ascii_lowercase();
    let words = without_urls
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if words.iter().any(|word| IMPLEMENTATION_CUES.contains(word)) {
        return false;
    }
    if words.is_empty() {
        return true;
    }
    let repository_subject_words = [&repo.owner, &repo.repo]
        .into_iter()
        .flat_map(|value| {
            value
                .split(|ch: char| !ch.is_ascii_alphanumeric())
                .filter(|word| !word.is_empty())
                .map(str::to_ascii_lowercase)
        })
        .collect::<HashSet<_>>();
    words.iter().all(|word| {
        GENERIC_OVERVIEW_WORDS.contains(word) || repository_subject_words.contains(*word)
    })
}

fn query_terms(query: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "about",
        "after",
        "also",
        "and",
        "app",
        "build",
        "can",
        "code",
        "could",
        "create",
        "describe",
        "from",
        "github",
        "have",
        "http",
        "https",
        "into",
        "need",
        "overview",
        "please",
        "plan",
        "read",
        "repo",
        "repository",
        "search",
        "that",
        "the",
        "their",
        "this",
        "tell",
        "use",
        "using",
        "want",
        "web",
        "what",
        "with",
        "would",
        "you",
        "your",
    ];
    let mut seen = HashSet::new();
    let without_urls = query
        .split_whitespace()
        .filter(|part| !part.contains("http://") && !part.contains("https://"))
        .collect::<Vec<_>>()
        .join(" ");
    without_urls
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-')
        .map(str::to_ascii_lowercase)
        .filter(|term| term.len() >= 3 && !STOP_WORDS.contains(&term.as_str()))
        .filter(|term| seen.insert(term.clone()))
        .take(24)
        .collect()
}

fn search_and_fetch(
    transport: &dyn WebTransport,
    reason: &'static str,
    query: String,
) -> WebResearchResponse {
    let mut warnings = Vec::new();
    let mut hits = search_brave(transport, &query, &mut warnings);
    if hits.is_empty() {
        hits = search_ddg(transport, &query, &mut warnings);
    }
    if hits.is_empty() {
        hits = search_bing(transport, &query, &mut warnings);
    }
    if hits.is_empty() {
        return finish_response(reason, Some(query), Vec::new(), warnings);
    }
    let mut sources = Vec::new();
    for hit in hits.into_iter().take(MAX_SEARCH_FETCHES) {
        fetch_source(
            transport,
            &hit.url,
            Some(&hit.title),
            &query,
            &mut sources,
            &mut warnings,
        );
    }
    finish_response(reason, Some(query), sources, warnings)
}

fn search_brave(
    transport: &dyn WebTransport,
    query: &str,
    warnings: &mut Vec<String>,
) -> Vec<SearchHit> {
    let mut url = Url::parse(BRAVE_SEARCH_ENDPOINT).expect("static Brave search endpoint");
    url.query_pairs_mut()
        .append_pair("q", query)
        .append_pair("source", "web");
    let response = match transport.fetch(&url, "text/html") {
        Ok(response) => response,
        Err(error) => {
            warnings.push(format!(
                "Brave web search was unavailable ({error}); trying DuckDuckGo Lite."
            ));
            return Vec::new();
        }
    };
    let body = String::from_utf8_lossy(&response.body);
    if response.status == 202 || looks_like_brave_challenge(&body) {
        warnings.push(format!(
            "Brave web search challenged the automated request (HTTP {}); trying DuckDuckGo Lite.",
            response.status
        ));
        return Vec::new();
    }
    if response.status != 200 {
        warnings.push(format!(
            "Brave web search returned HTTP {}; trying DuckDuckGo Lite.",
            response.status
        ));
        return Vec::new();
    }
    let hits = parse_brave_results(&body);
    if hits.is_empty() {
        warnings.push(
            "Brave web search returned no usable web results; trying DuckDuckGo Lite.".to_string(),
        );
    }
    hits
}

fn search_ddg(
    transport: &dyn WebTransport,
    query: &str,
    warnings: &mut Vec<String>,
) -> Vec<SearchHit> {
    let mut url = Url::parse(DDG_SEARCH_ENDPOINT).expect("static DDG Lite endpoint");
    url.query_pairs_mut().append_pair("q", query);
    let response = match transport.fetch(&url, "text/html") {
        Ok(response) => response,
        Err(error) => {
            warnings.push(format!("DuckDuckGo Lite was unavailable: {error}"));
            return Vec::new();
        }
    };
    let body = String::from_utf8_lossy(&response.body);
    if response.status == 202 || looks_like_ddg_challenge(&body) {
        warnings.push(format!(
            "DuckDuckGo Lite challenged the automated request (HTTP {}); no result was treated as evidence.",
            response.status
        ));
        return Vec::new();
    }
    if response.status != 200 {
        warnings.push(format!(
            "DuckDuckGo Lite returned HTTP {}; no result was treated as evidence.",
            response.status
        ));
        return Vec::new();
    }
    let hits = parse_ddg_results(&body);
    if hits.is_empty() {
        warnings.push("DuckDuckGo Lite returned no usable results.".to_string());
    }
    hits
}

fn search_bing(
    transport: &dyn WebTransport,
    query: &str,
    warnings: &mut Vec<String>,
) -> Vec<SearchHit> {
    let mut url = Url::parse(BING_SEARCH_ENDPOINT).expect("static Bing search endpoint");
    url.query_pairs_mut().append_pair("q", query);
    let response = match transport.fetch(&url, "text/html") {
        Ok(response) => response,
        Err(error) => {
            warnings.push(format!("Bing web search was unavailable: {error}"));
            return Vec::new();
        }
    };
    let body = String::from_utf8_lossy(&response.body);
    if response.status == 202 || looks_like_bing_challenge(&body) {
        warnings.push(format!(
            "Bing web search challenged the automated request (HTTP {}); no result was treated as evidence.",
            response.status
        ));
        return Vec::new();
    }
    if response.status != 200 {
        warnings.push(format!(
            "Bing web search returned HTTP {}; no result was treated as evidence.",
            response.status
        ));
        return Vec::new();
    }
    let hits = parse_bing_results(&body);
    if hits.is_empty() {
        warnings.push("Bing web search returned no usable web results.".to_string());
    }
    hits
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SearchHit {
    title: String,
    url: Url,
}

fn looks_like_brave_challenge(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("<title>captcha")
        || lower.contains("<title>challenge")
        || lower.contains("cf-chl-")
        || lower.contains("scheduled a captcha")
        || lower.contains("suspicious request")
}

fn looks_like_ddg_challenge(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("captcha")
        || lower.contains("automated requests")
        || lower.contains("anomaly-modal")
        || lower.contains("challenge-form")
}

fn looks_like_bing_challenge(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("captcha")
        || lower.contains("unusual traffic")
        || lower.contains("verify you are human")
}

fn parse_brave_results(html: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut seen = HashSet::new();
    let markers = html
        .match_indices("data-type=\"web\"")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    for (position, marker) in markers.iter().copied().enumerate() {
        let start = html[..marker].rfind("<div").unwrap_or(marker);
        let end = markers.get(position + 1).copied().unwrap_or(html.len());
        let snippet = &html[start..end];
        let Some(anchor_start) = snippet.match_indices("<a ").find_map(|(index, _)| {
            let anchor = &snippet[index..];
            let tag_end = anchor.find('>')?;
            let attributes = &anchor[..tag_end];
            let classes = html_attribute(attributes, "class")?;
            classes
                .split_whitespace()
                .any(|class| class == "l1")
                .then_some(index)
        }) else {
            continue;
        };
        let anchor = &snippet[anchor_start..];
        let Some(tag_end) = anchor.find('>') else {
            continue;
        };
        let attributes = &anchor[..tag_end];
        let Some(href) = html_attribute(attributes, "href") else {
            continue;
        };
        let Some(url) = unwrap_search_redirect(&decode_html_entities(href)) else {
            continue;
        };
        let title = brave_result_title(snippet).unwrap_or_else(|| display_url(&url));
        if title.is_empty() || !seen.insert(url.as_str().to_string()) {
            continue;
        }
        hits.push(SearchHit { title, url });
        if hits.len() == MAX_SEARCH_RESULTS {
            break;
        }
    }
    hits
}

fn brave_result_title(snippet: &str) -> Option<String> {
    let marker = snippet.find("search-snippet-title")?;
    let tag_start = snippet[..marker].rfind('<')?;
    let tag_end = tag_start + snippet[tag_start..].find('>')?;
    let tag = &snippet[tag_start..tag_end];
    if let Some(title) = html_attribute(tag, "title") {
        let title = decode_html_entities(title).trim().to_string();
        if !title.is_empty() {
            return Some(title);
        }
    }
    let content_end = tag_end + 1 + snippet[tag_end + 1..].find("</")?;
    let title = strip_html(&snippet[tag_end + 1..content_end]);
    (!title.is_empty()).then_some(title)
}

fn html_attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    for quote in ['"', '\''] {
        let pattern = format!("{name}={quote}");
        if let Some((_, rest)) = tag.split_once(&pattern) {
            if let Some(end) = rest.find(quote) {
                return Some(&rest[..end]);
            }
        }
    }
    None
}

fn parse_ddg_results(html: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut seen = HashSet::new();
    for (index, _) in html.match_indices("result-link") {
        let before = &html[..index];
        let Some(anchor_at) = before.rfind("<a ") else {
            continue;
        };
        let anchor = &html[anchor_at..];
        let Some(tag_end) = anchor.find('>') else {
            continue;
        };
        let attributes = &anchor[..tag_end];
        let Some(href_at) = attributes.find("href=") else {
            continue;
        };
        let rest = &attributes[href_at + 5..];
        let Some(quote) = rest.chars().next().filter(|ch| matches!(ch, '\'' | '"')) else {
            continue;
        };
        let rest = &rest[quote.len_utf8()..];
        let Some(url_end) = rest.find(quote) else {
            continue;
        };
        let raw_href = decode_html_entities(&rest[..url_end]);
        let Some(url) = unwrap_search_redirect(&raw_href) else {
            continue;
        };
        let title = strip_html(anchor[tag_end + 1..].split("</a>").next().unwrap_or(""));
        if title.is_empty() || !seen.insert(url.as_str().to_string()) {
            continue;
        }
        hits.push(SearchHit { title, url });
        if hits.len() == MAX_SEARCH_RESULTS {
            break;
        }
    }
    hits
}

fn parse_bing_results(html: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut seen = HashSet::new();
    for (marker, _) in html.match_indices("class=\"b_algo\"") {
        let start = html[..marker].rfind("<li").unwrap_or(marker);
        let tail = &html[start..];
        let end = tail.find("</li>").unwrap_or(tail.len());
        let snippet = &tail[..end];
        let Some(heading) = snippet.find("<h2") else {
            continue;
        };
        let heading = &snippet[heading..];
        let Some(anchor_start) = heading.find("<a ") else {
            continue;
        };
        let anchor = &heading[anchor_start..];
        let Some(tag_end) = anchor.find('>') else {
            continue;
        };
        let Some(href) = html_attribute(&anchor[..tag_end], "href") else {
            continue;
        };
        let Some(url) = unwrap_search_redirect(&decode_html_entities(href)) else {
            continue;
        };
        let title = strip_html(anchor[tag_end + 1..].split("</a>").next().unwrap_or(""));
        if title.is_empty() || !seen.insert(url.as_str().to_string()) {
            continue;
        }
        hits.push(SearchHit { title, url });
        if hits.len() == MAX_SEARCH_RESULTS {
            break;
        }
    }
    hits
}

fn unwrap_search_redirect(href: &str) -> Option<Url> {
    let absolute = if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_string()
    };
    let parsed = Url::parse(&absolute).ok()?;
    if parsed.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("duckduckgo.com")
            || host.to_ascii_lowercase().ends_with(".duckduckgo.com")
    }) {
        if let Some((_, target)) = parsed.query_pairs().find(|(key, _)| key == "uddg") {
            return Url::parse(&target).ok().map(canonical_prompt_url);
        }
    }
    if parsed.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("bing.com") || host.to_ascii_lowercase().ends_with(".bing.com")
    }) && parsed.path().starts_with("/ck/")
    {
        let encoded = parsed
            .query_pairs()
            .find_map(|(key, value)| (key == "u").then_some(value.into_owned()))?;
        let payload = encoded.strip_prefix("a1")?;
        let decoded = URL_SAFE_NO_PAD.decode(payload.as_bytes()).ok()?;
        let target = String::from_utf8(decoded).ok()?;
        return Url::parse(&target).ok().map(canonical_prompt_url);
    }
    matches!(parsed.scheme(), "http" | "https").then(|| canonical_prompt_url(parsed))
}

fn response_excerpt(response: &FetchResponse) -> Option<String> {
    response_excerpt_with_limit(response, MAX_SOURCE_EXCERPT_CHARS)
}

fn response_excerpt_with_limit(response: &FetchResponse, max_chars: usize) -> Option<String> {
    if !supported_text_content_type(&response.content_type) || !is_probably_text(&response.body) {
        return None;
    }
    let body = String::from_utf8_lossy(&response.body);
    let text = if response.content_type.to_ascii_lowercase().contains("html") {
        strip_html(&body)
    } else {
        clean_plain_text(&body)
    };
    (!text.is_empty()).then(|| clip_chars(&text, max_chars, true))
}

fn code_excerpt(
    response: &FetchResponse,
    max_chars: usize,
    query_terms: &[String],
) -> Option<String> {
    if !supported_text_content_type(&response.content_type) || !is_probably_text(&response.body) {
        return None;
    }
    let text = clean_plain_text(&String::from_utf8_lossy(&response.body));
    if text.is_empty() {
        return None;
    }
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return Some(text);
    }

    // Center long files on the strongest query-matching line while retaining a
    // small header for imports/types. This is repository-agnostic: no project
    // paths, symbols, or previously extracted facts are built into selection.
    let mut byte_offset = 0usize;
    let mut best = None;
    for line in text.split_inclusive('\n') {
        let lower = line.to_ascii_lowercase();
        let score = query_terms
            .iter()
            .filter(|term| lower.contains(term.as_str()))
            .count();
        if score > 0
            && best
                .as_ref()
                .is_none_or(|(best_score, _)| score > *best_score)
        {
            best = Some((score, text[..byte_offset].chars().count()));
        }
        byte_offset += line.len();
    }
    let anchor = best.map(|(_, char_offset)| char_offset);
    let Some(anchor) = anchor.filter(|anchor| *anchor > 500) else {
        return Some(clip_chars(&text, max_chars, true));
    };

    let marker = "\n\n[earlier content omitted; query-relevant section follows]\n";
    let marker_chars = marker.chars().count();
    let prefix_chars = 400.min(max_chars / 4);
    let window_chars = max_chars.saturating_sub(prefix_chars + marker_chars);
    let prefix = chars.iter().take(prefix_chars).collect::<String>();
    let window = chars
        .iter()
        .skip(anchor)
        .take(window_chars)
        .collect::<String>();
    Some(format!("{prefix}{marker}{window}"))
}

fn supported_text_content_type(content_type: &str) -> bool {
    let content_type = content_type.to_ascii_lowercase();
    content_type.is_empty()
        || content_type.starts_with("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
        || content_type.contains("javascript")
        || content_type.contains("x-sh")
}

fn is_probably_text(body: &[u8]) -> bool {
    if body.is_empty() || body.contains(&0) {
        return false;
    }
    let sample = &body[..body.len().min(32 * 1024)];
    let control_bytes = sample
        .iter()
        .filter(|byte| **byte < 0x20 && !matches!(**byte, b'\n' | b'\r' | b'\t' | 0x0c))
        .count();
    if control_bytes > sample.len() / 100 + 2 {
        return false;
    }
    let decoded = String::from_utf8_lossy(sample);
    let replacements = decoded.chars().filter(|ch| *ch == '\u{fffd}').count();
    replacements <= decoded.chars().count() / 50 + 1
}

fn clean_plain_text(value: &str) -> String {
    value
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\0', "")
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn strip_html(value: &str) -> String {
    let without_scripts = remove_html_element(value, "script");
    let without_styles = remove_html_element(&without_scripts, "style");
    let mut text = String::with_capacity(without_styles.len());
    let mut inside_tag = false;
    for ch in without_styles.chars() {
        match ch {
            '<' => {
                inside_tag = true;
                text.push(' ');
            }
            '>' => inside_tag = false,
            _ if !inside_tag => text.push(ch),
            _ => {}
        }
    }
    decode_html_entities(&text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn remove_html_element(value: &str, tag: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(value.len());
    let mut cursor = 0;
    while let Some(relative_start) = lower[cursor..].find(&open) {
        let start = cursor + relative_start;
        out.push_str(&value[cursor..start]);
        let Some(relative_end) = lower[start..].find(&close) else {
            return out;
        };
        cursor = start + relative_end + close.len();
    }
    out.push_str(&value[cursor..]);
    out
}

fn extract_html_title(value: &str) -> Option<String> {
    let lower = value.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let content_start = start + lower[start..].find('>')? + 1;
    let content_end = content_start + lower[content_start..].find("</title>")?;
    let title = strip_html(&value[content_start..content_end]);
    (!title.is_empty()).then_some(title)
}

fn markdown_title(value: &str) -> Option<String> {
    value.lines().find_map(|line| {
        let title = line.trim().strip_prefix("# ")?.trim();
        (!title.is_empty()).then(|| title.to_string())
    })
}

fn decode_html_entities(value: &str) -> String {
    value
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn clip_chars(value: &str, max_chars: usize, marker: bool) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut clipped: String = value.chars().take(max_chars).collect();
    if marker {
        clipped.push_str("\n[excerpt truncated]");
    }
    clipped
}

fn display_url(url: &Url) -> String {
    let host = url.host_str().unwrap_or("web target");
    let path = url.path();
    if path == "/" || path.is_empty() {
        host.to_string()
    } else {
        format!("{host}{path}")
    }
}

fn validate_url_shape(url: &Url) -> Result<(), String> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err("only public http:// and https:// URLs are allowed".to_string());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("URLs containing credentials are not allowed".to_string());
    }
    if url.port().is_some() {
        return Err("only the default HTTP and HTTPS ports are allowed".to_string());
    }
    match url.host() {
        Some(Host::Domain(host)) => {
            let lower = host.to_ascii_lowercase();
            if lower == "localhost"
                || lower.ends_with(".localhost")
                || lower.ends_with(".local")
                || lower.ends_with(".internal")
            {
                return Err("local and internal hostnames are not allowed".to_string());
            }
        }
        Some(Host::Ipv4(address)) => {
            if !is_public_ip(IpAddr::V4(address)) {
                return Err(
                    "loopback, private, link-local, and reserved IPs are not allowed".into(),
                );
            }
        }
        Some(Host::Ipv6(address)) => {
            if !is_public_ip(IpAddr::V6(address)) {
                return Err(
                    "loopback, private, link-local, and reserved IPs are not allowed".into(),
                );
            }
        }
        None => return Err("URL must include a host".to_string()),
    }
    Ok(())
}

fn research_deadline_error() -> String {
    format!(
        "the {}-second web research deadline was reached",
        RESEARCH_TOTAL_DEADLINE.as_secs()
    )
}

fn resolve_public_addresses(
    url: &Url,
    cancel: Option<&tokio_util::sync::CancellationToken>,
    deadline: Option<Instant>,
) -> Result<Vec<IpAddr>, String> {
    resolve_public_addresses_with(url, cancel, deadline, |host, port| {
        (host.as_str(), port)
            .to_socket_addrs()
            .map_err(|error| format!("could not resolve public host {host}: {error}"))
            .map(|addresses| addresses.map(|address| address.ip()).collect())
    })
}

fn resolve_public_addresses_with<F>(
    url: &Url,
    cancel: Option<&tokio_util::sync::CancellationToken>,
    deadline: Option<Instant>,
    resolver: F,
) -> Result<Vec<IpAddr>, String>
where
    F: FnOnce(String, u16) -> Result<Vec<IpAddr>, String> + Send + 'static,
{
    validate_url_shape(url)?;
    if cancel.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
        return Err("web research was cancelled".to_string());
    }
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err(research_deadline_error());
    }
    if let Some(ip) = match url.host() {
        Some(Host::Ipv4(address)) => Some(IpAddr::V4(address)),
        Some(Host::Ipv6(address)) => Some(IpAddr::V6(address)),
        _ => None,
    } {
        return Ok(vec![ip]);
    }
    let host = url
        .host_str()
        .ok_or_else(|| "URL must include a host".to_string())?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| "URL has no HTTP(S) port".to_string())?;
    let host = host.to_string();
    let display_host = host.clone();
    // The research worker may abandon a stuck system resolver at its deadline,
    // but the resolver thread itself can only be reclaimed by the OS call
    // returning. Keep a separate, process-wide cap owned by those threads so
    // repeated abandoned lookups cannot accumulate without bound.
    let dns_permit = dns_lookup_gate().try_acquire().ok_or_else(|| {
        "DNS lookup capacity is busy; web research can continue without this source".to_string()
    })?;
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
    thread::Builder::new()
        .name("camelid-web-dns".to_string())
        .spawn(move || {
            let _dns_permit = dns_permit;
            let _ = result_tx.send(resolver(host, port));
        })
        .map_err(|error| format!("could not start DNS lookup for {display_host}: {error}"))?;

    let mut addresses = loop {
        if cancel.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
            return Err("web research was cancelled".to_string());
        }
        let wait = match deadline {
            Some(deadline) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(research_deadline_error());
                }
                remaining.min(Duration::from_millis(10))
            }
            None => Duration::from_millis(10),
        };
        match result_rx.recv_timeout(wait) {
            Ok(result) => break result?,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(format!(
                    "DNS lookup worker for {display_host} stopped unexpectedly"
                ))
            }
        }
    };
    addresses.sort_unstable();
    addresses.dedup();
    if addresses.is_empty() {
        return Err(format!("public host {display_host} did not resolve"));
    }
    if addresses.iter().any(|address| !is_public_ip(*address)) {
        return Err(format!(
            "host {display_host} resolved to a loopback, private, link-local, or reserved address"
        ));
    }
    Ok(addresses)
}

fn is_public_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_public_ipv4(address),
        IpAddr::V6(address) => is_public_ipv6(address),
    }
}

fn is_public_ipv4(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    !(address.is_unspecified()
        || address.is_loopback()
        || address.is_private()
        || address.is_link_local()
        || address.is_multicast()
        || address.is_broadcast()
        || a == 0
        || (a == 100 && (64..=127).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 192 && b == 88 && c == 99)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 240)
}

fn is_public_ipv6(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4() {
        return is_public_ipv4(mapped);
    }
    let segments = address.segments();
    !(address.is_unspecified()
        || address.is_loopback()
        || address.is_multicast()
        // Public IPv6 global unicast is allocated from 2000::/3. Reject
        // unallocated/reserved literals even if a host has a local route.
        || (segments[0] & 0xe000) != 0x2000
        || (segments[0] & 0xfe00) == 0xfc00
        || (segments[0] & 0xffc0) == 0xfe80
        || (segments[0] & 0xffc0) == 0xfec0
        || (segments[0] == 0x0064
            && segments[1] == 0xff9b
            && ((segments[2] == 0 && segments[3] == 0 && segments[4] == 0 && segments[5] == 0)
                || segments[2] == 1))
        || (segments[0] == 0x0100 && segments[1] == 0 && segments[2] == 0 && segments[3] == 0)
        || segments[0] == 0x2002
        || (segments[0] == 0x2001 && segments[1] == 0)
        || (segments[0] == 0x2001 && segments[1] == 0x0002 && segments[2] == 0)
        || (segments[0] == 0x2001 && (segments[1] & 0xfff0) == 0x0010)
        || (segments[0] == 0x2001 && (segments[1] & 0xfff0) == 0x0020)
        || (segments[0] == 0x2001 && segments[1] == 0x0db8)
        || (segments[0] == 0x3fff && (segments[1] & 0xf000) == 0))
}

fn validated_redirect_target(current: &Url, location: &str) -> Result<Url, String> {
    let next = current
        .join(location.trim())
        .map_err(|error| format!("invalid redirect target: {error}"))?;
    validate_url_shape(&next)?;
    if current.scheme() == "https" && next.scheme() != "https" {
        return Err("HTTPS redirects may not downgrade to cleartext HTTP".to_string());
    }
    Ok(next)
}

// This is the final SSRF/auth boundary before spawning curl. Keeping every
// validated input explicit makes accidental credential or redirect widening
// visible at each call site.
#[expect(clippy::too_many_arguments)]
fn curl_single_hop(
    url: &Url,
    accept: &str,
    addresses: &[IpAddr],
    etag: Option<&str>,
    github_token: Option<&str>,
    cancel: Option<&tokio_util::sync::CancellationToken>,
    deadline: Option<Instant>,
    head_only: bool,
) -> Result<FetchResponse, String> {
    if cancel.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
        return Err("web research was cancelled".to_string());
    }
    let remaining = deadline
        .map(|deadline| deadline.saturating_duration_since(Instant::now()))
        .unwrap_or(Duration::from_secs(15));
    if remaining.is_zero() {
        return Err(research_deadline_error());
    }
    let hop_timeout = remaining.min(Duration::from_secs(15));
    let connect_timeout = hop_timeout.min(Duration::from_secs(5));
    let hop_timeout_arg = curl_timeout_arg(hop_timeout).ok_or_else(research_deadline_error)?;
    let connect_timeout_arg =
        curl_timeout_arg(connect_timeout).ok_or_else(research_deadline_error)?;
    let host = url
        .host_str()
        .ok_or_else(|| "URL must include a host".to_string())?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| "URL has no HTTP(S) port".to_string())?;
    let mut command = Command::new("curl");
    // -q/--disable must be the first curl option to prevent ~/.curlrc from
    // injecting credentials, proxies, or broader protocols.
    command.args([
        "-q",
        "--silent",
        "--show-error",
        "--no-progress-meter",
        "--noproxy",
        "*",
        "--proto",
        "=http,https",
        "--max-redirs",
        "0",
        "--connect-timeout",
        &connect_timeout_arg,
        "--max-time",
        &hop_timeout_arg,
        "--max-filesize",
        &MAX_HTTP_BODY_BYTES.to_string(),
        "--user-agent",
        USER_AGENT,
        "--header",
        &format!("Accept: {accept}"),
        "--dump-header",
        "-",
    ]);
    if let Some(etag) = etag.filter(|etag| safe_http_header_value(etag)) {
        command.args(["--header", &format!("If-None-Match: {etag}")]);
    }
    if github_token.is_some() {
        // Feed provider credentials through curl's stdin config instead of the
        // process argument list. The caller scopes this token to HTTPS requests
        // whose host is exactly api.github.com, and redirect hops are rebuilt.
        command.args([
            "--config",
            "-",
            "--header",
            "X-GitHub-Api-Version: 2022-11-28",
        ]);
    }
    if head_only {
        command.arg("--head");
    }
    if url
        .host()
        .is_some_and(|host| matches!(host, Host::Domain(_)))
    {
        for address in addresses {
            let pinned = match address {
                IpAddr::V4(address) => format!("{host}:{port}:{address}"),
                IpAddr::V6(address) => format!("{host}:{port}:[{address}]"),
            };
            command.args(["--resolve", &pinned]);
        }
    }
    command
        .arg("--")
        .arg(url.as_str())
        .stdin(if github_token.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("could not start curl: {error}"))?;
    if let Some(token) = github_token {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "curl credential input was unavailable".to_string())?;
        stdin
            .write_all(format!("header = \"Authorization: Bearer {token}\"\n").as_bytes())
            .map_err(|error| format!("could not provide GitHub credential to curl: {error}"))?;
    }
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "curl stdout was unavailable".to_string())?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| "curl stderr was unavailable".to_string())?;
    let output_limit = MAX_HTTP_HEADER_BYTES + MAX_HTTP_BODY_BYTES + 1;
    let stdout_reader = thread::spawn(move || {
        let mut output = Vec::with_capacity(output_limit.min(64 * 1024));
        stdout
            .by_ref()
            .take(output_limit as u64)
            .read_to_end(&mut output)
            .map_err(|error| format!("could not read curl response: {error}"))?;
        Ok::<_, String>(output)
    });
    let stderr_reader = thread::spawn(move || {
        let mut output = Vec::new();
        stderr
            .by_ref()
            .take(MAX_HTTP_HEADER_BYTES as u64)
            .read_to_end(&mut output)
            .map_err(|error| format!("could not read curl diagnostics: {error}"))?;
        Ok::<_, String>(output)
    });
    let status = loop {
        if cancel.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err("web research was cancelled".to_string());
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(research_deadline_error());
        }
        match child
            .try_wait()
            .map_err(|error| format!("could not poll curl: {error}"))?
        {
            Some(status) => break status,
            None => thread::sleep(Duration::from_millis(10)),
        }
    };
    let output = stdout_reader
        .join()
        .map_err(|_| "curl response reader stopped unexpectedly".to_string())??;
    let diagnostics = stderr_reader
        .join()
        .map_err(|_| "curl diagnostics reader stopped unexpectedly".to_string())??;
    if output.len() >= output_limit {
        return Err(format!(
            "response exceeded the {MAX_HTTP_BODY_BYTES}-byte body limit"
        ));
    }
    if !status.success() {
        let detail = String::from_utf8_lossy(&diagnostics);
        return Err(if detail.trim().is_empty() {
            format!("curl exited with {status}")
        } else {
            format!("curl failed: {}", detail.trim())
        });
    }
    parse_curl_response(&output)
}

fn curl_timeout_arg(duration: Duration) -> Option<String> {
    // Round down so curl can never be configured beyond the caller's
    // remaining total deadline. A sub-microsecond budget is treated as
    // already expired rather than rounded upward.
    let micros = duration.as_micros();
    (micros > 0).then(|| format!("{}.{:06}", micros / 1_000_000, micros % 1_000_000))
}

fn safe_http_header_value(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 1_024
        && value
            .bytes()
            .all(|byte| byte == b'\t' || (0x20..=0x7e).contains(&byte))
}

fn parse_curl_response(output: &[u8]) -> Result<FetchResponse, String> {
    let mut cursor = 0;
    loop {
        if !output[cursor..].starts_with(b"HTTP/") {
            return Err("web response did not include an HTTP status line".to_string());
        }
        let (header_end, delimiter_len) = find_header_end(&output[cursor..])
            .ok_or_else(|| "web response headers were incomplete".to_string())?;
        if header_end > MAX_HTTP_HEADER_BYTES {
            return Err("web response headers exceeded the safety limit".to_string());
        }
        let absolute_end = cursor + header_end;
        let header = String::from_utf8_lossy(&output[cursor..absolute_end]);
        let mut lines = header.lines();
        let status = lines
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u16>().ok())
            .ok_or_else(|| "web response had an invalid HTTP status".to_string())?;
        let mut headers = HashMap::new();
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let body_start = absolute_end + delimiter_len;
        // A proxy CONNECT or informational response can precede the actual
        // response. Proxy use is disabled, but tolerate the shape defensively.
        if body_start < output.len() && output[body_start..].starts_with(b"HTTP/") && status <= 200
        {
            cursor = body_start;
            continue;
        }
        let body = output[body_start..].to_vec();
        if body.len() > MAX_HTTP_BODY_BYTES {
            return Err(format!(
                "response exceeded the {MAX_HTTP_BODY_BYTES}-byte body limit"
            ));
        }
        return Ok(FetchResponse {
            status,
            content_type: headers.get("content-type").cloned().unwrap_or_default(),
            location: headers.get("location").cloned(),
            etag: headers
                .get("etag")
                .filter(|value| safe_http_header_value(value))
                .cloned(),
            body,
            final_url: None,
        });
    }
}

fn find_header_end(value: &[u8]) -> Option<(usize, usize)> {
    value
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| (index, 4))
        .or_else(|| {
            value
                .windows(2)
                .position(|window| window == b"\n\n")
                .map(|index| (index, 2))
        })
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Mutex,
        },
    };

    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use super::*;

    #[test]
    fn public_http_entry_point_has_a_strict_method_allowlist() {
        assert_eq!(
            PublicHttpMethod::parse("GET").unwrap(),
            PublicHttpMethod::Get
        );
        assert_eq!(
            PublicHttpMethod::parse("head").unwrap(),
            PublicHttpMethod::Head
        );
        let error = fetch_public_http("POST", "https://example.com/").unwrap_err();
        assert!(error.contains("only GET and HEAD"), "{error}");
    }

    #[test]
    fn public_http_entry_point_rejects_non_http_and_private_targets_before_curl() {
        let scheme_error = fetch_public_http("GET", "file:///etc/passwd").unwrap_err();
        assert!(scheme_error.contains("public http://"), "{scheme_error}");

        let private_error = fetch_public_http("GET", "http://127.0.0.1/").unwrap_err();
        assert!(private_error.contains("reserved IPs"), "{private_error}");
    }

    #[derive(Default)]
    struct FakeTransport {
        responses: Mutex<HashMap<String, Result<FetchResponse, String>>>,
        calls: Mutex<Vec<(String, String)>>,
    }

    impl FakeTransport {
        fn insert(&self, url: Url, response: FetchResponse) {
            self.responses
                .lock()
                .unwrap()
                .insert(url.to_string(), Ok(response));
        }

        fn insert_error(&self, url: Url, error: impl Into<String>) {
            self.responses
                .lock()
                .unwrap()
                .insert(url.to_string(), Err(error.into()));
        }

        fn calls(&self) -> Vec<String> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .map(|(url, _)| url.clone())
                .collect()
        }
    }

    #[derive(Debug, Deserialize)]
    struct ClassifierFixtureCase {
        prompt: String,
        needed: bool,
        has_query: bool,
        url_count: usize,
    }

    impl WebTransport for FakeTransport {
        fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String> {
            self.calls
                .lock()
                .unwrap()
                .push((url.to_string(), accept.to_string()));
            self.responses
                .lock()
                .unwrap()
                .get(url.as_str())
                .cloned()
                .unwrap_or_else(|| Err(format!("no fixture for {url}")))
        }
    }

    #[derive(Default)]
    struct ConditionalTransport {
        responses: Mutex<VecDeque<Result<FetchResponse, String>>>,
        etags: Mutex<Vec<Option<String>>>,
    }

    impl ConditionalTransport {
        fn push(&self, response: FetchResponse) {
            self.responses.lock().unwrap().push_back(Ok(response));
        }

        fn calls(&self) -> usize {
            self.etags.lock().unwrap().len()
        }
    }

    impl WebTransport for ConditionalTransport {
        fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String> {
            self.fetch_conditional(url, accept, None)
        }

        fn fetch_conditional(
            &self,
            _url: &Url,
            _accept: &str,
            etag: Option<&str>,
        ) -> Result<FetchResponse, String> {
            self.etags.lock().unwrap().push(etag.map(str::to_string));
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Err("no queued response".to_string()))
        }
    }

    struct BlockingTransport {
        started: Mutex<Option<mpsc::Sender<()>>>,
        cancelled: AtomicBool,
    }

    impl WebTransport for BlockingTransport {
        fn fetch(&self, _url: &Url, _accept: &str) -> Result<FetchResponse, String> {
            Err("blocking transport requires cancellable fetch".to_string())
        }

        fn fetch_cancellable(
            &self,
            _url: &Url,
            _accept: &str,
            _etag: Option<&str>,
            cancel: Option<&tokio_util::sync::CancellationToken>,
            _deadline: Option<Instant>,
            _auth: ProviderAuthIntent,
        ) -> Result<FetchResponse, String> {
            if let Some(started) = self.started.lock().unwrap().take() {
                let _ = started.send(());
            }
            let cancel = cancel.ok_or_else(|| "missing cancellation token".to_string())?;
            while !cancel.is_cancelled() {
                thread::sleep(Duration::from_millis(5));
            }
            self.cancelled.store(true, Ordering::SeqCst);
            Err("web research was cancelled".to_string())
        }
    }

    fn github_fixtures(fake: &FakeTransport, owner: &str, repo: &str, files: &[(&str, &str)]) {
        fake.insert(
            github_api_url(owner, repo, &[]),
            FetchResponse::text(
                200,
                "application/json",
                br#"{"default_branch":"main","private":false}"#.to_vec(),
            ),
        );
        let mut readme_url = github_api_url(owner, repo, &["readme"]);
        readme_url.query_pairs_mut().append_pair("ref", "main");
        fake.insert(
            readme_url,
            FetchResponse::text(
                200,
                "text/plain",
                format!("# {repo} documentation\nPublic repository details."),
            ),
        );
        let tree = json!({
            "tree": files.iter().map(|(path, _)| json!({
                "path": path,
                "type": "blob",
                "size": 128,
            })).collect::<Vec<_>>(),
            "truncated": false,
        });
        fake.insert(
            github_api_url(owner, repo, &["git", "trees", "main"]),
            FetchResponse::text(200, "application/json", tree.to_string()),
        );
        for (path, body) in files {
            fake.insert(
                raw_github_url(owner, repo, "main", path),
                FetchResponse::text(200, "text/plain", body.as_bytes().to_vec()),
            );
        }
    }

    #[test]
    fn ordinary_prompt_skips_without_touching_transport() {
        let fake = FakeTransport::default();
        let response = research_prompt("Turn this checklist into ordered steps.", &fake);
        assert_eq!(response.status, "skipped");
        assert!(!response.triggered);
        assert_eq!(response.reason, "not_needed");
        assert!(response.sources.is_empty());
        assert!(fake.calls().is_empty());
    }

    #[test]
    fn shared_classifier_fixture_matches_frontend_planning() {
        let cases: Vec<ClassifierFixtureCase> = serde_json::from_str(include_str!(
            "../../tests/fixtures/websearch/classifier_cases.json"
        ))
        .expect("shared web-search classifier fixture");
        for case in cases {
            match classify_prompt(&case.prompt) {
                ResearchDecision::Research { urls, query, .. } => {
                    assert!(case.needed, "unexpected research: {}", case.prompt);
                    assert_eq!(query.is_some(), case.has_query, "query: {}", case.prompt);
                    assert_eq!(urls.len(), case.url_count, "URLs: {}", case.prompt);
                }
                ResearchDecision::Skip => {
                    assert!(!case.needed, "unexpected skip: {}", case.prompt);
                    assert!(!case.has_query, "skipped query: {}", case.prompt);
                    assert_eq!(case.url_count, 0, "skipped URLs: {}", case.prompt);
                }
            }
        }
    }

    #[test]
    fn current_and_explicit_frontend_cues_classify_as_search() {
        for prompt in [
            "Look up the Xcode documentation",
            "Read the linked docs for this package",
            "Check the website for recent changes",
            "What is today's weather?",
            "Who is the current CEO?",
            "Show recent events and updates",
            "What's new with Xcode?",
            "Explain this release and cite your sources",
            "Summarize the state as of 1999",
        ] {
            assert!(
                matches!(
                    classify_prompt(prompt),
                    ResearchDecision::Research { query: Some(_), .. }
                ),
                "did not search: {prompt}"
            );
        }
        assert!(matches!(
            classify_prompt("Explain what happened as of 20 minutes ago"),
            ResearchDecision::Skip
        ));
        assert!(matches!(
            classify_prompt("Build a lookup table from this text."),
            ResearchDecision::Skip
        ));
    }

    #[test]
    fn directive_privacy_forms_skip_without_capturing_descriptive_copy() {
        for prompt in [
            "Do not connect to the internet; what is the latest Xcode release?",
            "Answer without external sources; compare the latest releases in my notes.",
        ] {
            assert!(
                matches!(classify_prompt(prompt), ResearchDecision::Skip),
                "privacy directive unexpectedly researched: {prompt}"
            );
        }

        for prompt in [
            "Describe the state shown when there is no network access; search the web for current recovery guidance.",
            "Review the copy no external sources; search the web for current alternatives.",
            "Write UI copy that says do not connect to the internet; search the web for current examples.",
            "Explain the phrase answer without external sources; search the web for current guidance.",
        ] {
            assert!(
                matches!(
                    classify_prompt(prompt),
                    ResearchDecision::Research { query: Some(_), .. }
                ),
                "descriptive copy unexpectedly vetoed research: {prompt}"
            );
        }
    }

    #[test]
    fn expired_total_deadline_blocks_transport_and_degrades_to_warning() {
        let fake = FakeTransport::default();
        let bounded = DeadlineTransport {
            inner: &fake,
            deadline: Instant::now() - Duration::from_millis(1),
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        let response = research_prompt("Read https://example.com/reference", &bounded);
        assert_eq!(response.status, "failed");
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("deadline")));
        assert!(fake.calls().is_empty());
    }

    #[test]
    fn cancelled_research_stops_before_the_next_transport_fetch() {
        let fake = FakeTransport::default();
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let bounded = DeadlineTransport {
            inner: &fake,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel,
        };
        let response = research_prompt("Read https://example.com/reference", &bounded);
        assert_eq!(response.status, "failed");
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("cancelled")));
        assert!(fake.calls().is_empty());
    }

    #[test]
    fn dns_lookup_releases_the_caller_at_the_total_deadline() {
        let url = Url::parse("https://example.com/slow").unwrap();
        let (release_tx, release_rx) = mpsc::channel();
        let result = resolve_public_addresses_with(
            &url,
            None,
            Some(Instant::now() + Duration::from_millis(30)),
            move |_host, _port| {
                let _ = release_rx.recv_timeout(Duration::from_secs(1));
                Ok(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))])
            },
        );
        assert!(result.unwrap_err().contains("deadline"));
        release_tx.send(()).unwrap();
    }

    #[test]
    fn dns_lookup_releases_the_caller_when_cancelled() {
        let url = Url::parse("https://example.com/slow").unwrap();
        let cancel = tokio_util::sync::CancellationToken::new();
        let cancel_worker = cancel.clone();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let canceller = thread::spawn(move || {
            started_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("controlled resolver did not start");
            cancel_worker.cancel();
        });
        let result = resolve_public_addresses_with(
            &url,
            Some(&cancel),
            Some(Instant::now() + Duration::from_secs(1)),
            move |_host, _port| {
                started_tx.send(()).unwrap();
                let _ = release_rx.recv_timeout(Duration::from_secs(1));
                Ok(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))])
            },
        );
        canceller.join().unwrap();
        assert!(result.unwrap_err().contains("cancelled"));
        release_tx.send(()).unwrap();
    }

    #[test]
    fn dns_lookup_rejects_the_entire_answer_set_if_any_address_is_private() {
        let url = Url::parse("https://example.com/mixed-dns").unwrap();
        let result = resolve_public_addresses_with(&url, None, None, |_host, _port| {
            Ok(vec![
                IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34)),
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            ])
        });
        assert!(result.unwrap_err().contains("reserved address"));
    }

    #[test]
    fn dns_lookup_gate_fails_fast_at_capacity_and_recovers_on_drop() {
        let gate = DnsLookupGate::new(2);
        let first = gate.try_acquire().expect("first DNS slot");
        let second = gate.try_acquire().expect("second DNS slot");
        assert!(gate.try_acquire().is_none());
        drop(first);
        assert!(gate.try_acquire().is_some());
        drop(second);
    }

    #[test]
    fn malformed_prompt_github_url_is_canonicalized_without_losing_valid_digits() {
        let (urls, omitted) = extract_prompt_urls(
            "Read https://github.com/PanamaHitek/SmartScale)2 and https://github.com/acme/repo2.",
        );
        assert!(!omitted);
        assert_eq!(
            urls.iter().map(Url::as_str).collect::<Vec<_>>(),
            [
                "https://github.com/PanamaHitek/SmartScale",
                "https://github.com/acme/repo2",
            ]
        );
    }

    #[test]
    fn markdown_escaped_numbered_github_repo_paste_is_recovered() {
        let (urls, omitted) = extract_prompt_urls(
            r#"Read [SmartScale](https://github.com/PanamaHitek/SmartScale\)2)."#,
        );
        assert!(!omitted);
        assert_eq!(
            urls.iter().map(Url::as_str).collect::<Vec<_>>(),
            ["https://github.com/PanamaHitek/SmartScale"]
        );
    }

    #[test]
    fn legal_github_tree_and_blob_parentheses_are_not_treated_as_paste_wrappers() {
        let (urls, omitted) = extract_prompt_urls(
            "Compare https://github.com/acme/widgets/tree/release)2 and https://github.com/acme/widgets/blob/main/spec)",
        );
        assert!(!omitted);
        assert_eq!(
            urls.iter().map(Url::as_str).collect::<Vec<_>>(),
            [
                "https://github.com/acme/widgets/tree/release)2",
                "https://github.com/acme/widgets/blob/main/spec)",
            ]
        );
    }

    #[test]
    fn markdown_closing_parenthesis_is_removed_from_the_destination_only() {
        let (urls, _) = extract_prompt_urls(
            "Read [the spec](https://github.com/acme/widgets/blob/main/spec.md).",
        );
        assert_eq!(
            urls[0].as_str(),
            "https://github.com/acme/widgets/blob/main/spec.md"
        );
    }

    #[test]
    fn malformed_github_slugs_are_not_silently_truncated_into_other_repositories() {
        for raw in [
            "https://github.com/acme!shadow/widgets",
            "https://github.com/acme/widgets!shadow",
            "https://github.com/-acme/widgets",
        ] {
            let parsed = Url::parse(raw).unwrap();
            assert!(
                github_repo(&parsed).is_none(),
                "accepted malformed slug: {raw}"
            );
            assert_eq!(canonical_prompt_url(parsed).as_str(), raw);
        }
    }

    #[test]
    fn prompt_url_scanner_distinguishes_markdown_brackets_from_ipv6_hosts() {
        let (urls, omitted) = extract_prompt_urls(
            "Compare [https://example.com/label](https://example.com/destination) with http://[2606:4700:4700::1111]/dns.",
        );
        assert!(!omitted);
        assert_eq!(
            urls.iter().map(Url::as_str).collect::<Vec<_>>(),
            [
                "https://example.com/label",
                "https://example.com/destination",
                "http://[2606:4700:4700::1111]/dns",
            ]
        );
    }

    #[test]
    fn markdown_urls_are_removed_from_supplemental_queries_without_overlap() {
        let prompt = "Compare [https://example.com/label](https://example.com/destination) and search the web for current alternatives.";
        let query = search_query_from_prompt(prompt);
        assert!(!query.contains("http"));
        assert!(query.contains("search the web"));
        assert!(matches!(
            classify_prompt(prompt),
            ResearchDecision::Research {
                reason: "embedded_urls_and_search",
                query: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn markdown_wrappers_do_not_become_part_of_github_tree_or_blob_paths() {
        let (urls, _) = extract_prompt_urls(
            "Read [tree](https://github.com/acme/widgets/tree/feature/web-research) and [blob](https://github.com/acme/widgets/blob/feature/web-research/src/lib.rs).",
        );
        assert_eq!(
            urls.iter().map(Url::as_str).collect::<Vec<_>>(),
            [
                "https://github.com/acme/widgets/tree/feature/web-research",
                "https://github.com/acme/widgets/blob/feature/web-research/src/lib.rs",
            ]
        );
    }

    #[test]
    fn linked_read_only_language_does_not_trigger_supplemental_search() {
        for prompt in [
            "Read the linked docs https://example.com/reference",
            "Check https://example.com/reference and summarize it",
            "Cite https://example.com/reference in the answer",
        ] {
            assert!(
                matches!(
                    classify_prompt(prompt),
                    ResearchDecision::Research {
                        reason: "embedded_urls",
                        query: None,
                        ..
                    }
                ),
                "link-only prompt unexpectedly added search: {prompt}"
            );
        }
    }

    #[test]
    fn explicit_coordinated_research_adds_a_bounded_supplemental_query() {
        for (prompt, query_fragment) in [
            (
                "Read https://example.com/spec and look up competing libraries too.",
                "look up competing libraries",
            ),
            (
                "Read https://example.com/spec and research alternatives too.",
                "research alternatives",
            ),
            (
                "Read https://example.com/spec; additionally research similar libraries.",
                "research similar libraries",
            ),
            (
                "Read https://example.com/spec, then please also look up alternatives.",
                "look up alternatives",
            ),
        ] {
            match classify_prompt(prompt) {
                ResearchDecision::Research {
                    reason,
                    urls,
                    query: Some(query),
                    ..
                } => {
                    assert_eq!(reason, "embedded_urls_and_search");
                    assert_eq!(urls.len(), 1);
                    assert_eq!(urls[0].as_str(), "https://example.com/spec");
                    assert!(!query.contains("http"));
                    assert!(query.contains(query_fragment));
                }
                decision => panic!("unexpected research decision for {prompt}: {decision:?}"),
            }
        }

        assert!(matches!(
            classify_prompt("Look up https://example.com/spec and summarize it."),
            ResearchDecision::Research {
                reason: "embedded_urls",
                query: None,
                ..
            }
        ));
    }

    #[test]
    fn github_repository_overview_uses_readme_without_fetching_a_file_tree() {
        let fake = FakeTransport::default();
        fake.insert(
            github_api_url("acme", "widgets", &[]),
            FetchResponse::text(
                200,
                "application/json",
                br#"{"default_branch":"main","private":false}"#,
            ),
        );
        let mut readme_url = github_api_url("acme", "widgets", &["readme"]);
        readme_url.query_pairs_mut().append_pair("ref", "main");
        fake.insert(
            readme_url,
            FetchResponse::text(
                200,
                "text/plain",
                "# Widgets\nA concise repository overview.",
            ),
        );

        let response = research_prompt(
            "Can you tell me about this GitHub repo? https://github.com/acme/widgets",
            &fake,
        );

        assert_eq!(response.status, "complete", "{:?}", response.warnings);
        assert_eq!(response.sources.len(), 1);
        assert_eq!(response.sources[0].chunks.len(), 1);
        assert_eq!(
            response.sources[0].chunks[0].path.as_deref(),
            Some("README")
        );
        assert!(response.sources[0].excerpt.contains("repository overview"));
        let calls = fake.calls();
        assert_eq!(calls.len(), 2, "unexpected overview fetches: {calls:?}");
        assert!(!calls.iter().any(|url| url.contains("/git/trees/")));
    }

    #[test]
    fn github_overview_intent_is_distinct_from_implementation_requests() {
        let repo = github_repo(&Url::parse("https://github.com/acme/widgets").unwrap()).unwrap();
        for prompt in [
            "Can you tell me about this GitHub repo? https://github.com/acme/widgets",
            "Summarize this repository: https://github.com/acme/widgets",
            "Give me an overview of Widgets https://github.com/acme/widgets",
            "https://github.com/acme/widgets",
        ] {
            assert!(
                is_github_repository_overview_request(prompt, &repo),
                "missed overview intent: {prompt}"
            );
        }
        for prompt in [
            "Build this app from https://github.com/acme/widgets",
            "Read the code in this repo https://github.com/acme/widgets",
            "Explain the architecture in https://github.com/acme/widgets",
            "Find the protocol implementation in https://github.com/acme/widgets",
            "Summarize this repository and trace its request cancellation logic https://github.com/acme/widgets",
            "Describe this repo and explain how authentication works https://github.com/acme/widgets",
        ] {
            assert!(
                !is_github_repository_overview_request(prompt, &repo),
                "implementation request was reduced to an overview: {prompt}"
            );
        }
    }

    #[test]
    fn github_overview_tries_raw_readme_before_html_when_api_is_unavailable() {
        let fake = FakeTransport::default();
        fake.insert(
            github_api_url("acme", "widgets", &[]),
            FetchResponse::text(403, "application/json", "rate limited"),
        );
        fake.insert(
            raw_github_url("acme", "widgets", "HEAD", "README.md"),
            FetchResponse::text(200, "text/plain", "# Widgets\nRaw overview evidence."),
        );

        let response = research_prompt(
            "Tell me about this repo https://github.com/acme/widgets",
            &fake,
        );

        assert_eq!(response.status, "partial");
        assert_eq!(response.sources.len(), 1);
        assert!(response.sources[0]
            .excerpt
            .contains("Raw overview evidence"));
        let calls = fake.calls();
        assert_eq!(
            calls.len(),
            2,
            "unexpected overview fallback calls: {calls:?}"
        );
        assert!(!calls
            .iter()
            .any(|url| url.starts_with("https://github.com/acme/widgets/tree/")));
    }

    #[test]
    fn github_tree_branch_path_is_preserved_in_fetch_and_provenance() {
        let fake = FakeTransport::default();
        fake.insert(
            github_api_url("acme", "widgets", &[]),
            FetchResponse::text(
                200,
                "application/json",
                br#"{"default_branch":"main","private":false}"#,
            ),
        );
        let branch = "feature/web-research";
        let tree = json!({
            "tree": [{"path":"src/research.rs","type":"blob","size":128}],
            "truncated": false,
        });
        fake.insert(
            github_api_url("acme", "widgets", &["git", "trees", branch]),
            FetchResponse::text(200, "application/json", tree.to_string()),
        );
        let mut readme_url = github_api_url("acme", "widgets", &["readme"]);
        readme_url.query_pairs_mut().append_pair("ref", branch);
        fake.insert(
            readme_url,
            FetchResponse::text(200, "text/plain", "# Widgets branch"),
        );
        fake.insert(
            raw_github_url("acme", "widgets", branch, "src/research.rs"),
            FetchResponse::text(
                200,
                "text/plain",
                "pub fn research_pipeline() -> bool { true }",
            ),
        );

        let prompt = "Read https://github.com/acme/widgets/tree/feature/web-research for the research pipeline.";
        let (urls, _) = extract_prompt_urls(prompt);
        assert_eq!(
            urls[0].as_str(),
            "https://github.com/acme/widgets/tree/feature/web-research"
        );
        let response = research_prompt(prompt, &fake);
        assert_eq!(response.status, "complete", "{:?}", response.warnings);
        assert_eq!(response.sources.len(), 1);
        assert_eq!(response.sources[0].url, urls[0].as_str());
        assert!(response.sources[0]
            .chunks
            .iter()
            .any(|chunk| chunk.path.as_deref() == Some("src/research.rs")));
        assert!(response.sources[0].chunks.iter().any(|chunk| {
            chunk.path.as_deref() == Some("src/research.rs")
                && chunk.url.as_deref()
                    == Some(
                        "https://github.com/acme/widgets/blob/feature/web-research/src/research.rs",
                    )
        }));
        assert!(response.sources[0].chunks.iter().any(|chunk| {
            chunk.path.as_deref() == Some("README")
                && chunk.url.as_deref()
                    == Some("https://github.com/acme/widgets/tree/feature/web-research#readme")
        }));
        assert!(fake
            .calls()
            .iter()
            .any(|url| url.contains("feature%2Fweb-research")));
    }

    #[test]
    fn recursive_tree_body_limit_uses_bounded_sha_walk_for_slash_branch_and_path() {
        let fake = FakeTransport::default();
        fake.insert(
            github_api_url("acme", "widgets", &[]),
            FetchResponse::text(
                200,
                "application/json",
                br#"{"default_branch":"main","private":false}"#,
            ),
        );
        let branch = "feature/web-research";
        for invalid_ref in [
            "feature/web-research/frontend/src",
            "feature/web-research/frontend",
        ] {
            fake.insert(
                github_api_url("acme", "widgets", &["git", "trees", invalid_ref]),
                FetchResponse::text(404, "application/json", "not found"),
            );
        }
        fake.insert_error(
            github_api_url("acme", "widgets", &["git", "trees", branch]),
            "response exceeded the 524288-byte body limit",
        );

        let frontend_sha = "a".repeat(40);
        let src_sha = "b".repeat(40);
        let mut root_entries = vec![json!({
            "path": "frontend",
            "type": "tree",
            "sha": frontend_sha,
        })];
        root_entries.extend((0..20).map(|index| {
            json!({
                "path": format!("unrelated-{index}"),
                "type": "tree",
                "sha": format!("{index:040x}"),
            })
        }));
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", branch),
            FetchResponse::text(
                200,
                "application/json",
                json!({ "tree": root_entries, "truncated": false }).to_string(),
            ),
        );
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", &frontend_sha),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [{ "path": "src", "type": "tree", "sha": src_sha }],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", &src_sha),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [{
                        "path": "research.rs",
                        "type": "blob",
                        "size": 128,
                        "sha": "c".repeat(40),
                    }],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        let mut readme_url = github_api_url("acme", "widgets", &["readme"]);
        readme_url.query_pairs_mut().append_pair("ref", branch);
        fake.insert(
            readme_url,
            FetchResponse::text(200, "text/plain", "# Bounded tree"),
        );
        fake.insert(
            raw_github_url("acme", "widgets", branch, "frontend/src/research.rs"),
            FetchResponse::text(200, "text/plain", "pub fn bounded_research_endpoint() {}"),
        );

        let response = research_prompt(
            "Read https://github.com/acme/widgets/tree/feature/web-research/frontend/src for the research endpoint.",
            &fake,
        );
        assert_eq!(response.sources.len(), 1, "{:?}", response.warnings);
        assert_eq!(
            response.sources[0].url,
            "https://github.com/acme/widgets/tree/feature/web-research/frontend/src"
        );
        assert!(response.sources[0].chunks.iter().any(|chunk| {
            chunk.path.as_deref() == Some("frontend/src/research.rs")
                && chunk.url.as_deref()
                    == Some(
                        "https://github.com/acme/widgets/blob/feature/web-research/frontend/src/research.rs",
                    )
                && chunk.text.contains("bounded_research_endpoint")
        }));
        let calls = fake.calls();
        let tree_api_calls = calls
            .iter()
            .filter(|url| url.starts_with("https://api.github.com/repos/acme/widgets/git/trees/"))
            .count();
        assert!(tree_api_calls <= MAX_GITHUB_API_TREE_PAGES);
        assert!(calls
            .contains(&github_nonrecursive_tree_api_url("acme", "widgets", branch).to_string()));
        assert!(calls.contains(
            &github_nonrecursive_tree_api_url("acme", "widgets", &frontend_sha).to_string()
        ));
        assert!(calls
            .contains(&github_nonrecursive_tree_api_url("acme", "widgets", &src_sha).to_string()));
        let coverage_warnings = response
            .warnings
            .iter()
            .filter(|warning| warning.contains("repository file coverage"))
            .collect::<Vec<_>>();
        assert_eq!(coverage_warnings.len(), 1, "{:?}", response.warnings);
        assert!(coverage_warnings[0].contains("ranked visible files"));
    }

    #[test]
    fn truncated_recursive_tree_is_replaced_by_bounded_nonrecursive_evidence() {
        let fake = FakeTransport::default();
        fake.insert(
            github_api_url("acme", "widgets", &[]),
            FetchResponse::text(
                200,
                "application/json",
                br#"{"default_branch":"main","private":false}"#,
            ),
        );
        fake.insert(
            github_api_url("acme", "widgets", &["git", "trees", "main"]),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [{ "path": "src/unrelated.rs", "type": "blob", "size": 64 }],
                    "truncated": true,
                })
                .to_string(),
            ),
        );
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", "main"),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [{ "path": "src/protocol.rs", "type": "blob", "size": 64 }],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        let mut readme_url = github_api_url("acme", "widgets", &["readme"]);
        readme_url.query_pairs_mut().append_pair("ref", "main");
        fake.insert(
            readme_url,
            FetchResponse::text(200, "text/plain", "# Widgets"),
        );
        fake.insert(
            raw_github_url("acme", "widgets", "main", "src/protocol.rs"),
            FetchResponse::text(200, "text/plain", "pub fn protocol_decoder() {}"),
        );

        let response = research_prompt(
            "Read https://github.com/acme/widgets for the protocol decoder.",
            &fake,
        );
        assert_eq!(response.sources.len(), 1, "{:?}", response.warnings);
        assert!(response.sources[0]
            .chunks
            .iter()
            .any(|chunk| chunk.path.as_deref() == Some("src/protocol.rs")));
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
        assert!(fake
            .calls()
            .contains(&github_nonrecursive_tree_api_url("acme", "widgets", "main").to_string()));
    }

    #[test]
    fn bounded_sha_walk_keeps_identical_trees_mounted_at_distinct_paths() {
        let fake = FakeTransport::default();
        let repo = github_repo(&Url::parse("https://github.com/acme/widgets").unwrap()).unwrap();
        let shared_sha = "d".repeat(40);
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", "main"),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [
                        { "path": "first", "type": "tree", "sha": shared_sha },
                        { "path": "second", "type": "tree", "sha": shared_sha },
                    ],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", &shared_sha),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [{ "path": "shared.rs", "type": "blob", "size": 64 }],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        let mut pages = MAX_GITHUB_API_TREE_PAGES;
        let mut warnings = Vec::new();
        let GithubTreeCandidateResult::Found(tree) =
            fetch_bounded_github_tree(&fake, &repo, "main", None, &[], &mut pages, &mut warnings)
        else {
            panic!("bounded tree walk failed: {warnings:?}");
        };
        let paths = tree
            .tree
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(
            paths,
            HashSet::from(["first/shared.rs", "second/shared.rs"])
        );
        assert_eq!(
            fake.calls()
                .iter()
                .filter(|url| {
                    *url == &github_nonrecursive_tree_api_url("acme", "widgets", &shared_sha)
                        .to_string()
                })
                .count(),
            2
        );
    }

    #[test]
    fn bounded_sha_walk_rejects_an_oversized_combined_path() {
        let fake = FakeTransport::default();
        let repo = github_repo(&Url::parse("https://github.com/acme/widgets").unwrap()).unwrap();
        let directory_sha = "e".repeat(40);
        let long_prefix = "a".repeat(3_000);
        let long_leaf = format!("{}.rs", "b".repeat(1_500));
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", "main"),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [{ "path": long_prefix, "type": "tree", "sha": directory_sha }],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", &directory_sha),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [{ "path": long_leaf, "type": "blob", "size": 64 }],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        let mut pages = MAX_GITHUB_API_TREE_PAGES;
        let mut warnings = Vec::new();
        let GithubTreeCandidateResult::Found(tree) =
            fetch_bounded_github_tree(&fake, &repo, "main", None, &[], &mut pages, &mut warnings)
        else {
            panic!("bounded tree walk failed: {warnings:?}");
        };
        assert!(tree.tree.is_empty());
    }

    #[test]
    fn bounded_sha_walk_marks_a_skipped_child_page_as_incomplete() {
        let fake = FakeTransport::default();
        let repo = github_repo(&Url::parse("https://github.com/acme/widgets").unwrap()).unwrap();
        let good_sha = "1".repeat(40);
        let missing_sha = "2".repeat(40);
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", "main"),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [
                        { "path": "good", "type": "tree", "sha": good_sha },
                        { "path": "missing", "type": "tree", "sha": missing_sha },
                    ],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", &good_sha),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [{ "path": "lib.rs", "type": "blob", "size": 64 }],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        fake.insert_error(
            github_nonrecursive_tree_api_url("acme", "widgets", &missing_sha),
            "deadline expired",
        );

        let mut pages = MAX_GITHUB_API_TREE_PAGES;
        let mut warnings = Vec::new();
        let GithubTreeCandidateResult::Found(tree) =
            fetch_bounded_github_tree(&fake, &repo, "main", None, &[], &mut pages, &mut warnings)
        else {
            panic!("bounded tree walk failed: {warnings:?}");
        };
        assert!(tree.truncated);
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("deadline expired")));
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("page, depth, or entry limit")));
    }

    #[test]
    fn github_warning_normalization_keeps_operational_failures() {
        let mut warnings = vec![
            "acme/widgets: bounded GitHub tree page was unavailable (deadline expired)"
                .to_string(),
            "acme/widgets: bounded GitHub tree traversal stopped at its page, depth, or entry limit"
                .to_string(),
        ];
        normalize_github_tree_warnings(&mut warnings, 0, "acme/widgets", Some(true));
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("deadline expired")));
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("repository file coverage")));

        for size_error in [
            "response exceeded the 524288-byte body limit",
            "curl: (63) Maximum file size exceeded",
            "curl: Exceeded the maximum allowed file size (524288)",
        ] {
            let mut recovered = vec![format!(
                "acme/widgets: GitHub recursive file tree unavailable ({size_error}); trying bounded tree traversal"
            )];
            normalize_github_tree_warnings(&mut recovered, 0, "acme/widgets", Some(false));
            assert!(
                recovered.is_empty(),
                "successful bounded recovery retained obsolete size warning: {recovered:?}"
            );
        }
    }

    #[test]
    fn bounded_sha_walk_stops_at_the_directory_depth_limit() {
        let fake = FakeTransport::default();
        let repo = github_repo(&Url::parse("https://github.com/acme/widgets").unwrap()).unwrap();
        let shas = (0..=MAX_GITHUB_API_TREE_DEPTH)
            .map(|index| format!("{index:040x}"))
            .collect::<Vec<_>>();
        fake.insert(
            github_nonrecursive_tree_api_url("acme", "widgets", "main"),
            FetchResponse::text(
                200,
                "application/json",
                json!({
                    "tree": [{ "path": "level-0", "type": "tree", "sha": shas[0] }],
                    "truncated": false,
                })
                .to_string(),
            ),
        );
        for index in 0..MAX_GITHUB_API_TREE_DEPTH {
            fake.insert(
                github_nonrecursive_tree_api_url("acme", "widgets", &shas[index]),
                FetchResponse::text(
                    200,
                    "application/json",
                    json!({
                        "tree": [{
                            "path": format!("level-{}", index + 1),
                            "type": "tree",
                            "sha": shas[index + 1],
                        }],
                        "truncated": false,
                    })
                    .to_string(),
                ),
            );
        }
        let mut pages = MAX_GITHUB_API_TREE_PAGES;
        let mut warnings = Vec::new();
        let GithubTreeCandidateResult::Found(tree) =
            fetch_bounded_github_tree(&fake, &repo, "main", None, &[], &mut pages, &mut warnings)
        else {
            panic!("bounded tree walk failed: {warnings:?}");
        };
        assert!(tree.truncated);
        assert!(!fake.calls().contains(
            &github_nonrecursive_tree_api_url("acme", "widgets", &shas[MAX_GITHUB_API_TREE_DEPTH])
                .to_string()
        ));
    }

    #[test]
    fn bounded_tree_truncation_and_entry_limits_have_exact_boundaries() {
        let run = |blob_count: usize, provider_truncated: bool, pending_tree: bool| {
            let fake = FakeTransport::default();
            let repo =
                github_repo(&Url::parse("https://github.com/acme/widgets").unwrap()).unwrap();
            let mut entries = (0..blob_count)
                .map(|index| {
                    json!({
                        "path": format!("src/file-{index}.rs"),
                        "type": "blob",
                        "size": 64,
                    })
                })
                .collect::<Vec<_>>();
            if pending_tree {
                entries.push(json!({
                    "path": "pending",
                    "type": "tree",
                    "sha": "f".repeat(40),
                }));
            }
            fake.insert(
                github_nonrecursive_tree_api_url("acme", "widgets", "main"),
                FetchResponse::text(
                    200,
                    "application/json",
                    json!({
                        "tree": entries,
                        "truncated": provider_truncated,
                    })
                    .to_string(),
                ),
            );
            let mut pages = 1;
            let mut warnings = Vec::new();
            let GithubTreeCandidateResult::Found(tree) = fetch_bounded_github_tree(
                &fake,
                &repo,
                "main",
                None,
                &[],
                &mut pages,
                &mut warnings,
            ) else {
                panic!("bounded tree walk failed: {warnings:?}");
            };
            (tree, warnings)
        };

        let (exact, exact_warnings) = run(MAX_GITHUB_API_TREE_ENTRIES, false, false);
        assert!(!exact.truncated);
        assert!(exact_warnings.is_empty());

        let (overflow, overflow_warnings) = run(MAX_GITHUB_API_TREE_ENTRIES + 1, false, false);
        assert!(overflow.truncated);
        assert!(!overflow_warnings.is_empty());

        let (pending, pending_warnings) = run(MAX_GITHUB_API_TREE_ENTRIES - 1, false, true);
        assert!(pending.truncated);
        assert!(!pending_warnings.is_empty());

        let (provider_truncated, provider_warnings) = run(1, true, false);
        assert!(provider_truncated.truncated);
        assert!(!provider_warnings.is_empty());
    }

    #[test]
    fn github_rate_limit_fallback_recovers_slash_branch_and_subpath() {
        let fake = FakeTransport::default();
        fake.insert(
            github_api_url("acme", "widgets", &[]),
            FetchResponse::text(403, "application/json", "rate limited"),
        );
        let branch = "feature/web-research";
        let subpath = "frontend/src";
        let embedded = json!({
            "payload": {
                "codeViewTreeRoute": {
                    "path": subpath,
                    "refInfo": { "name": branch }
                }
            }
        });
        fake.insert(
            github_tree_page_url(
                "acme",
                "widgets",
                "feature/web-research/frontend/src",
                None,
            ),
            FetchResponse::text(
                200,
                "text/html",
                format!(
                    r#"<script type="application/json" data-target="react-app.embeddedData">{embedded}</script>"#
                ),
            ),
        );
        fake.insert(
            github_tree_page_url("acme", "widgets", branch, Some(subpath)),
            FetchResponse::text(
                200,
                "text/html",
                r#"<a href="/acme/widgets/blob/feature/web-research/frontend/src/research.rs">research</a>"#,
            ),
        );
        fake.insert(
            raw_github_url("acme", "widgets", branch, "readme.md"),
            FetchResponse::text(200, "text/plain", "# Branch fallback"),
        );
        fake.insert(
            raw_github_url("acme", "widgets", branch, "frontend/src/research.rs"),
            FetchResponse::text(200, "text/plain", "pub fn bounded_research() {}"),
        );

        let prompt = "Read https://github.com/acme/widgets/tree/feature/web-research/frontend/src for bounded research.";
        let response = research_prompt(prompt, &fake);
        assert_eq!(response.status, "partial", "{:?}", response.warnings);
        assert_eq!(
            response.sources[0].url,
            "https://github.com/acme/widgets/tree/feature/web-research/frontend/src"
        );
        assert!(response.sources[0].chunks.iter().any(|chunk| {
            chunk.path.as_deref() == Some("frontend/src/research.rs")
                && chunk.url.as_deref()
                    == Some(
                        "https://github.com/acme/widgets/blob/feature/web-research/frontend/src/research.rs",
                    )
                && chunk.text.contains("bounded_research")
        }));
        assert!(fake
            .calls()
            .iter()
            .any(|url| { url.contains("feature%2Fweb-research%2Ffrontend%2Fsrc") }));
    }

    #[test]
    fn github_api_403_uses_public_raw_and_html_fallback() {
        let fake = FakeTransport::default();
        fake.insert(
            github_api_url("acme", "widgets", &[]),
            FetchResponse::text(403, "application/json", "rate limited"),
        );
        let rendered_sha = "0123456789abcdef0123456789abcdef01234567";
        let page_url = github_tree_page_url("acme", "widgets", "HEAD", None);
        fake.insert(
            page_url,
            FetchResponse::text(
                200,
                "text/html",
                format!(r#"<a href="/acme/widgets/tree/{rendered_sha}/src">src</a>"#),
            ),
        );
        fake.insert(
            github_tree_page_url("acme", "widgets", "HEAD", Some("src")),
            FetchResponse::text(
                200,
                "text/html",
                format!(r#"<a href="/acme/widgets/tree/{rendered_sha}/src/protocol">protocol</a>"#),
            ),
        );
        fake.insert(
            github_tree_page_url("acme", "widgets", "HEAD", Some("src/protocol")),
            FetchResponse::text(
                200,
                "text/html",
                format!(
                    r#"<a href="/acme/widgets/blob/{rendered_sha}/src/protocol/decoder.rs">decoder</a>"#
                ),
            ),
        );
        fake.insert(
            raw_github_url("acme", "widgets", "HEAD", "readme.md"),
            FetchResponse::text(200, "text/plain", "# Widgets\nPublic fallback docs."),
        );
        fake.insert(
            raw_github_url("acme", "widgets", "HEAD", "src/protocol/decoder.rs"),
            FetchResponse::text(200, "text/plain", "pub fn decode_protocol() {}"),
        );

        let response = research_prompt("Read https://github.com/acme/widgets protocol code", &fake);
        assert_eq!(response.status, "partial");
        assert_eq!(response.sources.len(), 1);
        assert!(response.sources[0].excerpt.contains("decode_protocol"));
        assert!(response.sources[0].chunks.iter().any(|chunk| {
            chunk.path.as_deref() == Some("readme.md")
                && chunk.url.as_deref()
                    == Some("https://github.com/acme/widgets/blob/HEAD/readme.md")
        }));
        assert!(fake
            .calls()
            .iter()
            .any(|url| url.ends_with("/tree/HEAD/src/protocol")));
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("403") && warning.contains("raw")));
    }

    #[test]
    fn private_github_repository_is_not_ingested_with_provider_credentials() {
        let fake = FakeTransport::default();
        fake.insert(
            github_api_url("acme", "private-widgets", &[]),
            FetchResponse::text(
                200,
                "application/json",
                br#"{"default_branch":"main","private":true}"#,
            ),
        );
        let response = research_prompt("Read https://github.com/acme/private-widgets", &fake);
        assert_eq!(response.status, "failed");
        assert!(response.sources.is_empty());
        assert_eq!(fake.calls().len(), 1);
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("private") && warning.contains("public-web")));
    }

    #[test]
    fn github_api_enrichment_requires_explicit_public_metadata() {
        let fake = FakeTransport::default();
        fake.insert(
            github_api_url("acme", "unverified-widgets", &[]),
            FetchResponse::text(200, "application/json", br#"{"default_branch":"main"}"#),
        );
        let response = research_prompt("Read https://github.com/acme/unverified-widgets", &fake);
        assert_eq!(response.status, "failed");
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("could not verify a public repository")));
        assert!(fake
            .calls()
            .iter()
            .filter(|url| url.starts_with("https://api.github.com/"))
            .all(|url| url == "https://api.github.com/repos/acme/unverified-widgets"));
    }

    #[test]
    fn linked_url_and_explicit_search_run_both_research_legs() {
        let fake = FakeTransport::default();
        let prompt = "Read https://example.com/reference and also search the web for current protocol guidance.";
        let direct = Url::parse("https://example.com/reference").unwrap();
        fake.insert(
            direct,
            FetchResponse::text(200, "text/html", "<title>Direct</title>linked evidence"),
        );
        let query = search_query_from_prompt(prompt);
        let mut search_url = Url::parse(BRAVE_SEARCH_ENDPOINT).unwrap();
        search_url
            .query_pairs_mut()
            .append_pair("q", &query)
            .append_pair("source", "web");
        fake.insert(
            search_url,
            FetchResponse::text(
                200,
                "text/html",
                include_str!("../../tests/fixtures/websearch/brave_two_results.html"),
            ),
        );
        fake.insert(
            Url::parse("https://blog.logrocket.com/introducing-rust-borrow-checker/").unwrap(),
            FetchResponse::text(200, "text/html", "<title>Supplement</title>search evidence"),
        );

        let response = research_prompt(prompt, &fake);
        assert_eq!(response.reason, "embedded_urls_and_search");
        assert_eq!(response.query.as_deref(), Some(query.as_str()));
        assert_eq!(response.sources.len(), 2);
        assert!(response.sources[0].excerpt.contains("linked evidence"));
        assert!(response.sources[1].excerpt.contains("search evidence"));
        assert!(fake
            .calls()
            .iter()
            .any(|url| url.starts_with(BRAVE_SEARCH_ENDPOINT)));
    }

    #[test]
    fn linked_evidence_survives_a_failed_supplemental_search() {
        let fake = FakeTransport::default();
        fake.insert(
            Url::parse("https://example.com/reference").unwrap(),
            FetchResponse::text(200, "text/plain", "linked evidence survives"),
        );
        let response = research_prompt(
            "Read https://example.com/reference and search the web for current alternatives.",
            &fake,
        );
        assert_eq!(response.status, "partial");
        assert_eq!(response.sources.len(), 1);
        assert!(response.sources[0]
            .excerpt
            .contains("linked evidence survives"));
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("search")));
    }

    #[test]
    fn source_provenance_uses_the_revalidated_final_url() {
        let fake = FakeTransport::default();
        let original = Url::parse("https://example.com/old").unwrap();
        let final_url = Url::parse("https://docs.example.com/current").unwrap();
        let mut fetched = FetchResponse::text(200, "text/plain", "redirected evidence");
        fetched.final_url = Some(final_url.clone());
        fake.insert(original, fetched);
        let response = research_prompt("Read https://example.com/old", &fake);
        assert_eq!(response.sources[0].url, final_url.as_str());
        assert_eq!(
            response.sources[0].chunks[0].path.as_deref(),
            Some("/current")
        );
        assert_eq!(
            response.sources[0].chunks[0].url.as_deref(),
            Some(final_url.as_str())
        );
    }

    #[test]
    fn redirected_github_raw_chunks_use_the_revalidated_final_url() {
        let fake = FakeTransport::default();
        let repo = github_repo(&Url::parse("https://github.com/acme/widgets").unwrap()).unwrap();
        let code_raw = raw_github_url("acme", "widgets", "main", "src/lib.rs");
        let code_final = Url::parse("https://cdn.example.com/widgets/src/lib.rs").unwrap();
        let mut code_response = FetchResponse::text(200, "text/plain", "pub fn redirected() {}");
        code_response.final_url = Some(code_final.clone());
        fake.insert(code_raw, code_response);
        let mut chunks = Vec::new();
        let mut warnings = Vec::new();
        fetch_github_code_chunk(
            &fake,
            &repo,
            "main",
            "src/lib.rs",
            &[],
            &mut chunks,
            &mut warnings,
        );
        assert!(warnings.is_empty());
        assert_eq!(chunks[0].url.as_deref(), Some(code_final.as_str()));

        let readme_raw = raw_github_url("acme", "widgets", "main", "README.md");
        let readme_final = Url::parse("https://docs.example.com/widgets/readme").unwrap();
        let mut readme_response = FetchResponse::text(200, "text/plain", "# Redirected README");
        readme_response.final_url = Some(readme_final.clone());
        fake.insert(readme_raw, readme_response);
        let readme = fetch_raw_github_readme(&fake, "acme", "widgets", "main").unwrap();
        assert_eq!(readme.path, "README.md");
        assert_eq!(readme.url, readme_final);
    }

    #[test]
    fn two_github_urls_fetch_query_ranked_generic_chunks() {
        let fake = FakeTransport::default();
        let protocol = format!(
            "use crate::types::Frame;\n{}\npub fn decode_frame(bytes: &[u8]) -> Frame {{ verify_checksum(bytes); Frame::parse(bytes) }}",
            "fn unrelated_helper() {}\n".repeat(180)
        );
        github_fixtures(
            &fake,
            "acme",
            "widget-core",
            &[
                ("assets/theme.css", "body {}"),
                ("src/main.rs", "fn main() {}"),
                ("src/protocol.rs", &protocol),
            ],
        );
        github_fixtures(
            &fake,
            "example",
            "client-sdk",
            &[
                ("src/client.ts", "export class Client {}"),
                (
                    "src/transport.ts",
                    "export function transportHandshake() { return 'ready' }",
                ),
                ("tests/client.test.ts", "test('client', () => {})"),
            ],
        );

        let response = research_prompt(
            "Use https://github.com/acme/widget-core and https://github.com/example/client-sdk to plan a frame decode protocol and transport handshake architecture.",
            &fake,
        );
        assert_eq!(response.status, "complete", "{:?}", response.warnings);
        assert_eq!(response.reason, "embedded_urls");
        assert_eq!(response.sources.len(), 2);
        assert_eq!(response.sources[0].id, 1);
        assert_eq!(
            response.sources[0].url,
            "https://github.com/acme/widget-core"
        );
        for marker in ["Source: src/protocol.rs", "decode_frame", "verify_checksum"] {
            assert!(
                response.sources[0].excerpt.contains(marker),
                "lost {marker} in {:?}",
                response.sources[0].excerpt
            );
        }
        assert_eq!(response.sources[1].id, 2);
        assert_eq!(
            response.sources[1].url,
            "https://github.com/example/client-sdk"
        );
        for marker in ["src/transport.ts", "transportHandshake"] {
            assert!(
                response.sources[1].excerpt.contains(marker),
                "lost {marker}"
            );
        }
        assert!(response
            .sources
            .iter()
            .all(|source| !source.chunks.is_empty()));
        assert!(response
            .sources
            .iter()
            .flat_map(|source| &source.chunks)
            .all(|chunk| chunk.url.is_some()));
        assert!(fake.calls().iter().all(|url| !url.contains("duckduckgo")));
    }

    #[test]
    fn explicit_search_fetches_ranked_result_page() {
        let fake = FakeTransport::default();
        let prompt = "Search the web for the current Rust borrow checker documentation";
        let mut search_url = Url::parse(BRAVE_SEARCH_ENDPOINT).unwrap();
        search_url
            .query_pairs_mut()
            .append_pair("q", prompt)
            .append_pair("source", "web");
        fake.insert(
            search_url,
            FetchResponse::text(
                200,
                "text/html",
                include_str!("../../tests/fixtures/websearch/brave_two_results.html"),
            ),
        );
        let first =
            Url::parse("https://blog.logrocket.com/introducing-rust-borrow-checker/").unwrap();
        fake.insert(
            first,
            FetchResponse::text(
                200,
                "text/html",
                "<html><title>Borrow checker guide</title><body>Ownership keeps Rust code safe.</body></html>",
            ),
        );
        // The fixture has a second result. Let it fail to prove partial results
        // still survive rather than pretending the whole search was empty.
        let response = research_prompt(prompt, &fake);
        assert_eq!(response.reason, "explicit_search");
        assert_eq!(response.status, "partial");
        assert_eq!(response.sources.len(), 1);
        assert!(response.sources[0]
            .excerpt
            .contains("Ownership keeps Rust code safe"));
        assert!(response.query.as_deref().unwrap().contains("current Rust"));
        assert!(fake.calls().iter().all(|url| !url.contains("duckduckgo")));
    }

    #[test]
    fn ddg_202_or_captcha_is_a_provider_warning_not_no_results() {
        let fake = FakeTransport::default();
        let prompt = "What is the latest Xcode release?";
        let mut brave_url = Url::parse(BRAVE_SEARCH_ENDPOINT).unwrap();
        brave_url
            .query_pairs_mut()
            .append_pair("q", prompt)
            .append_pair("source", "web");
        fake.insert(
            brave_url,
            FetchResponse::text(503, "text/html", "temporarily unavailable"),
        );
        let mut search_url = Url::parse(DDG_SEARCH_ENDPOINT).unwrap();
        search_url.query_pairs_mut().append_pair("q", prompt);
        fake.insert(
            search_url,
            FetchResponse::text(
                202,
                "text/html",
                "<form class=challenge-form>CAPTCHA</form>",
            ),
        );
        let response = research_prompt(prompt, &fake);
        assert_eq!(response.status, "failed");
        assert!(response.triggered);
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("Brave") && warning.contains("503")));
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("DuckDuckGo Lite") && warning.contains("challenged")));
        assert!(response
            .warnings
            .iter()
            .all(|warning| !warning.contains("no usable results")));
    }

    #[test]
    fn bing_fallback_survives_brave_and_ddg_challenges() {
        let fake = FakeTransport::default();
        let prompt = "What is the latest Xcode release?";
        let mut brave_url = Url::parse(BRAVE_SEARCH_ENDPOINT).unwrap();
        brave_url
            .query_pairs_mut()
            .append_pair("q", prompt)
            .append_pair("source", "web");
        fake.insert(
            brave_url,
            FetchResponse::text(
                200,
                "text/html",
                "<title>Brave Search</title><p>We scheduled a captcha for this suspicious request.</p>",
            ),
        );
        let mut ddg_url = Url::parse(DDG_SEARCH_ENDPOINT).unwrap();
        ddg_url.query_pairs_mut().append_pair("q", prompt);
        fake.insert(
            ddg_url,
            FetchResponse::text(
                202,
                "text/html",
                "<form class=challenge-form>CAPTCHA</form>",
            ),
        );
        let mut bing_url = Url::parse(BING_SEARCH_ENDPOINT).unwrap();
        bing_url.query_pairs_mut().append_pair("q", prompt);
        fake.insert(
            bing_url,
            FetchResponse::text(
                200,
                "text/html",
                r#"<ol><li class="b_algo"><h2><a target="_blank" href="https://www.bing.com/ck/a?!&amp;&amp;p=fixture&amp;u=a1aHR0cHM6Ly9kZXZlbG9wZXIuYXBwbGUuY29tL3hjb2RlLw&amp;ntb=1"><strong>Xcode</strong> - Apple Developer</a></h2></li></ol>"#,
            ),
        );
        let source_url = Url::parse("https://developer.apple.com/xcode/").unwrap();
        fake.insert(
            source_url,
            FetchResponse::text(
                200,
                "text/html",
                "<title>Xcode - Apple Developer</title><main>Current Xcode release details.</main>",
            ),
        );

        let response = research_prompt(prompt, &fake);
        assert_eq!(response.status, "partial");
        assert_eq!(response.sources.len(), 1);
        assert_eq!(
            response.sources[0].url,
            "https://developer.apple.com/xcode/"
        );
        assert!(response.sources[0].excerpt.contains("Current Xcode"));
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("Brave") && warning.contains("challenged")));
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("DuckDuckGo Lite") && warning.contains("challenged")));
    }

    #[test]
    fn unsafe_targets_are_rejected_before_transport() {
        let fake = FakeTransport::default();
        for prompt in [
            "Read http://127.0.0.1/admin".to_string(),
            "Read http://169.254.169.254/latest/meta-data".to_string(),
            format!("Read http://{}.0.0.2/private", 10),
            "Read http://192.88.99.1/relay".to_string(),
            "Read https://user:secret@example.com/private".to_string(),
            "Read https://user:secret@github.com/acme/repo".to_string(),
            "Read https://github.com:444/acme/repo".to_string(),
            "Read http://[::1]/private".to_string(),
            "Read http://[64:ff9b:1::7f00:1]/private".to_string(),
            "Read http://[3fff::1]/reserved".to_string(),
            "Read http://[4000::1]/reserved".to_string(),
            "Read http://[100::1]/discard".to_string(),
            "Read http://[2001::1]/teredo".to_string(),
            "Read http://[2001:2::1]/benchmarking".to_string(),
            "Read http://[2001:2:0:ffff::1]/benchmarking".to_string(),
            "Read http://[2001:10::1]/deprecated-orchid".to_string(),
            "Read http://[2001:20::1]/orchid".to_string(),
            "Read http://[2001:2f:ffff::1]/orchid".to_string(),
            "Read http://[2002:7f00:1::]/six-to-four".to_string(),
        ] {
            let response = research_prompt(&prompt, &fake);
            assert_eq!(response.status, "failed", "{prompt}");
            assert!(response.sources.is_empty());
        }
        assert!(fake.calls().is_empty());
    }

    #[test]
    fn ipv6_special_use_masks_do_not_overreach_adjacent_global_addresses() {
        for address in [
            "2001:2::1",
            "2001:2:0:ffff::1",
            "2001:10::1",
            "2001:1f:ffff::1",
            "2001:20::1",
            "2001:2f:ffff::1",
        ] {
            assert!(!is_public_ipv6(address.parse().unwrap()), "{address}");
        }
        for address in [
            "2001:2:1::1",
            "2001:4860:4860::8888",
            "2606:4700:4700::1111",
        ] {
            assert!(is_public_ipv6(address.parse().unwrap()), "{address}");
        }
    }

    #[test]
    fn binary_and_non_http_responses_are_rejected() {
        assert!(validate_url_shape(&Url::parse("file:///etc/passwd").unwrap()).is_err());
        assert!(response_excerpt(&FetchResponse::text(
            200,
            "image/png",
            b"not really a png".to_vec(),
        ))
        .is_none());
        assert!(response_excerpt(&FetchResponse::text(
            200,
            "application/octet-stream",
            b"text-looking binary\0payload".to_vec(),
        ))
        .is_none());
    }

    #[test]
    fn redirect_target_is_revalidated() {
        let base = Url::parse("https://example.com/start").unwrap();
        let private_target = format!("http://{}.168.1.2/secret", 192);
        let error = validated_redirect_target(&base, &private_target).unwrap_err();
        assert!(error.contains("not allowed"));
        let safe = validated_redirect_target(&base, "/next").unwrap();
        assert_eq!(safe.as_str(), "https://example.com/next");
        let downgrade = validated_redirect_target(&base, "http://example.com/next").unwrap_err();
        assert!(downgrade.contains("downgrade"));
    }

    #[test]
    fn source_selection_prioritizes_relevant_code() {
        let entries = [
            ("docs/guide.md", "blob"),
            ("style.css", "blob"),
            ("src/main.rs", "blob"),
            ("src/transport_adapter.rs", "blob"),
            ("src/protocol_decoder.rs", "blob"),
            ("node_modules/no.js", "blob"),
        ]
        .into_iter()
        .map(|(path, kind)| GithubTreeEntry {
            path: path.to_string(),
            kind: kind.to_string(),
            size: Some(100),
            sha: None,
        })
        .collect::<Vec<_>>();
        assert_eq!(
            select_github_source_paths(&entries, None, &query_terms("protocol decoder transport"),),
            [
                "src/protocol_decoder.rs",
                "src/transport_adapter.rs",
                "src/main.rs",
            ]
        );
    }

    #[test]
    fn curl_response_parser_bounds_and_extracts_headers() {
        let response = parse_curl_response(
            b"HTTP/2 302\r\nlocation: https://example.com/next\r\ncontent-type: text/plain\r\netag: W/\"abc\"\r\n\r\nredirect",
        )
        .unwrap();
        assert_eq!(response.status, 302);
        assert_eq!(
            response.location.as_deref(),
            Some("https://example.com/next")
        );
        assert_eq!(response.body, b"redirect");
        assert_eq!(response.etag.as_deref(), Some("W/\"abc\""));
        assert!(safe_http_header_value("W/\"safe\""));
        assert!(!safe_http_header_value("safe\r\nInjected: yes"));
    }

    #[test]
    fn curl_timeout_argument_never_rounds_past_the_remaining_budget() {
        assert_eq!(curl_timeout_arg(Duration::ZERO), None);
        assert_eq!(
            curl_timeout_arg(Duration::from_nanos(999)),
            None,
            "sub-microsecond budget must be treated as expired"
        );
        assert_eq!(
            curl_timeout_arg(Duration::from_nanos(1_999_999_999)).as_deref(),
            Some("1.999999")
        );
        assert_eq!(
            curl_timeout_arg(Duration::from_secs(15)).as_deref(),
            Some("15.000000")
        );
    }

    #[test]
    fn source_cache_uses_fresh_entry_without_refetching() {
        let inner = Arc::new(ConditionalTransport::default());
        let mut response = FetchResponse::text(200, "text/plain", "cached evidence");
        response.etag = Some("\"v1\"".to_string());
        inner.push(response);
        let cache = CachedWebTransport::new(inner.clone(), Duration::from_secs(60), 4);
        let url = Url::parse("https://example.com/source").unwrap();
        let first = cache.fetch(&url, "text/plain").unwrap();
        let second = cache.fetch(&url, "text/plain").unwrap();
        assert_eq!(first.body, second.body);
        assert_eq!(inner.calls(), 1);
    }

    #[test]
    fn stale_etag_304_reuses_cached_source_body() {
        let inner = Arc::new(ConditionalTransport::default());
        let mut first = FetchResponse::text(200, "text/plain", "bounded source body");
        first.etag = Some("W/\"v1\"".to_string());
        inner.push(first);
        inner.push(FetchResponse::text(304, "text/plain", Vec::new()));
        let cache = CachedWebTransport::new(inner.clone(), Duration::ZERO, 4);
        let url = Url::parse("https://example.com/source").unwrap();
        let first = cache.fetch(&url, "text/plain").unwrap();
        let second = cache.fetch(&url, "text/plain").unwrap();
        assert_eq!(first.body, b"bounded source body");
        assert_eq!(second.body, first.body);
        assert_eq!(
            inner.etags.lock().unwrap().as_slice(),
            &[None, Some("W/\"v1\"".to_string())]
        );
    }

    #[test]
    fn verified_public_github_sources_cache_but_unverified_metadata_does_not() {
        let public_inner = Arc::new(ConditionalTransport::default());
        public_inner.push(FetchResponse::text(
            200,
            "application/json",
            r#"{"tree":[]}"#,
        ));
        let public_cache =
            CachedWebTransport::new(public_inner.clone(), Duration::from_secs(60), 4);
        let tree_url = github_api_url("acme", "widgets", &["git", "trees", "main"]);
        public_cache
            .fetch_github_verified_public_api(&tree_url, "application/vnd.github+json")
            .unwrap();
        public_cache
            .fetch_github_verified_public_api(&tree_url, "application/vnd.github+json")
            .unwrap();
        assert_eq!(public_inner.calls(), 1);

        let metadata_inner = Arc::new(ConditionalTransport::default());
        for _ in 0..2 {
            metadata_inner.push(FetchResponse::text(
                200,
                "application/json",
                r#"{"default_branch":"main","private":false}"#,
            ));
        }
        let metadata_cache =
            CachedWebTransport::new(metadata_inner.clone(), Duration::from_secs(60), 4);
        let metadata_url = github_api_url("acme", "widgets", &[]);
        metadata_cache
            .fetch_github_repository_api(&metadata_url, "application/vnd.github+json")
            .unwrap();
        metadata_cache
            .fetch_github_repository_api(&metadata_url, "application/vnd.github+json")
            .unwrap();
        assert_eq!(metadata_inner.calls(), 2);
    }

    #[test]
    fn github_provider_token_is_scoped_to_managed_repository_requests() {
        let token = Some("sentinel_token");
        let metadata_url = Url::parse("https://api.github.com/repos/acme/widgets").unwrap();
        let metadata_scope = github_auth_scope_for_request(
            &metadata_url,
            "application/vnd.github+json",
            ProviderAuthIntent::GithubRepositoryMetadata,
        )
        .unwrap();
        assert_eq!(
            github_token_for_request(
                &metadata_url,
                "application/vnd.github+json",
                ProviderAuthIntent::GithubRepositoryMetadata,
                Some(&metadata_scope),
                token
            ),
            token
        );
        let tree_url = Url::parse(
            "https://api.github.com/repos/acme/widgets/git/trees/feature%2Fweb-research",
        )
        .unwrap();
        let tree_scope = github_auth_scope_for_request(
            &tree_url,
            "application/vnd.github+json",
            ProviderAuthIntent::GithubVerifiedPublicApi,
        )
        .unwrap();
        assert_eq!(
            github_token_for_request(
                &tree_url,
                "application/vnd.github+json",
                ProviderAuthIntent::GithubVerifiedPublicApi,
                Some(&tree_scope),
                token
            ),
            token
        );
        for (url, accept, auth) in [
            (
                "https://api.github.com/repos/acme/widgets",
                "application/vnd.github+json",
                ProviderAuthIntent::None,
            ),
            (
                "https://api.github.com/repos/acme/widgets/readme",
                "text/html,text/plain,application/json;q=0.8",
                ProviderAuthIntent::GithubRepositoryMetadata,
            ),
            (
                "https://api.github.com/user/repos",
                "application/vnd.github+json",
                ProviderAuthIntent::GithubRepositoryMetadata,
            ),
            (
                "https://github.com/acme/widgets",
                "application/vnd.github+json",
                ProviderAuthIntent::GithubRepositoryMetadata,
            ),
            (
                "https://raw.githubusercontent.com/acme/widgets/main/README.md",
                "application/vnd.github.raw+json",
                ProviderAuthIntent::GithubRepositoryMetadata,
            ),
            (
                "https://api.github.com.evil.example/repos/acme/widgets",
                "application/vnd.github+json",
                ProviderAuthIntent::GithubRepositoryMetadata,
            ),
            (
                "http://api.github.com/repos/acme/widgets",
                "application/vnd.github+json",
                ProviderAuthIntent::GithubRepositoryMetadata,
            ),
        ] {
            let url = Url::parse(url).unwrap();
            let scope = github_auth_scope_for_request(&url, accept, auth);
            assert_eq!(
                github_token_for_request(&url, accept, auth, scope.as_ref(), token,),
                None
            );
        }
    }

    #[test]
    fn github_provider_scope_rejects_cross_repository_api_redirects() {
        let original = Url::parse("https://api.github.com/repos/acme/widgets").unwrap();
        let scope = github_auth_scope_for_request(
            &original,
            "application/vnd.github+json",
            ProviderAuthIntent::GithubRepositoryMetadata,
        )
        .unwrap();
        let cross_repo = Url::parse("https://api.github.com/repos/other/private-widgets").unwrap();
        assert!(validate_github_redirect_scope(&cross_repo, Some(&scope))
            .unwrap_err()
            .contains("changed repository scope"));
        assert_eq!(
            github_token_for_request(
                &cross_repo,
                "application/vnd.github+json",
                ProviderAuthIntent::GithubRepositoryMetadata,
                Some(&scope),
                Some("sentinel_token"),
            ),
            None
        );

        let cross_host =
            Url::parse("https://raw.githubusercontent.com/other/private-widgets/main/README.md")
                .unwrap();
        assert!(validate_github_redirect_scope(&cross_host, Some(&scope)).is_ok());
        assert_eq!(
            github_token_for_request(
                &cross_host,
                "application/vnd.github+json",
                ProviderAuthIntent::GithubRepositoryMetadata,
                Some(&scope),
                Some("sentinel_token"),
            ),
            None
        );
    }

    #[tokio::test]
    async fn dropping_the_http_request_cancels_an_in_flight_fetch() {
        let (started_tx, started_rx) = mpsc::channel();
        let transport = Arc::new(BlockingTransport {
            started: Mutex::new(Some(started_tx)),
            cancelled: AtomicBool::new(false),
        });
        let state = AppState::default().with_web_research_transport_for_tests(transport.clone());
        let app = super::super::router_with_state(state);
        let request = Request::builder()
            .method("POST")
            .uri("/api/web/research")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"prompt":"Read https://example.com/slow"}"#))
            .unwrap();
        let task = tokio::spawn(async move { app.oneshot(request).await });
        tokio::task::spawn_blocking(move || started_rx.recv_timeout(Duration::from_secs(2)))
            .await
            .unwrap()
            .expect("research transport did not start");
        task.abort();
        let _ = task.await;
        tokio::time::timeout(Duration::from_secs(2), async {
            while !transport.cancelled.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("in-flight research did not observe request cancellation");
    }

    #[tokio::test]
    async fn api_route_returns_typed_sources_and_skips_irrelevant_prompts() {
        let fake = Arc::new(FakeTransport::default());
        let url = Url::parse("https://example.com/reference").unwrap();
        fake.insert(
            url,
            FetchResponse::text(
                200,
                "text/html",
                "<title>Reference</title><main>grounded fact marker</main>",
            ),
        );
        let state = AppState::default().with_web_research_transport_for_tests(fake.clone());
        let app = super::super::router_with_state(state);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/web/research")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"prompt":"Read https://example.com/reference"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["triggered"], true);
        assert_eq!(body["sources"][0]["url"], "https://example.com/reference");
        assert!(body["sources"][0]["excerpt"]
            .as_str()
            .unwrap()
            .contains("grounded fact marker"));

        let before = fake.calls().len();
        let skipped = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/web/research")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt":"Rewrite this sentence."}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(skipped.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(skipped.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["status"], "skipped");
        assert_eq!(body["triggered"], false);
        assert_eq!(fake.calls().len(), before);
    }
}
