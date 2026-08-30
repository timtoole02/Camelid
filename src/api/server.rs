//! Startup-resolved network policy for the HTTP server.
//!
//! Security-sensitive settings are parsed once before the listener begins
//! serving. Request handlers receive immutable, redacted state; no handler
//! reads API keys or policy knobs from the process environment.

use std::{
    fmt,
    io::{Error, ErrorKind, Result},
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
};

use axum::{
    extract::{Request, State},
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE, WWW_AUTHENTICATE},
        HeaderName, HeaderValue, Method, StatusCode,
    },
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::tls_pair::{resolve_tls, TlsFiles};

pub(crate) const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const DEFAULT_MAX_PROMPT_TOKENS: usize = 131_072;
pub(crate) const DEFAULT_MAX_GENERATION_TOKENS: u32 = 8_192;
pub(crate) const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_API_KEY_BYTES: usize = 4 * 1024;

static X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");

/// HTTP surface exposed after authentication.
///
/// `LanChatOnly` is deliberately an allowlist. Adding a route to the main
/// router does not make it remotely reachable until this contract names the
/// exact method and path as part of the embedded Chat UI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiSurface {
    #[default]
    Full,
    LanChatOnly,
}

impl ApiSurface {
    fn allows(self, method: &Method, path: &str) -> bool {
        match self {
            Self::Full => true,
            Self::LanChatOnly => matches!(
                (method, path),
                (&Method::GET, "/v1/models")
                    | (&Method::GET, "/api/capabilities")
                    | (&Method::GET, "/api/models/current")
                    | (&Method::GET, "/api/models/local")
                    | (&Method::GET, "/api/models/catalog/downloads")
                    | (&Method::POST, "/api/models/load")
                    | (&Method::POST, "/api/web/research")
                    | (&Method::POST, "/v1/chat/completions")
            ),
        }
    }

    fn forbidden(self) -> Response {
        debug_assert_eq!(self, Self::LanChatOnly);
        (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": {
                    "code": "lan_chat_only",
                    "message": "this Camelid listener serves authenticated Chat and local model switching only",
                    "type": "permission_error"
                }
            })),
        )
            .into_response()
    }
}

/// Public serve-time inputs. The CLI fills these directly; embedded callers
/// can use `Default` and retain Camelid's anonymous loopback behavior.
#[derive(Clone)]
pub struct ServeOptions {
    pub api_key: Option<String>,
    pub api_key_file: Option<PathBuf>,
    pub cors_origins: Vec<String>,
    pub allow_unauthenticated_remote: bool,
    pub allow_cleartext_remote: bool,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    pub max_request_body_bytes: usize,
    pub max_prompt_tokens: usize,
    pub max_generation_tokens: u32,
    pub max_download_bytes: u64,
    pub api_surface: ApiSurface,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            api_key: None,
            api_key_file: None,
            cors_origins: Vec::new(),
            allow_unauthenticated_remote: false,
            allow_cleartext_remote: false,
            tls_cert: None,
            tls_key: None,
            max_request_body_bytes: DEFAULT_MAX_REQUEST_BODY_BYTES,
            max_prompt_tokens: DEFAULT_MAX_PROMPT_TOKENS,
            max_generation_tokens: DEFAULT_MAX_GENERATION_TOKENS,
            max_download_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            api_surface: ApiSurface::Full,
        }
    }
}

impl fmt::Debug for ServeOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServeOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("api_key_file", &self.api_key_file)
            .field("cors_origins", &self.cors_origins)
            .field(
                "allow_unauthenticated_remote",
                &self.allow_unauthenticated_remote,
            )
            .field("allow_cleartext_remote", &self.allow_cleartext_remote)
            .field("tls_cert", &self.tls_cert)
            .field("tls_key", &self.tls_key.as_ref().map(|_| "[REDACTED PATH]"))
            .field("max_request_body_bytes", &self.max_request_body_bytes)
            .field("max_prompt_tokens", &self.max_prompt_tokens)
            .field("max_generation_tokens", &self.max_generation_tokens)
            .field("max_download_bytes", &self.max_download_bytes)
            .field("api_surface", &self.api_surface)
            .finish()
    }
}

#[derive(Clone)]
pub(crate) struct ApiAuth {
    key: Option<Arc<[u8]>>,
}

impl fmt::Debug for ApiAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiAuth")
            .field("enabled", &self.enabled())
            .finish()
    }
}

impl ApiAuth {
    /// Require `key`, or accept every request when it is `None`.
    pub(crate) fn new(key: Option<String>) -> Self {
        Self {
            key: key.map(|key| Arc::<[u8]>::from(key.into_bytes())),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.key.is_some()
    }

    pub(crate) fn bearer_header_line(&self) -> Option<String> {
        let key = self.key.as_deref()?;
        let key = std::str::from_utf8(key).ok()?;
        Some(format!("Authorization: Bearer {key}\r\n"))
    }

    pub(crate) fn accepts(&self, headers: &axum::http::HeaderMap) -> bool {
        let Some(expected) = self.key.as_deref() else {
            return true;
        };
        let bearer = headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_bearer);
        let api_key = headers
            .get(&X_API_KEY)
            .and_then(|value| value.to_str().ok());
        bearer
            .or(api_key)
            .is_some_and(|candidate| constant_time_eq(expected, candidate.as_bytes()))
    }

    /// Whether `path` is behind the key at all.
    ///
    /// An associated function rather than a free one so that anything reusing
    /// [`ApiAuth`] — the fabric proxy does — applies the same exemptions. The
    /// health routes are public on purpose: a load balancer has to be able to
    /// probe a server it holds no credential for.
    pub(crate) fn route_requires_auth(path: &str) -> bool {
        !matches!(path, "/health" | "/v1/health" | "/" | "/index.html")
            && !path.starts_with("/assets/")
            && !path.starts_with("/favicon")
    }

    /// The refusal a caller without a usable key gets.
    ///
    /// Built here so every front door refuses in the same words, with the same
    /// challenge header, and none of them names which key would have worked.
    pub(crate) fn unauthorized() -> Response {
        let mut response = (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "code": "unauthorized",
                    "message": "provide Authorization: Bearer <key> or X-API-Key",
                    "type": "authentication_error"
                }
            })),
        )
            .into_response();
        response.headers_mut().insert(
            WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer realm=\"camelid\""),
        );
        response
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServerLimits {
    pub(crate) max_request_body_bytes: usize,
    pub(crate) max_prompt_tokens: usize,
    pub(crate) max_generation_tokens: u32,
    pub(crate) max_download_bytes: u64,
}

#[derive(Clone)]
pub(crate) struct ServerPolicy {
    pub(crate) auth: ApiAuth,
    pub(crate) limits: ServerLimits,
    pub(crate) tls: Option<TlsFiles>,
    cors_origins: Arc<[HeaderValue]>,
    pub(crate) remote_unauthenticated_override: bool,
    api_surface: ApiSurface,
}

impl fmt::Debug for ServerPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerPolicy")
            .field("auth", &self.auth)
            .field("limits", &self.limits)
            .field("tls_enabled", &self.tls.is_some())
            .field("cors_origin_count", &self.cors_origins.len())
            .field(
                "remote_unauthenticated_override",
                &self.remote_unauthenticated_override,
            )
            .field("api_surface", &self.api_surface)
            .finish()
    }
}

impl ServerPolicy {
    pub(crate) fn resolve(addr: SocketAddr, options: ServeOptions) -> Result<Self> {
        validate_positive("max request body bytes", options.max_request_body_bytes)?;
        validate_positive("max prompt tokens", options.max_prompt_tokens)?;
        validate_positive("max generation tokens", options.max_generation_tokens)?;
        validate_positive("max download bytes", options.max_download_bytes)?;

        let key = resolve_api_key(options.api_key, options.api_key_file)?;

        if options.api_surface == ApiSurface::LanChatOnly && key.is_none() {
            return Err(Error::new(
                ErrorKind::PermissionDenied,
                "--lan-chat-only requires --api-key or --api-key-file",
            ));
        }

        if !addr.ip().is_loopback() && key.is_none() && !options.allow_unauthenticated_remote {
            return Err(Error::new(
                ErrorKind::PermissionDenied,
                format!(
                    "refusing unauthenticated non-loopback listener {addr}; configure \
                     --api-key/--api-key-file or explicitly acknowledge the risk with \
                     --allow-unauthenticated-remote"
                ),
            ));
        }

        let tls = resolve_tls(options.tls_cert, options.tls_key)?;

        // Without TLS the whole request path is readable, including any
        // configured credential and every prompt and completion.
        // `crate::fabric::server` refuses this same shape on the same three
        // conditions.
        if !addr.ip().is_loopback() && tls.is_none() && !options.allow_cleartext_remote {
            return Err(Error::new(
                ErrorKind::PermissionDenied,
                format!(
                    "refusing cleartext non-loopback listener {addr}; any credentials, prompts, \
                     and completions would cross the network unencrypted. Bind a loopback \
                     address, configure --tls-cert/--tls-key, or explicitly acknowledge the \
                     risk with --allow-cleartext-remote"
                ),
            ));
        }

        let cors_origins = options
            .cors_origins
            .iter()
            .map(|origin| parse_origin(origin))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            auth: ApiAuth::new(key),
            limits: ServerLimits {
                max_request_body_bytes: options.max_request_body_bytes,
                max_prompt_tokens: options.max_prompt_tokens,
                max_generation_tokens: options.max_generation_tokens,
                max_download_bytes: options.max_download_bytes,
            },
            tls,
            cors_origins: cors_origins.into(),
            remote_unauthenticated_override: !addr.ip().is_loopback()
                && options.allow_unauthenticated_remote,
            api_surface: options.api_surface,
        })
    }

    pub(crate) fn loopback_default() -> Self {
        Self::resolve(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            ServeOptions::default(),
        )
        .expect("built-in loopback policy is valid")
    }

    pub(crate) fn cors_layer(&self) -> CorsLayer {
        let layer = CorsLayer::new()
            .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
            .allow_headers([AUTHORIZATION, CONTENT_TYPE, X_API_KEY.clone()]);
        if self.cors_origins.is_empty() {
            layer
        } else {
            layer.allow_origin(AllowOrigin::list(self.cors_origins.iter().cloned()))
        }
    }

    pub(crate) fn cors_origin_count(&self) -> usize {
        self.cors_origins.len()
    }

    pub(crate) fn request_policy(&self) -> (ApiAuth, ApiSurface) {
        (self.auth.clone(), self.api_surface)
    }

    pub(crate) fn api_surface(&self) -> ApiSurface {
        self.api_surface
    }
}

pub(crate) async fn authenticate(
    State((auth, surface)): State<(ApiAuth, ApiSurface)>,
    request: Request,
    next: Next,
) -> Response {
    if request.method() == Method::OPTIONS || !ApiAuth::route_requires_auth(request.uri().path()) {
        return next.run(request).await;
    }
    if !auth.accepts(request.headers()) {
        return ApiAuth::unauthorized();
    }
    if !surface.allows(request.method(), request.uri().path()) {
        return surface.forbidden();
    }
    next.run(request).await
}

fn parse_bearer(value: &str) -> Option<&str> {
    let (scheme, credential) = value.split_once(' ')?;
    (scheme.eq_ignore_ascii_case("bearer") && !credential.is_empty()).then_some(credential)
}

fn constant_time_eq(expected: &[u8], candidate: &[u8]) -> bool {
    let max = expected.len().max(candidate.len());
    let mut difference = expected.len() ^ candidate.len();
    for index in 0..max {
        difference |= usize::from(
            expected.get(index).copied().unwrap_or(0) ^ candidate.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

fn validate_api_key(key: String) -> Result<String> {
    let key = key.trim().to_string();
    if key.is_empty() {
        return Err(invalid("API key must not be empty"));
    }
    if key.len() > MAX_API_KEY_BYTES {
        return Err(invalid(format!(
            "API key exceeds the {MAX_API_KEY_BYTES}-byte limit"
        )));
    }
    if key.chars().any(char::is_control) {
        return Err(invalid("API key must not contain control characters"));
    }
    Ok(key)
}

/// Resolve the key clients must present, from a value or a file.
///
/// Shared with the fabric proxy: two front doors validating the same kind of
/// secret by two sets of rules is how they end up disagreeing about what a
/// valid key is.
pub(crate) fn resolve_api_key(
    api_key: Option<String>,
    api_key_file: Option<PathBuf>,
) -> Result<Option<String>> {
    match (api_key, api_key_file) {
        (Some(_), Some(_)) => Err(invalid(
            "--api-key and --api-key-file are mutually exclusive",
        )),
        (Some(key), None) => Ok(Some(validate_api_key(key)?)),
        (None, Some(path)) => {
            let key = std::fs::read_to_string(&path).map_err(|error| {
                Error::new(
                    error.kind(),
                    format!("could not read API key file {}: {error}", path.display()),
                )
            })?;
            Ok(Some(validate_api_key(key)?))
        }
        (None, None) => Ok(None),
    }
}

fn parse_origin(origin: &str) -> Result<HeaderValue> {
    let normalized = origin.trim().trim_end_matches('/');
    if normalized.is_empty() || normalized == "*" || normalized.eq_ignore_ascii_case("null") {
        return Err(invalid(
            "CORS origins must be explicit http:// or https:// origins; wildcard and null are refused",
        ));
    }
    let uri = normalized
        .parse::<axum::http::Uri>()
        .map_err(|_| invalid(format!("invalid CORS origin {origin:?}")))?;
    if !matches!(uri.scheme_str(), Some("http" | "https"))
        || uri.authority().is_none()
        || (uri.path() != "" && uri.path() != "/")
        || uri.query().is_some()
    {
        return Err(invalid(format!(
            "CORS origin {origin:?} must contain only scheme and authority"
        )));
    }
    HeaderValue::from_str(normalized)
        .map_err(|_| invalid(format!("invalid CORS origin header {origin:?}")))
}

fn validate_positive<T>(label: &str, value: T) -> Result<()>
where
    T: PartialEq + From<u8>,
{
    if value == T::from(0) {
        Err(invalid(format!("{label} must be greater than zero")))
    } else {
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{header::ORIGIN, Request},
    };
    use tower::ServiceExt;

    #[test]
    fn remote_listener_requires_auth_or_explicit_override() {
        let addr = SocketAddr::from(([0, 0, 0, 0], 8181));
        let error = ServerPolicy::resolve(addr, ServeOptions::default()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);

        // Cleartext is acknowledged throughout so this stays a test of the
        // authentication question alone; the encryption one is tested below.
        let authenticated = ServerPolicy::resolve(
            addr,
            ServeOptions {
                api_key: Some("secret".to_string()),
                allow_cleartext_remote: true,
                ..ServeOptions::default()
            },
        )
        .unwrap();
        assert!(authenticated.auth.enabled());

        let overridden = ServerPolicy::resolve(
            addr,
            ServeOptions {
                allow_unauthenticated_remote: true,
                allow_cleartext_remote: true,
                ..ServeOptions::default()
            },
        )
        .unwrap();
        assert!(overridden.remote_unauthenticated_override);
    }

    /// A key sent over cleartext protects nothing: it is on every request, so
    /// the bind gives away the credential that was supposed to guard it.
    #[test]
    fn a_routable_cleartext_listener_is_refused_even_with_a_key() {
        let addr = SocketAddr::from(([0, 0, 0, 0], 8181));

        let error = ServerPolicy::resolve(
            addr,
            ServeOptions {
                api_key: Some("secret".to_string()),
                ..ServeOptions::default()
            },
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        let message = error.to_string();
        assert!(
            message.contains("--allow-cleartext-remote"),
            "the refusal has to name the way out: {message}"
        );
        assert!(
            message.contains("--tls-cert"),
            "and the way to do it properly: {message}"
        );
    }

    #[test]
    fn an_anonymous_override_is_not_told_it_has_a_key() {
        let error = ServerPolicy::resolve(
            SocketAddr::from(([0, 0, 0, 0], 8181)),
            ServeOptions {
                allow_unauthenticated_remote: true,
                ..ServeOptions::default()
            },
        )
        .unwrap_err();

        let message = error.to_string();
        assert!(message.contains("any credentials"), "{message}");
        assert!(!message.contains("the API key"), "{message}");
        assert!(message.contains("--allow-cleartext-remote"), "{message}");
    }

    #[test]
    fn loopback_is_unaffected_and_an_acknowledgement_or_tls_lets_a_routable_bind_through() {
        let loopback = SocketAddr::from(([127, 0, 0, 1], 8181));
        ServerPolicy::resolve(loopback, ServeOptions::default())
            .expect("loopback never leaves the machine, so neither question applies");

        let addr = SocketAddr::from(([0, 0, 0, 0], 8181));
        ServerPolicy::resolve(
            addr,
            ServeOptions {
                api_key: Some("secret".to_string()),
                allow_cleartext_remote: true,
                ..ServeOptions::default()
            },
        )
        .expect("an operator who says so explicitly is still allowed to");

        let dir = tempfile::tempdir().unwrap();
        // Extensionless on purpose: the public-scrub guard bars a PEM suffix
        // anywhere in tracked source. Same names as the neighbouring TLS test.
        let cert = dir.path().join("certificate-chain");
        let key = dir.path().join("private-key");
        std::fs::write(&cert, "certificate-chain").unwrap();
        std::fs::write(&key, "private-key").unwrap();
        let tls = ServerPolicy::resolve(
            addr,
            ServeOptions {
                api_key: Some("secret".to_string()),
                tls_cert: Some(cert),
                tls_key: Some(key),
                ..ServeOptions::default()
            },
        )
        .expect("TLS answers the question the acknowledgement only waives");
        assert!(tls.tls.is_some());
    }

    #[test]
    fn the_cleartext_acknowledgement_is_not_a_secret_but_is_still_reported() {
        let options = ServeOptions {
            allow_cleartext_remote: true,
            ..ServeOptions::default()
        };
        assert!(format!("{options:?}").contains("allow_cleartext_remote: true"));
    }

    #[test]
    fn key_file_is_trimmed_and_never_exposed_by_debug() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("camelid.key");
        std::fs::write(&path, "file-secret\r\n").unwrap();
        let policy = ServerPolicy::resolve(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            ServeOptions {
                api_key_file: Some(path),
                ..ServeOptions::default()
            },
        )
        .unwrap();
        assert!(policy.auth.enabled());
        assert!(!format!("{policy:?}").contains("file-secret"));
        assert!(policy
            .auth
            .bearer_header_line()
            .unwrap()
            .contains("file-secret"));
    }

    #[test]
    fn cors_refuses_wildcard_null_paths_and_queries() {
        let addr = SocketAddr::from(([127, 0, 0, 1], 8181));
        for origin in [
            "*",
            "null",
            "https://example.test/path",
            "https://example.test/?query=1",
        ] {
            let error = ServerPolicy::resolve(
                addr,
                ServeOptions {
                    cors_origins: vec![origin.to_string()],
                    ..ServeOptions::default()
                },
            )
            .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
        }
        ServerPolicy::resolve(
            addr,
            ServeOptions {
                cors_origins: vec![
                    "https://example.test".to_string(),
                    "http://127.0.0.1:4173/".to_string(),
                ],
                ..ServeOptions::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn tls_files_and_resource_limits_fail_closed_when_incomplete() {
        let addr = SocketAddr::from(([127, 0, 0, 1], 8181));
        let tls_error = ServerPolicy::resolve(
            addr,
            ServeOptions {
                tls_cert: Some(PathBuf::from("certificate-chain")),
                ..ServeOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(tls_error.kind(), ErrorKind::InvalidInput);

        let limit_error = ServerPolicy::resolve(
            addr,
            ServeOptions {
                max_generation_tokens: 0,
                ..ServeOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(limit_error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn bearer_and_x_api_key_are_accepted_with_constant_time_compare() {
        let policy = ServerPolicy::resolve(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            ServeOptions {
                api_key: Some("test-key".to_string()),
                ..ServeOptions::default()
            },
        )
        .unwrap();
        let mut bearer = axum::http::HeaderMap::new();
        bearer.insert(AUTHORIZATION, HeaderValue::from_static("Bearer test-key"));
        assert!(policy.auth.accepts(&bearer));
        bearer.insert(AUTHORIZATION, HeaderValue::from_static("Bearer wrong"));
        assert!(!policy.auth.accepts(&bearer));

        let mut api_key = axum::http::HeaderMap::new();
        api_key.insert(&X_API_KEY, HeaderValue::from_static("test-key"));
        assert!(policy.auth.accepts(&api_key));
    }

    #[tokio::test]
    async fn router_authenticates_api_routes_but_keeps_health_public() {
        let policy = ServerPolicy::resolve(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            ServeOptions {
                api_key: Some("test-key".to_string()),
                ..ServeOptions::default()
            },
        )
        .unwrap();
        let state = super::super::AppState::default().with_server_policy(&policy);
        let app = super::super::router_with_state_and_policy(state, policy);

        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            unauthorized.headers()[WWW_AUTHENTICATE],
            "Bearer realm=\"camelid\""
        );

        let authorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header(&X_API_KEY, "test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);

        let health = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
    }

    #[test]
    fn lan_chat_surface_requires_a_key_even_on_loopback() {
        let error = ServerPolicy::resolve(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            ServeOptions {
                api_surface: ApiSurface::LanChatOnly,
                ..ServeOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        assert!(error.to_string().contains("--api-key"));
    }

    #[test]
    fn lan_chat_surface_obeys_the_remote_cleartext_guard() {
        let addr = SocketAddr::from(([0, 0, 0, 0], 8181));
        let options = ServeOptions {
            api_key: Some("test-key".to_string()),
            api_surface: ApiSurface::LanChatOnly,
            ..ServeOptions::default()
        };

        let error = ServerPolicy::resolve(addr, options.clone()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        assert!(error.to_string().contains("--allow-cleartext-remote"));

        ServerPolicy::resolve(
            addr,
            ServeOptions {
                allow_cleartext_remote: true,
                ..options
            },
        )
        .expect("LAN Chat has the same explicit cleartext escape hatch as the full API");
    }

    #[test]
    fn lan_chat_surface_is_an_exact_method_and_path_allowlist() {
        let surface = ApiSurface::LanChatOnly;
        for (method, path) in [
            (Method::GET, "/v1/models"),
            (Method::GET, "/api/capabilities"),
            (Method::GET, "/api/models/current"),
            (Method::GET, "/api/models/local"),
            (Method::GET, "/api/models/catalog/downloads"),
            (Method::POST, "/api/models/load"),
            (Method::POST, "/api/web/research"),
            (Method::POST, "/v1/chat/completions"),
        ] {
            assert!(surface.allows(&method, path), "refused {method} {path}");
        }
        for (method, path) in [
            (Method::POST, "/v1/models"),
            (Method::GET, "/v1/models/loaded-model"),
            (Method::POST, "/api/models/unload"),
            (Method::POST, "/api/models/local/delete"),
            (Method::POST, "/api/models/catalog/install"),
            (Method::POST, "/api/runtime/gpu"),
            (Method::GET, "/api/runtime/memory"),
            (Method::GET, "/api/telemetry/stream"),
            (Method::POST, "/api/agent/workspace/sessions"),
            (Method::POST, "/v1/responses"),
            (Method::POST, "/v1/conversations"),
            (Method::GET, "/metrics"),
            (Method::GET, "/api/not-yet-invented"),
        ] {
            assert!(!surface.allows(&method, path), "allowed {method} {path}");
        }
    }

    #[tokio::test]
    async fn lan_chat_surface_authenticates_before_it_enforces_the_allowlist() {
        let models = tempfile::tempdir().unwrap();
        std::fs::write(models.path().join("local.gguf"), b"not a real gguf").unwrap();
        let policy = ServerPolicy::resolve(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            ServeOptions {
                api_key: Some("test-key".to_string()),
                api_surface: ApiSurface::LanChatOnly,
                ..ServeOptions::default()
            },
        )
        .unwrap();
        let state = super::super::AppState::default()
            .with_models_dir(Some(models.path().to_path_buf()))
            .with_server_policy(&policy);
        let app = super::super::router_with_state_and_policy(state, policy);

        let unauthenticated = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/catalog/install")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);

        let forbidden = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/catalog/install")
                    .header(&X_API_KEY, "test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forbidden.status(), StatusCode::FORBIDDEN);
        let body = axum::body::to_bytes(forbidden.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "lan_chat_only");

        let allowed = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header(&X_API_KEY, "test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(allowed.status(), StatusCode::OK);

        // Web research is part of the same authenticated LAN Chat surface.
        // An irrelevant prompt deterministically skips and therefore proves
        // route/auth integration without making a network request.
        let research = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/web/research")
                    .header(&X_API_KEY, "test-key")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"prompt":"Rewrite this sentence."}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(research.status(), StatusCode::OK);
        let body = axum::body::to_bytes(research.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["status"], "skipped");
        assert_eq!(body["triggered"], false);

        let arbitrary_model = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/load")
                    .header(&X_API_KEY, "test-key")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"path":"outside.gguf","replace":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(arbitrary_model.status(), StatusCode::FORBIDDEN);
        let body = axum::body::to_bytes(arbitrary_model.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "lan_model_not_local");

        let local_model = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/models/load")
                    .header(&X_API_KEY, "test-key")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"path":"ignored","filename":"local.gguf","replace":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(local_model.status(), StatusCode::BAD_REQUEST);

        let chat = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header(&X_API_KEY, "test-key")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(chat.status(), StatusCode::FORBIDDEN);

        let health = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        let body = axum::body::to_bytes(health.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["api_surface"], "lan_chat_only");
        // Health answers before authentication, and this listener is read by a
        // phone even while it is bound to loopback behind a tunnel.
        assert!(body.get("executable").is_none(), "{body}");
        assert!(body.get("listen_addr").is_none(), "{body}");
    }

    #[tokio::test]
    async fn cors_emits_only_explicitly_allowed_origin() {
        let policy = ServerPolicy::resolve(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            ServeOptions {
                cors_origins: vec!["https://allowed.example".to_string()],
                ..ServeOptions::default()
            },
        )
        .unwrap();
        let state = super::super::AppState::default().with_server_policy(&policy);
        let app = super::super::router_with_state_and_policy(state, policy);

        let allowed = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header(ORIGIN, "https://allowed.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            allowed.headers()["access-control-allow-origin"],
            "https://allowed.example"
        );

        let denied = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header(ORIGIN, "https://denied.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(!denied.headers().contains_key("access-control-allow-origin"));
    }

    #[tokio::test]
    async fn request_body_ceiling_returns_payload_too_large() {
        let policy = ServerPolicy::resolve(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            ServeOptions {
                max_request_body_bytes: 8,
                ..ServeOptions::default()
            },
        )
        .unwrap();
        let state = super::super::AppState::default().with_server_policy(&policy);
        let app = super::super::router_with_state_and_policy(state, policy);
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"prompt":"too large"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
