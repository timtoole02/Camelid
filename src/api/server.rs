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

pub(crate) const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const DEFAULT_MAX_PROMPT_TOKENS: usize = 131_072;
pub(crate) const DEFAULT_MAX_GENERATION_TOKENS: u32 = 8_192;
pub(crate) const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_API_KEY_BYTES: usize = 4 * 1024;

static X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");

/// Public serve-time inputs. The CLI fills these directly; embedded callers
/// can use `Default` and retain Camelid's anonymous loopback behavior.
#[derive(Clone)]
pub struct ServeOptions {
    pub api_key: Option<String>,
    pub api_key_file: Option<PathBuf>,
    pub cors_origins: Vec<String>,
    pub allow_unauthenticated_remote: bool,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    pub max_request_body_bytes: usize,
    pub max_prompt_tokens: usize,
    pub max_generation_tokens: u32,
    pub max_download_bytes: u64,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            api_key: None,
            api_key_file: None,
            cors_origins: Vec::new(),
            allow_unauthenticated_remote: false,
            tls_cert: None,
            tls_key: None,
            max_request_body_bytes: DEFAULT_MAX_REQUEST_BODY_BYTES,
            max_prompt_tokens: DEFAULT_MAX_PROMPT_TOKENS,
            max_generation_tokens: DEFAULT_MAX_GENERATION_TOKENS,
            max_download_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
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
            .field("tls_cert", &self.tls_cert)
            .field("tls_key", &self.tls_key.as_ref().map(|_| "[REDACTED PATH]"))
            .field("max_request_body_bytes", &self.max_request_body_bytes)
            .field("max_prompt_tokens", &self.max_prompt_tokens)
            .field("max_generation_tokens", &self.max_generation_tokens)
            .field("max_download_bytes", &self.max_download_bytes)
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
    pub(crate) fn enabled(&self) -> bool {
        self.key.is_some()
    }

    pub(crate) fn bearer_header_line(&self) -> Option<String> {
        let key = self.key.as_deref()?;
        let key = std::str::from_utf8(key).ok()?;
        Some(format!("Authorization: Bearer {key}\r\n"))
    }

    fn accepts(&self, headers: &axum::http::HeaderMap) -> bool {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServerLimits {
    pub(crate) max_request_body_bytes: usize,
    pub(crate) max_prompt_tokens: usize,
    pub(crate) max_generation_tokens: u32,
    pub(crate) max_download_bytes: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct TlsFiles {
    pub(crate) cert: PathBuf,
    pub(crate) key: PathBuf,
}

#[derive(Clone)]
pub(crate) struct ServerPolicy {
    pub(crate) auth: ApiAuth,
    pub(crate) limits: ServerLimits,
    pub(crate) tls: Option<TlsFiles>,
    cors_origins: Arc<[HeaderValue]>,
    pub(crate) remote_unauthenticated_override: bool,
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
            .finish()
    }
}

impl ServerPolicy {
    pub(crate) fn resolve(addr: SocketAddr, options: ServeOptions) -> Result<Self> {
        validate_positive("max request body bytes", options.max_request_body_bytes)?;
        validate_positive("max prompt tokens", options.max_prompt_tokens)?;
        validate_positive("max generation tokens", options.max_generation_tokens)?;
        validate_positive("max download bytes", options.max_download_bytes)?;

        if options.api_key.is_some() && options.api_key_file.is_some() {
            return Err(invalid(
                "--api-key and --api-key-file are mutually exclusive",
            ));
        }
        let key = match (options.api_key, options.api_key_file) {
            (Some(key), None) => Some(validate_api_key(key)?),
            (None, Some(path)) => {
                let key = std::fs::read_to_string(&path).map_err(|error| {
                    Error::new(
                        error.kind(),
                        format!("could not read API key file {}: {error}", path.display()),
                    )
                })?;
                Some(validate_api_key(key)?)
            }
            (None, None) => None,
            (Some(_), Some(_)) => unreachable!("conflict handled above"),
        };

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

        let tls = match (options.tls_cert, options.tls_key) {
            (Some(cert), Some(key)) => Some(TlsFiles { cert, key }),
            (None, None) => None,
            _ => {
                return Err(invalid(
                    "--tls-cert and --tls-key must be provided together",
                ))
            }
        };
        let cors_origins = options
            .cors_origins
            .iter()
            .map(|origin| parse_origin(origin))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            auth: ApiAuth {
                key: key.map(|key| Arc::<[u8]>::from(key.into_bytes())),
            },
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
}

pub(crate) async fn authenticate(
    State(auth): State<ApiAuth>,
    request: Request,
    next: Next,
) -> Response {
    if request.method() == Method::OPTIONS
        || !route_requires_auth(request.uri().path())
        || auth.accepts(request.headers())
    {
        return next.run(request).await;
    }

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

fn route_requires_auth(path: &str) -> bool {
    !matches!(path, "/health" | "/v1/health" | "/" | "/index.html")
        && !path.starts_with("/assets/")
        && !path.starts_with("/favicon")
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

        let authenticated = ServerPolicy::resolve(
            addr,
            ServeOptions {
                api_key: Some("secret".to_string()),
                ..ServeOptions::default()
            },
        )
        .unwrap();
        assert!(authenticated.auth.enabled());

        let overridden = ServerPolicy::resolve(
            addr,
            ServeOptions {
                allow_unauthenticated_remote: true,
                ..ServeOptions::default()
            },
        )
        .unwrap();
        assert!(overridden.remote_unauthenticated_override);
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
