//! Which browser origins may read a Camelid HTTP surface.
//!
//! Shared by the engine's listener and the fabric proxy for the reason
//! [`crate::tls_pair`] is shared: two front doors that validated one flag
//! differently would give an operator two answers to the same question, and
//! the looser answer is the one that would matter.

use std::io::{Error, ErrorKind, Result};

use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderName, HeaderValue, Method};
use tower_http::cors::{AllowOrigin, CorsLayer};

/// The second header a client key may arrive in, beside `Authorization`.
pub(crate) static X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");

/// Accept only an explicit `http://` or `https://` origin: a scheme and an
/// authority, nothing else.
pub(crate) fn parse_origin(origin: &str) -> Result<HeaderValue> {
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

/// The layer for an allowlist already accepted by [`parse_origin`].
///
/// The allowed request headers are the two a key can arrive in plus the body's
/// type: without `authorization` here, a browser strips the key from every
/// cross-origin request it makes.
pub(crate) fn layer(origins: &[HeaderValue]) -> CorsLayer {
    let layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers([AUTHORIZATION, CONTENT_TYPE, X_API_KEY.clone()]);
    if origins.is_empty() {
        layer
    } else {
        layer.allow_origin(AllowOrigin::list(origins.iter().cloned()))
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}
