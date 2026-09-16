//! A separately authenticated Chat listener sharing the local engine's models.
use super::{server, ApiSurface, AppState};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::net::{SocketAddr, TcpListener, UdpSocket};

pub(super) struct Listener {
    handle: axum_server::Handle<SocketAddr>,
    task: tokio::task::JoinHandle<()>,
    port: u16,
    key: String,
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.handle.shutdown();
    }
}

#[derive(Serialize)]
pub(super) struct Status {
    enabled: bool,
    port: Option<u16>,
    url: Option<String>,
    key: Option<String>,
}

fn snapshot(listener: Option<&Listener>) -> Status {
    let listener = listener.filter(|listener| !listener.task.is_finished());
    Status {
        enabled: listener.is_some(),
        port: listener.map(|listener| listener.port),
        url: listener.and_then(|listener| {
            // UDP connect only asks the OS for a route; it sends no traffic.
            let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
            socket.connect("192.0.2.1:9").ok()?;
            let ip = socket.local_addr().ok()?.ip();
            Some(format!("http://{ip}:{}", listener.port))
        }),
        key: listener.map(|listener| listener.key.clone()),
    }
}

fn allowed(state: &AppState, headers: &HeaderMap) -> bool {
    state.serve_addr.ip().is_loopback()
        && state.api_surface == ApiSurface::Full
        && super::workspace::local_management_request_allowed(headers)
}

pub(super) async fn status(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !allowed(&state, &headers) {
        return StatusCode::FORBIDDEN.into_response();
    }
    let guard = state.lan_sharing.lock().await;
    let mut response = Json(snapshot(guard.as_ref())).into_response();
    response
        .headers_mut()
        .insert("cache-control", "no-store".parse().unwrap());
    response
}

#[derive(Deserialize)]
pub(super) struct Change {
    enabled: bool,
}

pub(super) async fn set_enabled(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(change): Json<Change>,
) -> Response {
    if !allowed(&state, &headers) {
        return StatusCode::FORBIDDEN.into_response();
    }
    let mut guard = state.lan_sharing.lock().await;
    if !change.enabled {
        if let Some(mut listener) = guard.take() {
            listener.handle.shutdown();
            // Wait until the listening socket and accepted connections close.
            let _ = (&mut listener.task).await;
        }
    } else if guard
        .as_ref()
        .is_none_or(|listener| listener.task.is_finished())
    {
        match start(state.clone()) {
            Ok(listener) => *guard = Some(listener),
            Err(error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": error.to_string()})),
                )
                    .into_response()
            }
        }
    }
    let mut response = Json(snapshot(guard.as_ref())).into_response();
    response
        .headers_mut()
        .insert("cache-control", "no-store".parse().unwrap());
    response
}

fn start(state: AppState) -> std::io::Result<Listener> {
    let credential = crate::lan_key::provision(false)?;
    start_with_key(state, credential.secret().to_owned())
}

fn start_with_key(state: AppState, key: String) -> std::io::Result<Listener> {
    let socket = TcpListener::bind("0.0.0.0:0")?;
    socket.set_nonblocking(true)?;
    let addr: SocketAddr = socket.local_addr()?;
    let policy = server::ServerPolicy::resolve(
        addr,
        super::ServeOptions {
            api_key: Some(key.clone()),
            api_surface: ApiSurface::LanChatOnly,
            allow_cleartext_remote: true,
            max_request_body_bytes: state.server_limits.max_request_body_bytes,
            max_prompt_tokens: state.server_limits.max_prompt_tokens,
            max_generation_tokens: state.server_limits.max_generation_tokens,
            max_download_bytes: state.server_limits.max_download_bytes,
            ..Default::default()
        },
    )?;
    let mut remote_state = state
        .with_serve_addr(addr)
        .with_local_model_delete(false)
        .with_workspace_cli_token(None)
        .with_server_policy(&policy);
    // The remote router cannot manage sharing. Give it no reference back to
    // its owner so dropping the local router also closes this listener.
    remote_state.lan_sharing = Default::default();
    let app = super::router_with_state_and_policy(remote_state, policy);
    let handle = axum_server::Handle::new();
    let server = axum_server::from_tcp(socket)?.handle(handle.clone());
    let task = tokio::spawn(async move {
        if let Err(error) = server.serve(app.into_make_service()).await {
            tracing::error!(%error, "LAN sharing listener stopped");
        }
    });
    Ok(Listener {
        handle,
        task,
        port: addr.port(),
        key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    async fn request(port: u16, path: &str, key: Option<&str>) -> String {
        let path = path.to_owned();
        let auth = key
            .map(|key| format!("Authorization: Bearer {key}\r\n"))
            .unwrap_or_default();
        tokio::task::spawn_blocking(move || {
            use std::io::{Read, Write};
            let mut stream =
                std::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port)).unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            write!(
                stream,
                "GET {path} HTTP/1.1\r\nHost: localhost:{port}\r\n{auth}Connection: close\r\n\r\n"
            )
            .unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            response
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn sharing_requires_key_restricts_routes_and_closes_socket() {
        let state = AppState::default();
        let listener = start_with_key(state.clone(), "test-sharing-key".into()).unwrap();
        *state.active_model_id.write().await = Some("shared-model".into());
        let port = listener.port;
        *state.lan_sharing.lock().await = Some(listener);
        assert!(request(port, "/", None).await.starts_with("HTTP/1.1 200"));
        let health = request(port, "/v1/health", None).await;
        assert!(health.contains("lan_chat_only"));
        assert!(health.contains("shared-model"));
        assert!(request(port, "/v1/models", None)
            .await
            .starts_with("HTTP/1.1 401"));
        assert!(request(port, "/v1/models", Some("wrong"))
            .await
            .starts_with("HTTP/1.1 401"));
        assert!(request(port, "/v1/models", Some("test-sharing-key"))
            .await
            .starts_with("HTTP/1.1 200"));
        assert!(
            request(port, "/api/runtime/lan-sharing", Some("test-sharing-key"))
                .await
                .starts_with("HTTP/1.1 403")
        );
        let mut headers = HeaderMap::new();
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        headers.insert("origin", "http://127.0.0.1:8181".parse().unwrap());
        let response = set_enabled(
            State(state.clone()),
            headers,
            Json(Change { enabled: false }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(state.lan_sharing.lock().await.is_none());
        assert!(std::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port)).is_err());
    }

    #[tokio::test]
    async fn management_rejects_cross_origin_and_remote_listeners() {
        for (addr, origin, expected) in [
            (
                "127.0.0.1:8181",
                "http://evil.example",
                StatusCode::FORBIDDEN,
            ),
            ("127.0.0.1:8181", "http://127.0.0.1:8181", StatusCode::OK),
            (
                "0.0.0.0:8181",
                "http://127.0.0.1:8181",
                StatusCode::FORBIDDEN,
            ),
        ] {
            let state = AppState::default().with_serve_addr(addr.parse().unwrap());
            let response = super::super::router_with_state(state)
                .oneshot(
                    Request::builder()
                        .uri("/api/runtime/lan-sharing")
                        .header("host", "127.0.0.1:8181")
                        .header("origin", origin)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
            if expected == StatusCode::OK {
                assert_eq!(response.headers()["cache-control"], "no-store");
            }
        }
    }
}
