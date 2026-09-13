//! Local MCP control plane. Configuration never starts a process on load, and
//! every tool call is frozen before a separate, single-use approval decision.
use super::{api_error, AppState};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use rmcp::{
    model::{CallToolRequestParams, PaginatedRequestParams},
    service::RunningService,
    transport::{
        streamable_http_client::StreamableHttpClientTransportConfig, StreamableHttpClientTransport,
        TokioChildProcess,
    },
    RoleClient, ServiceExt,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

const MAX_CONNECTIONS: usize = 16;
const MAX_TOOLS: usize = 128;
const MAX_RESULT_BYTES: usize = 64 * 1024;
const CALL_TIMEOUT: Duration = Duration::from_secs(60);
const APPROVAL_TTL: Duration = Duration::from_secs(300);

type Client = RunningService<RoleClient, ()>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ConnectionConfig {
    #[serde(default)]
    pub id: String,
    pub name: String,
    pub transport: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub url: String,
    /// Names only. Secret values are read in the engine, never returned to UI.
    #[serde(default)]
    pub env_vars: Vec<String>,
    #[serde(default)]
    pub bearer_env: String,
}
impl ConnectionConfig {
    fn validate(&mut self) -> Result<(), &'static str> {
        self.name = self.name.trim().to_string();
        if self.name.is_empty() || self.name.len() > 80 {
            return Err("Give the connection a name of at most 80 characters.");
        }
        if self.id.is_empty() {
            self.id = uuid::Uuid::new_v4().simple().to_string();
        }
        if self.id.len() != 32 || !self.id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("Invalid connection ID.");
        }
        if self.args.len() > 64 || self.args.iter().any(|s| s.len() > 4096) {
            return Err("Too many or oversized command arguments.");
        }
        if self.env_vars.len() > 32
            || self
                .env_vars
                .iter()
                .chain(std::iter::once(&self.bearer_env))
                .any(|s| {
                    !s.is_empty()
                        && (s.len() > 128
                            || !s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
                })
        {
            return Err("Environment variable names must use letters, numbers, or underscores.");
        }
        match self.transport.as_str() {
            "stdio" if !self.command.trim().is_empty() && self.command.len() <= 4096 => {
                self.url.clear();
                self.bearer_env.clear();
            }
            "http" => {
                let url =
                    url::Url::parse(&self.url).map_err(|_| "Enter a valid MCP endpoint URL.")?;
                let local = url.host_str().is_some_and(|h| {
                    h == "localhost"
                        || h.trim_matches(['[', ']'])
                            .parse::<std::net::IpAddr>()
                            .is_ok_and(|ip| ip.is_loopback())
                });
                if !(url.scheme() == "https" || (url.scheme() == "http" && local))
                    || url.host_str().is_none()
                    || !url.username().is_empty()
                    || url.password().is_some()
                    || url.fragment().is_some()
                    || url.query().is_some()
                {
                    return Err("Use HTTPS (or HTTP on localhost), without credentials, query parameters, or fragments. Use a token environment variable for authentication.");
                }
                self.command.clear();
                self.args.clear();
                self.env_vars.clear();
            }
            _ => return Err("Choose a local command or a Streamable HTTP endpoint."),
        }
        Ok(())
    }
}

#[derive(Clone, Serialize)]
struct ToolView {
    key: String,
    name: String,
    description: String,
    input_schema: Value,
}
struct Connection {
    config: ConnectionConfig,
    client: Option<Arc<Client>>,
    tools: Vec<ToolView>,
    error: Option<String>,
}
#[derive(Clone, Serialize)]
struct CallView {
    id: String,
    connection_id: String,
    connection_name: String,
    tool: String,
    arguments: Value,
    status: String,
    result: Option<Value>,
}
struct Call {
    view: CallView,
    client: Arc<Client>,
    created: Instant,
    cancel: CancellationToken,
}
#[derive(Default)]
struct Registry {
    loaded: bool,
    connections: BTreeMap<String, Connection>,
    calls: HashMap<String, Call>,
}
#[derive(Clone)]
pub(super) struct McpManager {
    inner: Arc<Mutex<Registry>>,
    path: Arc<PathBuf>,
}
impl Default for McpManager {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Registry::default())),
            path: Arc::new(
                crate::chat::workspace_memory::default_store_path()
                    .with_file_name("mcp-connections.json"),
            ),
        }
    }
}
impl McpManager {
    async fn load(&self) -> Result<(), String> {
        let mut registry = self.inner.lock().await;
        if registry.loaded {
            return Ok(());
        }
        let path = self.path.clone();
        let configs: Vec<ConnectionConfig> =
            tokio::task::spawn_blocking(move || -> Result<_, String> {
                match std::fs::read(&*path) {
                    Ok(bytes) if bytes.len() <= 512 * 1024 => serde_json::from_slice(&bytes)
                        .map_err(|_| "Saved MCP configuration is invalid.".into()),
                    Ok(_) => Err("Saved MCP configuration is too large.".into()),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
                    Err(_) => Err("Could not read MCP configuration.".into()),
                }
            })
            .await
            .map_err(|_| "MCP storage task failed.")??;
        if configs.len() > MAX_CONNECTIONS {
            return Err("Too many saved MCP connections.".into());
        }
        for mut config in configs {
            config.validate()?;
            registry.connections.insert(
                config.id.clone(),
                Connection {
                    config,
                    client: None,
                    tools: vec![],
                    error: None,
                },
            );
        }
        registry.loaded = true;
        Ok(())
    }
    async fn persist(&self, configs: Vec<ConnectionConfig>) -> Result<(), String> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || -> Result<(), String> {
            let parent = path
                .parent()
                .ok_or("Invalid MCP configuration directory.")?;
            std::fs::create_dir_all(parent)
                .map_err(|_| "Could not create MCP configuration directory.")?;
            let mut file = tempfile::NamedTempFile::new_in(parent)
                .map_err(|_| "Could not save MCP configuration.")?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                file.as_file()
                    .set_permissions(std::fs::Permissions::from_mode(0o600))
                    .map_err(|_| "Could not protect MCP configuration.")?;
            }
            std::io::Write::write_all(
                &mut file,
                &serde_json::to_vec_pretty(&configs)
                    .map_err(|_| "Could not encode MCP configuration.")?,
            )
            .map_err(|_| "Could not write MCP configuration.")?;
            file.as_file()
                .sync_all()
                .map_err(|_| "Could not flush MCP configuration.")?;
            file.persist(&*path)
                .map_err(|_| "Could not replace MCP configuration.")?;
            Ok(())
        })
        .await
        .map_err(|_| "MCP storage task failed.")?
    }
}
fn error(code: StatusCode, message: impl Into<String>) -> Response {
    api_error(code, "mcp_error", message.into(), None)
}
async fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    if !state.serve_addr.ip().is_loopback()
        || !super::workspace::local_management_request_allowed(headers)
        || headers.get("x-camelid-mcp").and_then(|v| v.to_str().ok()) != Some("1")
    {
        return Err(error(
            StatusCode::FORBIDDEN,
            "MCP connections require Camelid's local, same-origin interface.",
        ));
    }
    state
        .mcp
        .load()
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))
}
fn connection_view(c: &Connection) -> Value {
    json!({"config": c.config, "connected": c.client.as_ref().is_some_and(|s| !s.is_closed() && !s.peer().is_transport_closed()), "tools": c.tools, "error": c.error})
}
pub(super) async fn list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(e) = authorize(&state, &headers).await {
        return e;
    }
    let r = state.mcp.inner.lock().await;
    Json(json!({"connections": r.connections.values().map(connection_view).collect::<Vec<_>>()}))
        .into_response()
}
pub(super) async fn save(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut config): Json<ConnectionConfig>,
) -> Response {
    if let Err(e) = authorize(&state, &headers).await {
        return e;
    }
    if let Err(e) = config.validate() {
        return error(StatusCode::BAD_REQUEST, e);
    }
    let mut r = state.mcp.inner.lock().await;
    if r.connections.contains_key(&config.id) {
        return error(
            StatusCode::CONFLICT,
            "Remove the old connection before replacing its configuration.",
        );
    }
    if r.connections.len() >= MAX_CONNECTIONS {
        return error(
            StatusCode::BAD_REQUEST,
            "At most 16 MCP connections can be saved.",
        );
    }
    let mut configs: Vec<_> = r.connections.values().map(|c| c.config.clone()).collect();
    configs.push(config.clone());
    if let Err(e) = state.mcp.persist(configs).await {
        return error(StatusCode::INTERNAL_SERVER_ERROR, e);
    }
    let c = Connection {
        config,
        client: None,
        tools: vec![],
        error: None,
    };
    let view = connection_view(&c);
    r.connections.insert(c.config.id.clone(), c);
    (StatusCode::CREATED, Json(view)).into_response()
}
fn stop_connection(r: &mut Registry, id: &str) {
    if let Some(c) = r.connections.get_mut(id) {
        if let Some(client) = c.client.take() {
            client.cancellation_token().cancel();
        }
        c.tools.clear();
    }
    for call in r.calls.values_mut().filter(|c| {
        c.view.connection_id == id && matches!(c.view.status.as_str(), "pending" | "running")
    }) {
        call.cancel.cancel();
        call.view.status = "cancelled".into();
    }
}
pub(super) async fn disconnect(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authorize(&state, &headers).await {
        return e;
    }
    let mut r = state.mcp.inner.lock().await;
    stop_connection(&mut r, &id);
    Json(json!({"disconnected": true})).into_response()
}
pub(super) async fn remove(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authorize(&state, &headers).await {
        return e;
    }
    let mut r = state.mcp.inner.lock().await;
    let configs = r
        .connections
        .values()
        .filter(|c| c.config.id != id)
        .map(|c| c.config.clone())
        .collect();
    if let Err(e) = state.mcp.persist(configs).await {
        return error(StatusCode::INTERNAL_SERVER_ERROR, e);
    }
    stop_connection(&mut r, &id);
    r.connections.remove(&id);
    Json(json!({"removed": true})).into_response()
}
async fn open(config: &ConnectionConfig) -> Result<(Client, Vec<ToolView>), String> {
    let client = match config.transport.as_str() {
        "stdio" => {
            let mut cmd = tokio::process::Command::new(&config.command);
            cmd.args(&config.args).env_clear().kill_on_drop(true);
            for key in [
                "PATH",
                "HOME",
                "USERPROFILE",
                "SYSTEMROOT",
                "WINDIR",
                "TEMP",
                "TMP",
                "TMPDIR",
                "APPDATA",
                "LOCALAPPDATA",
            ]
            .into_iter()
            .chain(config.env_vars.iter().map(String::as_str))
            {
                if let Some(value) = std::env::var_os(key) {
                    cmd.env(key, value);
                }
            }
            let mut wrapped = process_wrap::tokio::CommandWrap::from(cmd);
            #[cfg(unix)]
            wrapped.wrap(process_wrap::tokio::ProcessGroup::leader());
            #[cfg(windows)]
            wrapped.wrap(process_wrap::tokio::JobObject);
            let (transport, _) = TokioChildProcess::builder(wrapped)
                .stderr(Stdio::null())
                .spawn()
                .map_err(|_| {
                    "Could not start the MCP command. Check its executable and arguments."
                })?;
            ().serve(transport)
                .await
                .map_err(|_| "The MCP server did not complete initialization.")?
        }
        "http" => {
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
            let http = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
                .connect_timeout(Duration::from_secs(10))
                .timeout(CALL_TIMEOUT)
                .build()
                .map_err(|_| "Could not create MCP HTTP client.")?;
            let mut options = StreamableHttpClientTransportConfig::with_uri(config.url.clone());
            options.max_sse_event_size = 1024 * 1024;
            options.max_concurrent_requests = 1;
            options.reinit_on_expired_session = false;
            if !config.bearer_env.is_empty() {
                options.auth_header = Some(std::env::var(&config.bearer_env).map_err(|_| {
                    "The token environment variable is not set in the Camelid engine."
                })?);
            }
            ().serve(StreamableHttpClientTransport::with_client(http, options))
                .await
                .map_err(|_| {
                    "Could not initialize the MCP endpoint. Check its URL and authentication."
                })?
        }
        _ => return Err("Unsupported MCP transport.".into()),
    };
    let mut tools = Vec::new();
    let mut cursor = None;
    let mut seen = std::collections::HashSet::new();
    let mut pages = 0;
    loop {
        pages += 1;
        if pages > 8 {
            return Err("MCP tool pagination exceeded 8 pages.".into());
        }
        let page = client
            .list_tools(
                cursor
                    .take()
                    .map(|cursor| PaginatedRequestParams::default().with_cursor(Some(cursor))),
            )
            .await
            .map_err(|_| "The MCP server could not list tools.")?;
        for tool in page.tools {
            if tools.len() >= MAX_TOOLS || !seen.insert(tool.name.to_string()) {
                return Err("MCP tool list exceeds 128 tools or contains duplicate names.".into());
            }
            let schema =
                serde_json::to_value(&tool.input_schema).map_err(|_| "Invalid MCP tool schema.")?;
            if schema.to_string().len() > 32 * 1024
                || schema.get("type").and_then(Value::as_str) != Some("object")
            {
                return Err(
                    "MCP tools must provide an object input schema of at most 32 KB.".into(),
                );
            }
            let digest = format!(
                "{:x}",
                Sha256::digest(format!("{}:{}", config.id, tool.name))
            );
            tools.push(ToolView {
                key: format!("mcp_{}", &digest[..40]),
                name: tool.name.to_string(),
                description: tool
                    .description
                    .unwrap_or_default()
                    .chars()
                    .take(4096)
                    .collect(),
                input_schema: schema,
            });
        }
        cursor = page.next_cursor;
        if cursor.is_none() {
            break;
        }
        if tools.len() >= MAX_TOOLS {
            return Err("MCP tool pagination exceeded the limit.".into());
        }
    }
    Ok((client, tools))
}
pub(super) async fn connect(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authorize(&state, &headers).await {
        return e;
    }
    // Serializes configuration changes and connect; calls never hold this lock
    // across execution, so Stop remains available during a running tool.
    let mut r = state.mcp.inner.lock().await;
    let Some(c) = r.connections.get_mut(&id) else {
        return error(StatusCode::NOT_FOUND, "Connection not found.");
    };
    if c.client
        .as_ref()
        .is_some_and(|s| !s.is_closed() && !s.peer().is_transport_closed())
    {
        return Json(connection_view(c)).into_response();
    }
    match tokio::time::timeout(Duration::from_secs(20), open(&c.config)).await {
        Ok(Ok((client, tools))) => {
            c.client = Some(Arc::new(client));
            c.tools = tools;
            c.error = None;
        }
        Ok(Err(message)) => {
            c.client = None;
            c.tools.clear();
            c.error = Some(message);
        }
        Err(_) => {
            c.client = None;
            c.tools.clear();
            c.error = Some("MCP initialization timed out.".into());
        }
    }
    Json(connection_view(c)).into_response()
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PrepareCall {
    tool_key: String,
    arguments: Value,
}
pub(super) async fn prepare_call(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<PrepareCall>,
) -> Response {
    if let Err(e) = authorize(&state, &headers).await {
        return e;
    }
    if !request.arguments.is_object() || request.arguments.to_string().len() > MAX_RESULT_BYTES {
        return error(
            StatusCode::BAD_REQUEST,
            "Tool arguments must be a JSON object of at most 64 KB.",
        );
    }
    let mut r = state.mcp.inner.lock().await;
    r.calls
        .retain(|_, c| c.created.elapsed() < APPROVAL_TTL || c.view.status == "running");
    if r.calls.len() >= 128 {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "Too many pending or recent tool calls.",
        );
    }
    let found = r.connections.values().find_map(|c| {
        c.tools
            .iter()
            .find(|t| t.key == request.tool_key)
            .map(|t| (c, t))
    });
    let Some((c, tool)) = found else {
        return error(
            StatusCode::BAD_REQUEST,
            "This tool is no longer available. Reconnect its server.",
        );
    };
    let Some(client) = c
        .client
        .as_ref()
        .filter(|s| !s.is_closed() && !s.peer().is_transport_closed())
        .cloned()
    else {
        return error(StatusCode::CONFLICT, "This MCP server is disconnected.");
    };
    let view = CallView {
        id: uuid::Uuid::new_v4().to_string(),
        connection_id: c.config.id.clone(),
        connection_name: c.config.name.clone(),
        tool: tool.name.clone(),
        arguments: request.arguments,
        status: "pending".into(),
        result: None,
    };
    r.calls.insert(
        view.id.clone(),
        Call {
            view: view.clone(),
            client,
            created: Instant::now(),
            cancel: CancellationToken::new(),
        },
    );
    (StatusCode::CREATED, Json(view)).into_response()
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Decision {
    approved: bool,
}
fn bounded_result(value: Value) -> Value {
    let encoded = value.to_string();
    if encoded.len() <= MAX_RESULT_BYTES {
        return value;
    }
    let mut end = MAX_RESULT_BYTES / 2;
    while !encoded.is_char_boundary(end) {
        end -= 1;
    }
    json!({"isError": true, "content": [{"type": "text", "text": format!("Tool output exceeded 64 KB and was truncated. Narrow the request.\n{}", &encoded[..end])}], "truncated": true})
}
pub(super) async fn decide_call(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(decision): Json<Decision>,
) -> Response {
    if let Err(e) = authorize(&state, &headers).await {
        return e;
    }
    let mut r = state.mcp.inner.lock().await;
    let Some(call) = r.calls.get_mut(&id) else {
        return error(StatusCode::NOT_FOUND, "Tool call not found or expired.");
    };
    if call.view.status != "pending" {
        return error(
            StatusCode::CONFLICT,
            "This tool call has already been decided. It will not run again.",
        );
    }
    if call.created.elapsed() >= APPROVAL_TTL {
        call.view.status = "expired".into();
        return error(
            StatusCode::CONFLICT,
            "Approval expired. Send the request again.",
        );
    }
    if !decision.approved {
        call.view.status = "denied".into();
        return Json(call.view.clone()).into_response();
    }
    call.view.status = "running".into();
    let view = call.view.clone();
    let client = call.client.clone();
    let cancel = call.cancel.clone();
    drop(r);
    // Spawn exactly once, then return a pollable receipt. A browser retry or
    // lost response cannot execute a side effect twice.
    let manager = state.mcp.clone();
    let response_view = view.clone();
    tokio::spawn(async move {
        let params = CallToolRequestParams::new(view.tool.clone())
            .with_arguments(view.arguments.as_object().unwrap().clone());
        let (status, result) = tokio::select! {
            _ = cancel.cancelled() => { client.cancellation_token().cancel(); ("cancelled", None) }
            outcome = tokio::time::timeout(CALL_TIMEOUT, client.call_tool(params)) => match outcome {
                Ok(Ok(result)) => ("complete", Some(bounded_result(serde_json::to_value(result).unwrap_or(Value::Null)))),
                Ok(Err(_)) => ("error", Some(json!({"isError": true, "content": [{"type":"text", "text":"The MCP server reported a protocol or transport error. The action may have started; inspect its outcome before retrying."}]}))),
                Err(_) => { client.cancellation_token().cancel(); ("timed_out", Some(json!({"isError": true, "content": [{"type":"text", "text":"Tool timed out after 60 seconds. Its connection was closed. Remote actions may still have completed; inspect their outcome before retrying."}]}))) }
            }
        };
        let mut r = manager.inner.lock().await;
        if let Some(call) = r.calls.get_mut(&view.id) {
            if call.view.status == "running" {
                call.view.status = status.into();
                call.view.result = result;
            }
        }
    });
    (StatusCode::ACCEPTED, Json(response_view)).into_response()
}
pub(super) async fn call_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authorize(&state, &headers).await {
        return e;
    }
    let r = state.mcp.inner.lock().await;
    match r.calls.get(&id) {
        Some(c) => Json(c.view.clone()).into_response(),
        None => error(StatusCode::NOT_FOUND, "Tool call not found."),
    }
}
pub(super) async fn cancel_call(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = authorize(&state, &headers).await {
        return e;
    }
    let mut r = state.mcp.inner.lock().await;
    if let Some(c) = r.calls.get_mut(&id) {
        if matches!(c.view.status.as_str(), "pending" | "running") {
            if c.view.status == "running" {
                c.client.cancellation_token().cancel();
            }
            c.cancel.cancel();
            c.view.status = "cancelled".into();
        }
    }
    Json(json!({"cancelled": true})).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
        Router,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt as _;

    fn config(url: &str) -> ConnectionConfig {
        ConnectionConfig {
            id: String::new(),
            name: "Test tools".into(),
            transport: "http".into(),
            url: url.into(),
            command: String::new(),
            args: vec![],
            env_vars: vec![],
            bearer_env: String::new(),
        }
    }
    async fn api(app: &Router, method: &str, path: &str, data: Value) -> (StatusCode, Value) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("host", "127.0.0.1:8181")
                    .header("origin", "http://127.0.0.1:8181")
                    .header("x-camelid-mcp", "1")
                    .header("content-type", "application/json")
                    .body(if data.is_null() {
                        Body::empty()
                    } else {
                        Body::from(data.to_string())
                    })
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }
    async fn fixture() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let server = Router::new().route("/mcp", axum::routing::post(move |Json(message): Json<Value>| {
            let calls = observed.clone();
            async move {
                let Some(id) = message.get("id").cloned() else { return StatusCode::ACCEPTED.into_response(); };
                let result = match message["method"].as_str().unwrap_or("") {
                    "initialize" => json!({"protocolVersion":"2025-11-25", "capabilities":{"tools":{}}, "serverInfo":{"name":"camelid-mcp-fixture","version":"1"}}),
                    "tools/list" => json!({"tools":[{"name":"echo","description":"Return the supplied text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}]}),
                    "tools/call" => {
                        calls.fetch_add(1, Ordering::SeqCst);
                        if message["params"]["arguments"]["text"] == "slow" {
                            tokio::time::sleep(Duration::from_millis(200)).await;
                        }
                        json!({"content":[{"type":"text","text":message["params"]["arguments"]["text"]}]})
                    },
                    _ => json!({}),
                };
                Json(json!({"jsonrpc":"2.0","id":id,"result":result})).into_response()
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, server).await.unwrap();
        });
        (url, calls, task)
    }
    fn app(dir: &tempfile::TempDir) -> Router {
        let mut state = AppState::default();
        state.mcp.path = Arc::new(dir.path().join("connections.json"));
        super::super::router_with_state(state)
    }
    #[test]
    fn configuration_rejects_unsafe_urls_and_preserves_only_environment_names() {
        for url in [
            "http://example.com/mcp",
            "https://user:secret@example.com/mcp",
            "file:///etc/passwd",
            "https://example.com/mcp?token=secret",
        ] {
            assert!(config(url).validate().is_err(), "{url}");
        }
        for url in [
            "https://example.com/mcp",
            "http://127.0.0.1:1234/mcp",
            "http://[::1]:1234/mcp",
        ] {
            assert!(config(url).validate().is_ok(), "{url}");
        }
        let mut c = config("https://example.com/mcp");
        c.bearer_env = "INVALID=secret".into();
        assert!(c.validate().is_err());
        let value = bounded_result(
            json!({"content":[{"type":"text","text":"🦙".repeat(MAX_RESULT_BYTES)}]}),
        );
        assert_eq!(value["truncated"], true);
        assert!(value.to_string().len() < MAX_RESULT_BYTES);
    }
    #[tokio::test]
    async fn mcp_http_approval_denial_replay_and_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let app = app(&dir);
        let (url, count, fixture) = fixture().await;
        let (status, saved) = api(
            &app,
            "POST",
            "/api/mcp/connections",
            serde_json::to_value(config(&url)).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        let id = saved["config"]["id"].as_str().unwrap();
        let (_, connected) = api(
            &app,
            "POST",
            &format!("/api/mcp/connections/{id}/connect"),
            Value::Null,
        )
        .await;
        assert_eq!(connected["connected"], true, "{connected}");
        let key = connected["tools"][0]["key"].as_str().unwrap();
        let (_, pending) = api(
            &app,
            "POST",
            "/api/mcp/calls",
            json!({"tool_key":key,"arguments":{"text":"hello"}}),
        )
        .await;
        let call_id = pending["id"].as_str().unwrap();
        assert_eq!(
            count.load(Ordering::SeqCst),
            0,
            "preparing must not execute"
        );
        let (_, denied) = api(
            &app,
            "POST",
            &format!("/api/mcp/calls/{call_id}/decision"),
            json!({"approved":false}),
        )
        .await;
        assert_eq!(denied["status"], "denied");
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert_eq!(
            api(
                &app,
                "POST",
                &format!("/api/mcp/calls/{call_id}/decision"),
                json!({"approved":true})
            )
            .await
            .0,
            StatusCode::CONFLICT
        );
        let (_, pending) = api(
            &app,
            "POST",
            "/api/mcp/calls",
            json!({"tool_key":key,"arguments":{"text":"hello"}}),
        )
        .await;
        let call_id = pending["id"].as_str().unwrap();
        assert_eq!(
            api(
                &app,
                "POST",
                &format!("/api/mcp/calls/{call_id}/decision"),
                json!({"approved":true})
            )
            .await
            .0,
            StatusCode::ACCEPTED
        );
        let mut completed = false;
        for _ in 0..100 {
            let (_, call) = api(
                &app,
                "GET",
                &format!("/api/mcp/calls/{call_id}"),
                Value::Null,
            )
            .await;
            if call["status"] == "complete" {
                assert_eq!(call["result"]["content"][0]["text"], "hello");
                completed = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(completed, "tool execution did not finish");
        assert_eq!(
            api(
                &app,
                "POST",
                &format!("/api/mcp/calls/{call_id}/decision"),
                json!({"approved":true})
            )
            .await
            .0,
            StatusCode::CONFLICT
        );
        assert_eq!(count.load(Ordering::SeqCst), 1);
        let (_, slow) = api(
            &app,
            "POST",
            "/api/mcp/calls",
            json!({"tool_key":key,"arguments":{"text":"slow"}}),
        )
        .await;
        let slow_id = slow["id"].as_str().unwrap();
        api(
            &app,
            "POST",
            &format!("/api/mcp/calls/{slow_id}/decision"),
            json!({"approved":true}),
        )
        .await;
        for _ in 0..100 {
            if count.load(Ordering::SeqCst) == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            count.load(Ordering::SeqCst),
            2,
            "call must start before cancellation"
        );
        api(
            &app,
            "DELETE",
            &format!("/api/mcp/calls/{slow_id}"),
            Value::Null,
        )
        .await;
        tokio::time::sleep(Duration::from_millis(250)).await;
        let (_, cancelled) = api(
            &app,
            "GET",
            &format!("/api/mcp/calls/{slow_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(
            cancelled["status"], "cancelled",
            "a late result must not replace cancellation"
        );
        assert!(cancelled["result"].is_null());
        assert_eq!(
            api(
                &app,
                "POST",
                &format!("/api/mcp/calls/{slow_id}/decision"),
                json!({"approved":true})
            )
            .await
            .0,
            StatusCode::CONFLICT
        );
        assert_eq!(
            api(
                &app,
                "POST",
                "/api/mcp/calls",
                json!({"tool_key":key,"arguments":{}})
            )
            .await
            .0,
            StatusCode::CONFLICT,
            "Stop must close the running connection"
        );
        api(
            &app,
            "POST",
            &format!("/api/mcp/connections/{id}/connect"),
            Value::Null,
        )
        .await;
        let (_, pending) = api(
            &app,
            "POST",
            "/api/mcp/calls",
            json!({"tool_key":key,"arguments":{}}),
        )
        .await;
        let pending_id = pending["id"].as_str().unwrap();
        api(
            &app,
            "POST",
            &format!("/api/mcp/connections/{id}/disconnect"),
            Value::Null,
        )
        .await;
        assert_eq!(
            api(
                &app,
                "POST",
                &format!("/api/mcp/calls/{pending_id}/decision"),
                json!({"approved":true})
            )
            .await
            .0,
            StatusCode::CONFLICT
        );
        let restored = self::app(&dir);
        let (_, list) = api(&restored, "GET", "/api/mcp/connections", Value::Null).await;
        assert_eq!(list["connections"][0]["config"]["name"], "Test tools");
        assert_eq!(
            list["connections"][0]["connected"], false,
            "restart must never launch a server"
        );
        api(
            &app,
            "DELETE",
            &format!("/api/mcp/connections/{id}"),
            Value::Null,
        )
        .await;
        assert_eq!(
            api(&self::app(&dir), "GET", "/api/mcp/connections", Value::Null)
                .await
                .1["connections"],
            json!([])
        );
        fixture.abort();
    }
    #[tokio::test]
    async fn mcp_routes_reject_foreign_origins_and_missing_intent_header() {
        let dir = tempfile::tempdir().unwrap();
        let app = app(&dir);
        for (origin, intent) in [
            ("https://evil.example", true),
            ("http://127.0.0.1:8181", false),
        ] {
            let mut request = Request::builder()
                .method("GET")
                .uri("/api/mcp/connections")
                .header("host", "127.0.0.1:8181")
                .header("origin", origin);
            if intent {
                request = request.header("x-camelid-mcp", "1");
            }
            let result = app
                .clone()
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(result.status(), StatusCode::FORBIDDEN);
        }
        assert!(!dir.path().join("connections.json").exists());
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_stdio_discovers_and_calls_a_local_process() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("server.py");
        std::fs::write(&script, r#"import sys,json
for line in sys.stdin:
 m=json.loads(line)
 if 'id' not in m: continue
 if m['method']=='initialize': r={'protocolVersion':'2025-11-25','capabilities':{'tools':{}},'serverInfo':{'name':'fixture','version':'1'}}
 elif m['method']=='tools/list': r={'tools':[{'name':'echo','inputSchema':{'type':'object'}}]}
 elif m['method']=='tools/call': r={'content':[{'type':'text','text':'local tool result'}]}
 else: r={}
 print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':r}),flush=True)
"#).unwrap();
        let mut c = config("");
        c.transport = "stdio".into();
        c.command = "/usr/bin/python3".into();
        c.args = vec![script.to_string_lossy().into_owned()];
        c.validate().unwrap();
        let (client, tools) = tokio::time::timeout(Duration::from_secs(10), open(&c))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(tools[0].name, "echo");
        let result = client
            .call_tool(CallToolRequestParams::new("echo").with_arguments(Default::default()))
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(result).unwrap()["content"][0]["text"],
            "local tool result"
        );
        client.cancel().await.unwrap();
    }
}
