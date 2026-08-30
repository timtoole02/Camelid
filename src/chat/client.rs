//! Minimal blocking HTTP/1.1 client for the local Camelid server.
//!
//! `camelid chat` is a thin client over the already-audited local API (see
//! `DECISIONS.md` D6 / `RECON_CHAT.md` §6). Rather than pull in an HTTP-client
//! crate, it reuses the same `TcpStream` + read-to-EOF + de-chunk shape the
//! receipt verifier uses (`src/receipt/verify.rs`), and adds an incremental
//! `data:` reader for the SSE chat lane so streamed terminal output rides the
//! exact wire path `/v1/chat/completions` already serves.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

/// The subset of `/v1/health` the REPL consults (extra fields are ignored).
#[derive(Debug, Deserialize)]
pub struct Health {
    pub ok: bool,
    #[serde(default)]
    pub generation_ready: bool,
    #[serde(default)]
    pub api_surface: Option<String>,
    /// Operator-enforced prompt ceiling for this exact server process. A zero
    /// value means an older server omitted the field; agent mode treats that as
    /// unavailable and fails closed instead of guessing a budget.
    #[serde(default)]
    pub max_prompt_tokens: usize,
    /// Operator-enforced generation ceiling for this exact server process.
    #[serde(default)]
    pub max_generation_tokens: u32,
}

/// One supported/planned row from `/api/capabilities` → `model_compatibility`
/// (extra fields are ignored).
#[derive(Debug, Clone, Deserialize)]
pub struct CompatRow {
    pub id: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub status: String,
    /// Whether this exact row is verified to drive tool-calling (agent mode).
    /// Default false; promoted only with a real tool-call round-trip as evidence.
    #[serde(default)]
    pub tool_capable: bool,
}

impl CompatRow {
    /// The picker's supported predicate, read from the ledger at runtime: a row
    /// is offered only when the engine marks it `supported…`. `planned`,
    /// `active_validation_partial`, and `unsupported` rows are excluded.
    pub fn is_supported(&self) -> bool {
        self.status.starts_with("supported")
    }
}

#[derive(Debug, Deserialize)]
struct Capabilities {
    #[serde(default)]
    model_compatibility: Vec<CompatRow>,
}

#[derive(Debug, Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<LoadedInfo>,
}

/// One currently-loaded model from `GET /v1/models`.
#[derive(Debug, Clone, Deserialize)]
pub struct LoadedInfo {
    pub id: String,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub gguf_sha256: String,
    #[serde(default)]
    pub meta: Option<LoadedMeta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadedMeta {
    #[serde(default)]
    pub n_ctx_train: Option<u32>,
    #[serde(default)]
    pub n_params: Option<u64>,
    #[serde(default)]
    pub size: Option<u64>,
}

impl LoadedInfo {
    pub fn context_length(&self) -> Option<u32> {
        self.meta.as_ref().and_then(|m| m.n_ctx_train)
    }
    /// A short human descriptor, e.g. "8.0B · ctx 8192 · 8.5 GB".
    pub fn descriptor(&self) -> String {
        let Some(meta) = &self.meta else {
            return String::new();
        };
        let mut parts = Vec::new();
        if let Some(p) = meta.n_params {
            parts.push(format!("{:.1}B", p as f64 / 1e9));
        }
        if let Some(c) = meta.n_ctx_train {
            parts.push(format!("ctx {c}"));
        }
        if let Some(s) = meta.size {
            parts.push(format!("{:.1} GB", s as f64 / 1e9));
        }
        parts.join(" · ")
    }
}

/// Outcome of a `/api/models/load` call.
pub enum LoadOutcome {
    /// The model loaded and exposes a Camelid-supported runtime config.
    Loaded { id: String },
    /// The model loaded as metadata but its architecture is not supported. The
    /// message is the engine's typed unsupported-state text, surfaced verbatim.
    Unsupported { message: String },
}

/// How a finished/halted chat stream ended.
#[derive(Debug, PartialEq, Eq)]
pub enum StreamEnd {
    /// `[DONE]` (or a clean close) was reached.
    Done,
    /// Aborted because the cancel flag was set (Ctrl-C mid-stream).
    Cancelled,
}

#[derive(Debug, PartialEq, Eq)]
pub struct StreamStats {
    pub end: StreamEnd,
    pub deltas: u32,
    pub total_ms: u64,
    pub ttft_ms: Option<u64>,
    /// Server-reported prompt tokens from the terminal usage chunk, present
    /// when the request opted in via `stream_options.include_usage` (the agent
    /// lane's calibration signal); `None` otherwise.
    pub prompt_tokens: Option<u32>,
    /// Server-reported completion tokens from the terminal usage chunk. This
    /// remains accurate when a tool-enabled server intentionally releases a
    /// buffered natural-language answer as one visible SSE delta.
    pub completion_tokens: Option<u32>,
    /// Structured tool calls lifted from `delta.tool_calls` (OpenAI shape).
    ///
    /// A tool-enabled stream HOLDS BACK every content delta until the turn is
    /// classified, then emits EITHER one content delta OR these — never both.
    /// So on a tool turn the forwarded content is empty and this is the only
    /// carrier of the call: a caller that reads content alone sees a blank
    /// answer and runs nothing. `chat_turn` documents the same contract for the
    /// blocking path.
    pub tool_calls: Vec<ToolCallOut>,
}

/// Running state folded out of one `/v1/chat/completions` SSE stream.
#[derive(Default)]
struct StreamAccumulator {
    tool_calls: Vec<ToolCallOut>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

impl StreamAccumulator {
    /// Fold one decoded `data:` payload, returning the content delta to forward
    /// (if this chunk carried one).
    ///
    /// Tool calls accumulate BY `index` and append their `arguments`, which is
    /// the OpenAI streaming contract: a call may arrive whole (what this server
    /// does today — one delta per completed call) or split across chunks. The
    /// by-index fold handles both, so the client does not depend on the server
    /// choosing to emit calls in a single event.
    fn absorb<'a>(&mut self, chunk: &'a Value) -> Option<&'a str> {
        if let Some(items) = chunk
            .pointer("/choices/0/delta/tool_calls")
            .and_then(Value::as_array)
        {
            for (ordinal, item) in items.iter().enumerate() {
                let index = item
                    .get("index")
                    .and_then(Value::as_u64)
                    .map(|i| i as usize)
                    .unwrap_or(ordinal);
                if self.tool_calls.len() <= index {
                    self.tool_calls.resize_with(index + 1, ToolCallOut::default);
                }
                let slot = &mut self.tool_calls[index];
                if let Some(name) = item.pointer("/function/name").and_then(Value::as_str) {
                    slot.name.push_str(name);
                }
                if let Some(args) = item.pointer("/function/arguments").and_then(Value::as_str) {
                    slot.arguments.push_str(args);
                }
            }
        }
        if let Some(pt) = chunk
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
        {
            self.prompt_tokens = Some(pt as u32);
        }
        if let Some(ct) = chunk
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
        {
            self.completion_tokens = Some(ct as u32);
        }
        chunk
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
    }

    /// The calls worth reporting. A slot with no name never got a `function.name`
    /// (a malformed or reserved index), and dispatching it would fail anyway.
    fn into_tool_calls(self) -> Vec<ToolCallOut> {
        self.tool_calls
            .into_iter()
            .filter(|call| !call.name.is_empty())
            .collect()
    }
}

#[derive(Clone)]
pub struct Client {
    addr: SocketAddr,
}

impl Client {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    fn connect(&self, read_timeout: Duration) -> std::io::Result<TcpStream> {
        let stream = TcpStream::connect_timeout(&self.addr, Duration::from_secs(2))?;
        stream.set_read_timeout(Some(read_timeout))?;
        stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        Ok(stream)
    }

    /// Send a request and read the whole response (status, JSON body). Used for
    /// the small control-plane calls (health, capabilities, load).
    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        timeout: Duration,
    ) -> anyhow::Result<(u16, Value)> {
        let mut stream = self.connect(timeout)?;
        let raw_request = encode_request(
            method,
            path,
            &self.addr.to_string(),
            body,
            "application/json",
        )?;
        stream.write_all(&raw_request.0)?;
        stream.write_all(&raw_request.1)?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw)?;
        parse_http_response(&raw).map_err(|err| anyhow::anyhow!(err))
    }

    pub(super) fn workspace_request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        token: &str,
        timeout: Duration,
    ) -> anyhow::Result<(u16, Value)> {
        anyhow::ensure!(
            self.addr.ip().is_loopback(),
            "Workspace CLI requires a loopback address"
        );
        let mut stream = self.connect(timeout)?;
        let raw_request = encode_request_with_bearer(
            method,
            path,
            &self.addr.to_string(),
            body,
            "application/json",
            Some(token),
        )?;
        stream.write_all(&raw_request.0)?;
        stream.write_all(&raw_request.1)?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw)?;
        parse_http_response(&raw).map_err(|error| anyhow::anyhow!(error))
    }

    pub(super) fn workspace_events(
        &self,
        path: &str,
        token: &str,
        cancel: &AtomicBool,
        timeout: Duration,
        mut on_event: impl FnMut(&Value) -> bool,
    ) -> anyhow::Result<StreamEnd> {
        anyhow::ensure!(
            self.addr.ip().is_loopback(),
            "Workspace CLI requires a loopback address"
        );
        let deadline = Instant::now() + timeout;
        let mut stream = self.connect(Duration::from_millis(250))?;
        let (head, body) = encode_request_with_bearer(
            "GET",
            path,
            &self.addr.to_string(),
            None,
            "text/event-stream",
            Some(token),
        )?;
        stream.write_all(&head)?;
        stream.write_all(&body)?;

        let mut reader = SseReader::new(stream);
        if reader.read_headers(cancel, Some(deadline))? {
            return Ok(StreamEnd::Cancelled);
        }
        if reader.status != 200 {
            anyhow::bail!(reader.drain_error_body());
        }
        anyhow::ensure!(
            reader
                .content_type
                .as_deref()
                .is_some_and(|value| value.starts_with("text/event-stream")),
            "Workspace event endpoint did not return text/event-stream"
        );

        let mut parse_error = None;
        let end = reader.stream(cancel, Some(deadline), |line| {
            let Some(payload) = line.strip_prefix("data:").map(str::trim) else {
                return SseControl::Continue;
            };
            match serde_json::from_str::<Value>(payload) {
                Ok(event) if on_event(&event) => SseControl::Done,
                Ok(_) => SseControl::Continue,
                Err(error) => {
                    parse_error = Some(error);
                    SseControl::Done
                }
            }
        })?;
        if let Some(error) = parse_error {
            anyhow::bail!("Workspace event stream returned invalid JSON: {error}");
        }
        Ok(end)
    }

    fn request_with_control(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        cancel: &AtomicBool,
        timeout: Duration,
    ) -> anyhow::Result<(u16, Value)> {
        let deadline = Instant::now() + timeout;
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("request cancelled");
        }
        let connect_timeout = timeout.min(Duration::from_secs(2));
        if connect_timeout.is_zero() {
            anyhow::bail!("request exceeded its deadline");
        }
        // When a budget below was set by the caller's deadline (rather than a
        // fixed reachability/stall cap), an OS-level timeout on it IS the
        // deadline elapsing — report it as such instead of leaking the
        // platform's socket-timeout message.
        let mut stream = TcpStream::connect_timeout(&self.addr, connect_timeout).map_err(
            |error| -> anyhow::Error {
                if connect_timeout == timeout && is_timeout(&error) {
                    anyhow::anyhow!("request exceeded its deadline")
                } else {
                    error.into()
                }
            },
        )?;
        stream.set_read_timeout(Some(Duration::from_millis(100)))?;
        let write_timeout = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs(30));
        if write_timeout.is_zero() {
            // set_write_timeout rejects a zero Duration; a spent budget means
            // the deadline elapsed during connect/encode.
            anyhow::bail!("request exceeded its deadline");
        }
        stream.set_write_timeout(Some(write_timeout))?;
        let raw_request = encode_request(
            method,
            path,
            &self.addr.to_string(),
            body,
            "application/json",
        )?;
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("request cancelled");
        }
        let map_write_error = |error: std::io::Error| -> anyhow::Error {
            if write_timeout < Duration::from_secs(30) && is_timeout(&error) {
                anyhow::anyhow!("request exceeded its deadline")
            } else {
                error.into()
            }
        };
        stream.write_all(&raw_request.0).map_err(map_write_error)?;
        stream.write_all(&raw_request.1).map_err(map_write_error)?;

        let mut raw = Vec::new();
        let mut chunk = [0_u8; 4096];
        loop {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("request cancelled");
            }
            if Instant::now() >= deadline {
                anyhow::bail!("request exceeded its deadline");
            }
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(read) => {
                    raw.extend_from_slice(&chunk[..read]);
                    anyhow::ensure!(raw.len() <= 1024 * 1024, "control response exceeded 1 MiB");
                }
                Err(error) if is_timeout(&error) => continue,
                Err(error) => return Err(error.into()),
            }
        }
        parse_http_response(&raw).map_err(|error| anyhow::anyhow!(error))
    }

    /// `GET /v1/health`. Returns `None` when the server is unreachable or
    /// reports not-ok, so callers can probe attach-vs-spawn cheaply.
    pub fn health(&self) -> Option<Health> {
        let (status, body) = self
            .request("GET", "/v1/health", None, Duration::from_secs(5))
            .ok()?;
        if status != 200 {
            return None;
        }
        let health: Health = serde_json::from_value(body).ok()?;
        health.ok.then_some(health)
    }

    /// `GET /api/capabilities` → the supported/planned ledger rows.
    pub fn capabilities(&self) -> anyhow::Result<Vec<CompatRow>> {
        let (status, body) =
            self.request("GET", "/api/capabilities", None, Duration::from_secs(10))?;
        anyhow::ensure!(status == 200, "/api/capabilities returned HTTP {status}");
        let caps: Capabilities = serde_json::from_value(body)?;
        Ok(caps.model_compatibility)
    }

    /// `GET /v1/models` → every model currently loaded in the server (for the
    /// instant loaded-model switcher). Empty on error.
    pub fn list_loaded(&self) -> Vec<LoadedInfo> {
        let Ok((status, body)) = self.request("GET", "/v1/models", None, Duration::from_secs(5))
        else {
            return Vec::new();
        };
        if status != 200 {
            return Vec::new();
        }
        serde_json::from_value::<ModelList>(body)
            .map(|list| list.data)
            .unwrap_or_default()
    }

    /// `GET /api/models/current` → the active `LoadedModel` JSON (for `/info`),
    /// or `None` when nothing is loaded.
    pub fn current_model(&self) -> Option<Value> {
        let (status, body) = self
            .request("GET", "/api/models/current", None, Duration::from_secs(5))
            .ok()?;
        (status == 200).then_some(body)
    }

    /// `POST /api/models/load`. On a recognized-but-unsupported architecture the
    /// server returns 200 with `unsupported_runtime` populated — surfaced here as
    /// `LoadOutcome::Unsupported` carrying the engine's typed message. A hard
    /// failure (bad path, unreadable GGUF) returns the server's error verbatim.
    pub fn load_model(&self, path: &str, id: Option<&str>) -> anyhow::Result<LoadOutcome> {
        let mut req = json!({ "path": path });
        if let Some(id) = id {
            req["id"] = json!(id);
        }
        let (status, body) = self.request(
            "POST",
            "/api/models/load",
            Some(&req),
            Duration::from_secs(600),
        )?;
        if status != 200 {
            let message =
                envelope_message(&body).unwrap_or_else(|| format!("load failed (HTTP {status})"));
            anyhow::bail!(message);
        }
        if let Some(unsupported) = body.get("unsupported_runtime").filter(|v| !v.is_null()) {
            let message = unsupported
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("loaded model architecture is not supported by this Camelid build")
                .to_string();
            return Ok(LoadOutcome::Unsupported { message });
        }
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("model")
            .to_string();
        Ok(LoadOutcome::Loaded { id })
    }

    /// Stream `POST /v1/chat/completions` with `stream=true`. `on_delta` is
    /// called with each assistant content delta as it arrives. `cancel` is polled
    /// between socket reads so Ctrl-C aborts cleanly. Returns how the stream
    /// ended plus the number of content deltas seen (one per generated token on
    /// Camelid's per-token SSE lane → the completion-token count).
    pub fn chat_stream(
        &self,
        request: &Value,
        cancel: &AtomicBool,
        on_delta: impl FnMut(&str),
    ) -> anyhow::Result<(StreamEnd, u32)> {
        let stats = self.chat_stream_timed(request, cancel, on_delta)?;
        Ok((stats.end, stats.deltas))
    }

    pub fn chat_stream_timed(
        &self,
        request: &Value,
        cancel: &AtomicBool,
        on_delta: impl FnMut(&str),
    ) -> anyhow::Result<StreamStats> {
        self.chat_stream_timed_with_timeout(request, cancel, None, on_delta)
    }

    pub fn chat_stream_timed_with_timeout(
        &self,
        request: &Value,
        cancel: &AtomicBool,
        timeout: Option<Duration>,
        mut on_delta: impl FnMut(&str),
    ) -> anyhow::Result<StreamStats> {
        let started = std::time::Instant::now();
        let deadline = timeout.map(|timeout| started + timeout);
        // A short read timeout lets the loop wake to check `cancel` even while
        // the server is mid-generation and no bytes are arriving.
        let mut stream = self.connect(Duration::from_millis(250))?;
        let (head, body) = encode_request(
            "POST",
            "/v1/chat/completions",
            &self.addr.to_string(),
            Some(request),
            "text/event-stream",
        )?;
        stream.write_all(&head)?;
        stream.write_all(&body)?;

        let mut reader = SseReader::new(stream);
        if reader.read_headers(cancel, deadline)? {
            return Ok(StreamStats {
                end: StreamEnd::Cancelled,
                deltas: 0,
                total_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
                ttft_ms: None,
                prompt_tokens: None,
                completion_tokens: None,
                tool_calls: Vec::new(),
            });
        }
        if reader.status != 200 {
            let message = reader.drain_error_body();
            anyhow::bail!(message);
        }

        let mut deltas: u32 = 0;
        let mut ttft_ms = None;
        // Tool calls, plus prompt tokens from the terminal usage chunk when the
        // request opted in via stream_options.include_usage (agent lane).
        let mut acc = StreamAccumulator::default();
        let end = reader.stream(cancel, deadline, |line| {
            if let Some(payload) = line.strip_prefix("data:") {
                let payload = payload.trim();
                if payload == "[DONE]" {
                    return SseControl::Done;
                }
                if let Ok(chunk) = serde_json::from_str::<Value>(payload) {
                    if let Some(content) = acc.absorb(&chunk) {
                        ttft_ms.get_or_insert_with(|| {
                            started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
                        });
                        deltas += 1;
                        on_delta(content);
                    }
                }
            }
            SseControl::Continue
        })?;
        let prompt_tokens = acc.prompt_tokens;
        let completion_tokens = acc.completion_tokens;
        Ok(StreamStats {
            end,
            deltas,
            total_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            ttft_ms,
            prompt_tokens,
            completion_tokens,
            tool_calls: acc.into_tool_calls(),
        })
    }

    /// POST a chat request and return the full assistant turn: text content PLUS
    /// any structured `tool_calls` (OpenAI shape). The server emits structured
    /// tool calls and EMPTIES `content` on a tool call, so an agent loop MUST read
    /// `tool_calls` here — reading only the text loses every tool call.
    pub fn chat_turn(&self, request: &Value) -> anyhow::Result<ChatTurn> {
        // Read deadline: a fast model answers in seconds; the 30-min ceiling is only
        // hit by slow runnable-lane prefill (a 9B qwen35 prefills a long tool prompt
        // at ~1s/token on CPU, and still ~76ms/token on the macOS resident Metal
        // lane), without masking a genuinely hung server.
        let (status, body) = self.request(
            "POST",
            "/v1/chat/completions",
            Some(request),
            Duration::from_secs(1800),
        )?;
        if status != 200 {
            anyhow::bail!(envelope_message(&body).unwrap_or_else(|| format!("HTTP {status}")));
        }
        Ok(parse_chat_turn(&body))
    }

    pub fn generation_preflight(&self, request: &Value) -> anyhow::Result<u32> {
        let (status, body) = self.request(
            "POST",
            "/api/generation/preflight",
            Some(request),
            Duration::from_secs(30),
        )?;
        parse_generation_preflight(status, &body)
    }

    pub fn generation_preflight_with_control(
        &self,
        request: &Value,
        cancel: &AtomicBool,
        timeout: Duration,
    ) -> anyhow::Result<u32> {
        let (status, body) = self.request_with_control(
            "POST",
            "/api/generation/preflight",
            Some(request),
            cancel,
            timeout,
        )?;
        parse_generation_preflight(status, &body)
    }

    /// Text-only convenience over [`chat_turn`] for the plain chat UI (which never
    /// supplies tools, so `content` always carries the answer).
    pub fn chat_blocking(
        &self,
        request: &Value,
    ) -> anyhow::Result<(String, Option<u32>, Option<u32>)> {
        let turn = self.chat_turn(request)?;
        Ok((turn.content, turn.prompt_tokens, turn.completion_tokens))
    }
}

/// One assistant turn from `/v1/chat/completions`.
pub struct ChatTurn {
    pub content: String,
    /// Structured tool calls (OpenAI shape); `content` is empty when this is set.
    pub tool_calls: Vec<ToolCallOut>,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

/// One structured tool call (OpenAI shape): a name + a JSON-encoded args string.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCallOut {
    pub name: String,
    pub arguments: String,
}

/// Extract the assistant turn (content + structured tool calls + token counts)
/// from a `/v1/chat/completions` response body.
fn parse_chat_turn(body: &Value) -> ChatTurn {
    let content = body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let tool_calls = body
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let name = tc
                        .pointer("/function/name")
                        .and_then(Value::as_str)?
                        .to_string();
                    let arguments = tc
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}")
                        .to_string();
                    Some(ToolCallOut { name, arguments })
                })
                .collect()
        })
        .unwrap_or_default();
    let prompt_tokens = body
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .map(|n| n as u32);
    let completion_tokens = body
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .map(|n| n as u32);
    ChatTurn {
        content,
        tool_calls,
        prompt_tokens,
        completion_tokens,
    }
}

/// Pull `error.message` out of a Camelid `ErrorEnvelope` body, if present.
fn envelope_message(body: &Value) -> Option<String> {
    body.pointer("/error/message")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Decode a generation-preflight result. A normal 200 returns its count. The
/// sole non-200 exception is the server's typed prompt-ceiling rejection when
/// it also carries the authoritative rendered count: fitting may safely trim
/// optional context and retry. Artifact, template, tokenization, cancellation,
/// and every other error remain errors and are never mistaken for size signals.
fn parse_generation_preflight(status: u16, body: &Value) -> anyhow::Result<u32> {
    let count_is_authoritative = status == 200
        || (status == 413
            && body.pointer("/error/code").and_then(Value::as_str)
                == Some("prompt_token_limit_exceeded"));
    if !count_is_authoritative {
        anyhow::bail!(envelope_message(body).unwrap_or_else(|| format!("HTTP {status}")));
    }
    let count = body
        .get("prompt_token_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("generation preflight omitted prompt_token_count"))?;
    u32::try_from(count).map_err(|_| anyhow::anyhow!("prompt token count exceeds u32"))
}

/// Build the request head (bytes up to and including the blank line) and the
/// separate body bytes.
fn encode_request(
    method: &str,
    path: &str,
    host: &str,
    body: Option<&Value>,
    accept: &str,
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    encode_request_with_bearer(method, path, host, body, accept, None)
}

fn encode_request_with_bearer(
    method: &str,
    path: &str,
    host: &str,
    body: Option<&Value>,
    accept: &str,
    bearer: Option<&str>,
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let body_bytes = match body {
        Some(value) => serde_json::to_vec(value)?,
        None => Vec::new(),
    };
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nAccept: {accept}\r\nConnection: close\r\n"
    );
    if let Some(token) = bearer {
        anyhow::ensure!(
            !token.is_empty() && !token.contains(['\r', '\n']),
            "invalid bearer token"
        );
        head.push_str("Authorization: Bearer ");
        head.push_str(token);
        head.push_str("\r\n");
    }
    if !body_bytes.is_empty() {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", body_bytes.len()));
    }
    head.push_str("\r\n");
    Ok((head.into_bytes(), body_bytes))
}

/// Incremental reader for a chunked `text/event-stream` body.
struct SseReader {
    stream: TcpStream,
    /// Raw bytes read from the socket but not yet consumed by the decoder.
    pending: Vec<u8>,
    status: u16,
    chunked: bool,
    content_type: Option<String>,
}

enum SseControl {
    Continue,
    Done,
}

impl SseReader {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            pending: Vec::new(),
            status: 0,
            chunked: false,
            content_type: None,
        }
    }

    /// Read until the header terminator, recording status + transfer-encoding and
    /// leaving any post-header body bytes in `pending`. Returns `true` if Ctrl-C
    /// cancelled while waiting (the server may prefill for many seconds before the
    /// first byte, so a read timeout here is normal and is retried, not an error).
    fn read_headers(
        &mut self,
        cancel: &AtomicBool,
        deadline: Option<std::time::Instant>,
    ) -> anyhow::Result<bool> {
        let mut scratch = [0u8; 4096];
        loop {
            if let Some(end) = find(&self.pending, b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&self.pending[..end]);
                let mut lines = head.split("\r\n");
                let status_line = lines.next().unwrap_or_default();
                self.status = status_line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|code| code.parse().ok())
                    .ok_or_else(|| {
                        anyhow::anyhow!("malformed HTTP status line: {status_line:?}")
                    })?;
                for line in lines {
                    if let Some((name, value)) = line.split_once(':') {
                        let name = name.trim();
                        if name.eq_ignore_ascii_case("transfer-encoding")
                            && value.to_ascii_lowercase().contains("chunked")
                        {
                            self.chunked = true;
                        } else if name.eq_ignore_ascii_case("content-type") {
                            self.content_type = Some(value.trim().to_ascii_lowercase());
                        }
                    }
                }
                self.pending.drain(..end + 4);
                return Ok(false);
            }
            if cancel.load(Ordering::Relaxed) {
                return Ok(true);
            }
            if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
                anyhow::bail!("chat stream exceeded its model-step deadline");
            }
            match self.stream.read(&mut scratch) {
                Ok(0) => anyhow::bail!("connection closed before HTTP headers completed"),
                Ok(n) => self.pending.extend_from_slice(&scratch[..n]),
                Err(err) if is_timeout(&err) => continue,
                Err(err) => return Err(err.into()),
            }
        }
    }

    /// On a non-200 status, read the rest of the (small) error body and extract a
    /// human-readable message.
    fn drain_error_body(&mut self) -> String {
        let mut scratch = [0u8; 4096];
        // Bounded read-to-close, tolerating the short read timeout (cap ~10s).
        let mut idle = 0;
        loop {
            match self.stream.read(&mut scratch) {
                Ok(0) => break,
                Ok(n) => {
                    idle = 0;
                    self.pending.extend_from_slice(&scratch[..n]);
                }
                Err(ref err) if is_timeout(err) => {
                    idle += 1;
                    if idle >= 40 {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let body = if self.chunked {
            decode_chunked(&self.pending).unwrap_or_else(|_| self.pending.clone())
        } else {
            self.pending.clone()
        };
        serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| envelope_message(&v))
            .unwrap_or_else(|| {
                format!(
                    "chat request failed (HTTP {}): {}",
                    self.status,
                    String::from_utf8_lossy(&body).trim()
                )
            })
    }

    /// Drive the body: de-chunk on the fly, split into lines, and hand each line
    /// to `on_line`. Stops on `[DONE]`, EOF, or a set cancel flag.
    fn stream(
        &mut self,
        cancel: &AtomicBool,
        deadline: Option<std::time::Instant>,
        mut on_line: impl FnMut(&str) -> SseControl,
    ) -> anyhow::Result<StreamEnd> {
        let mut decoder = ChunkDecoder::new(self.chunked);
        let mut line_acc: Vec<u8> = Vec::new();
        let mut scratch = [0u8; 4096];
        // Process anything that arrived alongside the headers first.
        let seeded = std::mem::take(&mut self.pending);
        if Self::feed(&seeded, &mut decoder, &mut line_acc, &mut on_line) {
            return Ok(StreamEnd::Done);
        }
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Ok(StreamEnd::Cancelled);
            }
            if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
                anyhow::bail!("chat stream exceeded its model-step deadline");
            }
            match self.stream.read(&mut scratch) {
                Ok(0) => return Ok(StreamEnd::Done),
                Ok(n) => {
                    if Self::feed(&scratch[..n], &mut decoder, &mut line_acc, &mut on_line) {
                        return Ok(StreamEnd::Done);
                    }
                }
                Err(err) if is_timeout(&err) => continue,
                Err(err) => return Err(err.into()),
            }
        }
    }

    /// Feed raw bytes through the de-chunker into `line_acc`, emitting complete
    /// lines. Returns true once `on_line` signalled `Done`.
    fn feed(
        bytes: &[u8],
        decoder: &mut ChunkDecoder,
        line_acc: &mut Vec<u8>,
        on_line: &mut impl FnMut(&str) -> SseControl,
    ) -> bool {
        decoder.push(bytes, line_acc);
        loop {
            let Some(nl) = line_acc.iter().position(|&b| b == b'\n') else {
                return false;
            };
            let mut line = line_acc[..nl].to_vec();
            line_acc.drain(..nl + 1);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let text = String::from_utf8_lossy(&line);
            if matches!(on_line(&text), SseControl::Done) {
                return true;
            }
        }
    }
}

/// Incremental HTTP chunked-transfer decoder. When `chunked` is false it is a
/// pass-through (some servers may close with identity encoding).
struct ChunkDecoder {
    chunked: bool,
    raw: Vec<u8>,
    /// Remaining bytes of the current chunk's data, or `None` while reading the
    /// next size line.
    remaining: Option<usize>,
    done: bool,
}

impl ChunkDecoder {
    fn new(chunked: bool) -> Self {
        Self {
            chunked,
            raw: Vec::new(),
            remaining: None,
            done: false,
        }
    }

    /// Append `bytes`; push any newly-decoded body bytes onto `out`.
    fn push(&mut self, bytes: &[u8], out: &mut Vec<u8>) {
        if !self.chunked {
            out.extend_from_slice(bytes);
            return;
        }
        self.raw.extend_from_slice(bytes);
        while !self.done {
            match self.remaining {
                None => {
                    // Need a full `<hex>\r\n` size line.
                    let Some(eol) = find(&self.raw, b"\r\n") else {
                        return;
                    };
                    let size_line = String::from_utf8_lossy(&self.raw[..eol]);
                    let hex = size_line.split(';').next().unwrap_or_default().trim();
                    let size = usize::from_str_radix(hex, 16).unwrap_or(0);
                    self.raw.drain(..eol + 2);
                    if size == 0 {
                        self.done = true;
                        return;
                    }
                    self.remaining = Some(size);
                }
                Some(0) => {
                    // Consume the chunk's trailing CRLF, then read the next size.
                    if self.raw.len() < 2 {
                        return;
                    }
                    self.raw.drain(..2);
                    self.remaining = None;
                }
                Some(n) => {
                    if self.raw.is_empty() {
                        return;
                    }
                    let take = n.min(self.raw.len());
                    out.extend_from_slice(&self.raw[..take]);
                    self.raw.drain(..take);
                    self.remaining = Some(n - take);
                }
            }
        }
    }
}

fn is_timeout(err: &std::io::Error) -> bool {
    matches!(
        err.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Parse a full HTTP/1.1 response read to EOF (status, JSON body), de-chunking
/// when needed. Mirrors `src/receipt/verify.rs::parse_http_response`.
fn parse_http_response(raw: &[u8]) -> Result<(u16, Value), String> {
    let header_end = find(raw, b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response: missing header terminator".to_string())?;
    let head = String::from_utf8_lossy(&raw[..header_end]);
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| format!("malformed HTTP status line: {status_line:?}"))?;

    let mut chunked = false;
    let mut content_length: Option<usize> = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        } else if name == "content-length" {
            content_length = value.parse().ok();
        }
    }

    let body_raw = &raw[header_end + 4..];
    let body = if chunked {
        decode_chunked(body_raw)?
    } else if let Some(length) = content_length {
        body_raw.get(..length).unwrap_or(body_raw).to_vec()
    } else {
        body_raw.to_vec()
    };
    if body.is_empty() {
        return Ok((status, Value::Null));
    }
    let value = serde_json::from_slice(&body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&body).trim().to_string()));
    Ok((status, value))
}

fn decode_chunked(mut raw: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let line_end = find(raw, b"\r\n")
            .ok_or_else(|| "malformed chunked body: missing size line".to_string())?;
        let size_line = String::from_utf8_lossy(&raw[..line_end]);
        let size_text = size_line.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| format!("malformed chunk size: {size_text:?}"))?;
        raw = &raw[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        let chunk = raw
            .get(..size)
            .ok_or_else(|| "malformed chunked body: truncated chunk".to_string())?;
        out.extend_from_slice(chunk);
        raw = raw.get(size + 2..).unwrap_or(&[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::net::TcpListener;
    use std::sync::mpsc;

    #[test]
    fn health_deserializes_runtime_prompt_and_generation_ceilings() {
        let health: Health = serde_json::from_value(json!({
            "ok": true,
            "generation_ready": true,
            "api_surface": "local",
            "max_prompt_tokens": 384,
            "max_generation_tokens": 96
        }))
        .unwrap();
        assert_eq!(health.max_prompt_tokens, 384);
        assert_eq!(health.max_generation_tokens, 96);

        let legacy: Health = serde_json::from_value(json!({"ok": true})).unwrap();
        assert_eq!(legacy.max_prompt_tokens, 0);
        assert_eq!(legacy.max_generation_tokens, 0);
    }

    #[test]
    fn preflight_prompt_ceiling_error_returns_authoritative_count_for_trimming() {
        let body = json!({
            "error": {
                "message": "prompt encoded to 110 tokens, above the server ceiling of 100",
                "type": "invalid_request",
                "code": "prompt_token_limit_exceeded",
                "param": "prompt"
            },
            "prompt_token_count": 110
        });
        assert_eq!(parse_generation_preflight(413, &body).unwrap(), 110);

        let unrelated = json!({
            "error": {
                "message": "chat template rejected system role",
                "code": "unsupported_chat_template"
            },
            "prompt_token_count": 110
        });
        assert!(parse_generation_preflight(422, &unrelated).is_err());

        let missing_count = json!({
            "error": {
                "message": "over limit",
                "code": "prompt_token_limit_exceeded"
            }
        });
        assert!(parse_generation_preflight(413, &missing_count).is_err());
    }

    /// Drive a canned SSE transcript through the production chunk decoder and
    /// the production [`StreamAccumulator`], returning what the caller of
    /// `chat_stream_timed_with_timeout` would observe: forwarded content and
    /// lifted tool calls.
    fn replay_sse(events: &[&str]) -> (String, Vec<ToolCallOut>) {
        let mut wire = String::new();
        for event in events {
            wire.push_str(&format!("{:x}\r\n{event}\r\n", event.len()));
        }
        wire.push_str("0\r\n\r\n");
        let mut decoder = ChunkDecoder::new(true);
        let mut line_acc = Vec::new();
        let mut acc = StreamAccumulator::default();
        let mut content = String::new();
        for slice in wire.as_bytes().chunks(9) {
            let finished = SseReader::feed(slice, &mut decoder, &mut line_acc, &mut |line| {
                if let Some(payload) = line.strip_prefix("data:") {
                    let payload = payload.trim();
                    if payload == "[DONE]" {
                        return SseControl::Done;
                    }
                    if let Ok(chunk) = serde_json::from_str::<Value>(payload) {
                        if let Some(delta) = acc.absorb(&chunk) {
                            content.push_str(delta);
                        }
                    }
                }
                SseControl::Continue
            });
            if finished {
                break;
            }
        }
        (content, acc.into_tool_calls())
    }

    /// The regression this exists for: a tool-enabled stream emits NO content
    /// deltas and carries the call only in `delta.tool_calls`. Reading content
    /// alone yielded an empty turn, so the agent loop ended with a blank answer
    /// and never dispatched a tool. Transcript shape mirrors the server's
    /// `chat_tool_call_delta_uses_openai_stream_shape` assertion in `api`.
    #[test]
    fn stream_lifts_tool_calls_when_content_is_held_back() {
        let (content, calls) = replay_sse(&[
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\
             \"type\":\"function\",\"function\":{\"name\":\"read_file\",\
             \"arguments\":\"{\\\"path\\\":\\\"src/main.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1234}}\n\n",
            "data: [DONE]\n\n",
        ]);
        assert_eq!(content, "", "a tool turn streams no content deltas");
        assert_eq!(
            calls,
            vec![ToolCallOut {
                name: "read_file".into(),
                arguments: r#"{"path":"src/main.rs"}"#.into(),
            }],
            "the call must survive as structured output"
        );
    }

    #[test]
    fn stream_accumulator_keeps_terminal_completion_usage() {
        let mut acc = StreamAccumulator::default();
        assert_eq!(
            acc.absorb(&json!({
                "choices": [],
                "usage": {"prompt_tokens": 1234, "completion_tokens": 77}
            })),
            None
        );
        assert_eq!(acc.prompt_tokens, Some(1234));
        assert_eq!(acc.completion_tokens, Some(77));
    }

    /// Two calls in one turn keep their order and stay separate.
    #[test]
    fn stream_lifts_multiple_tool_calls_by_index() {
        let (_, calls) = replay_sse(&[
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[\
             {\"index\":0,\"function\":{\"name\":\"list_dir\",\"arguments\":\"{}\"}},\
             {\"index\":1,\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"p\\\":1}\"}}\
             ]}}]}\n\n",
            "data: [DONE]\n\n",
        ]);
        let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["list_dir", "read_file"]);
        assert_eq!(calls[1].arguments, r#"{"p":1}"#);
    }

    /// The OpenAI contract allows a call to arrive in fragments. This server
    /// emits each call whole today, so the by-index fold is what keeps the
    /// client from depending on that choice.
    #[test]
    fn stream_folds_fragmented_tool_call_arguments() {
        let (_, calls) = replay_sse(&[
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\
             \"function\":{\"name\":\"write_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\
             \"function\":{\"arguments\":\"\\\"a.txt\\\"}\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        ]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].arguments, r#"{"path":"a.txt"}"#);
    }

    /// A plain answer is unaffected: content still streams and no call is
    /// invented.
    #[test]
    fn stream_without_tool_calls_still_yields_content() {
        let (content, calls) = replay_sse(&[
            "data: {\"choices\":[{\"delta\":{\"content\":\"all \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\n",
            "data: [DONE]\n\n",
        ]);
        assert_eq!(content, "all done");
        assert!(calls.is_empty());
    }

    #[test]
    fn workspace_request_encodes_bearer_without_browser_headers() {
        let token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let (head, body) = encode_request_with_bearer(
            "GET",
            "/api/agent/workspace/threads?workspace=C%3A%5Cwork",
            "127.0.0.1:8181",
            None,
            "application/json",
            Some(token),
        )
        .unwrap();
        let head = String::from_utf8(head).unwrap();
        assert!(head.contains(&format!("Authorization: Bearer {token}\r\n")));
        assert!(!head.contains("Origin:"));
        assert!(!head.contains("Sec-Fetch-Site:"));
        assert!(body.is_empty());
        assert!(encode_request_with_bearer(
            "GET",
            "/",
            "127.0.0.1:8181",
            None,
            "application/json",
            Some("bad\r\ntoken")
        )
        .is_err());
    }

    #[test]
    fn workspace_events_stream_authenticated_json_envelopes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let expected_authorization = format!("Authorization: Bearer {token}\r\n");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                request.extend_from_slice(&buffer[..read]);
                if read == 0 || find(&request, b"\r\n\r\n").is_some() {
                    break;
                }
            }
            let request = String::from_utf8(request).unwrap();
            assert!(request.contains(&expected_authorization));
            let event = "data: {\"sequence\":1,\"session_id\":\"workspace-1\",\"event\":\"session.finished\",\"outcome\":\"answered\"}\n\n";
            let body = format!("{:X}\r\n{event}\r\n0\r\n\r\n", event.len());
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
            )
            .unwrap();
        });

        let mut events = Vec::new();
        let end = Client::new(addr)
            .workspace_events(
                "/api/agent/workspace/sessions/workspace-1/events",
                token,
                &AtomicBool::new(false),
                Duration::from_secs(5),
                |event| {
                    events.push(event.clone());
                    event["event"] == "session.finished"
                },
            )
            .unwrap();
        assert_eq!(end, StreamEnd::Done);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["outcome"], "answered");
        server.join().unwrap();
    }

    #[test]
    fn workspace_events_reject_a_non_sse_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
            )
            .unwrap();
        });
        let error = Client::new(addr)
            .workspace_events(
                "/api/agent/workspace/sessions/workspace-1/events",
                &"0".repeat(64),
                &AtomicBool::new(false),
                Duration::from_secs(5),
                |_| false,
            )
            .unwrap_err();
        assert!(error.to_string().contains("text/event-stream"));
        server.join().unwrap();
    }

    #[test]
    fn parse_chat_turn_reads_structured_tool_calls() {
        // The server empties `content` and emits a structured tool_calls array
        // when the model calls a tool. The agent loop must read it from here.
        let body = json!({
            "choices": [{ "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "1", "type": "function",
                    "function": { "name": "read_file", "arguments": "{\"path\":\"notes.txt\"}" }
                }]
            }}],
            "usage": { "prompt_tokens": 11, "completion_tokens": 7 }
        });
        let turn = parse_chat_turn(&body);
        assert!(turn.content.is_empty());
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "read_file");
        assert!(turn.tool_calls[0].arguments.contains("notes.txt"));
        assert_eq!(turn.completion_tokens, Some(7));
    }

    #[test]
    fn parse_chat_turn_reads_plain_content() {
        let body = json!({"choices":[{"message":{"role":"assistant","content":"hello"}}]});
        let turn = parse_chat_turn(&body);
        assert_eq!(turn.content, "hello");
        assert!(turn.tool_calls.is_empty());
    }

    /// Decode a chunked SSE body delivered in several arbitrary socket reads and
    /// confirm the `data:` deltas come out intact and in order, ending at
    /// `[DONE]`. This is the Phase 8 SSE/delta parser unit test.
    #[test]
    fn sse_deltas_survive_arbitrary_chunk_splits() {
        // Two events, each its own HTTP chunk, plus the terminating 0-chunk.
        let event_a = "data: {\"choices\":[{\"delta\":{\"content\":\"Cert\"}}]}\n\n";
        let event_b = "data: {\"choices\":[{\"delta\":{\"content\":\"ainly\"}}]}\n\n";
        let done = "data: [DONE]\n\n";
        let wire = format!(
            "{:x}\r\n{event_a}\r\n{:x}\r\n{event_b}\r\n{:x}\r\n{done}\r\n0\r\n\r\n",
            event_a.len(),
            event_b.len(),
            done.len(),
        );
        let bytes = wire.into_bytes();

        // Feed the wire bytes in 7-byte slices to exercise partial size lines and
        // split chunk data.
        let mut decoder = ChunkDecoder::new(true);
        let mut line_acc = Vec::new();
        let mut collected = String::new();
        let mut done_seen = false;
        for slice in bytes.chunks(7) {
            let finished = SseReader::feed(slice, &mut decoder, &mut line_acc, &mut |line| {
                if let Some(payload) = line.strip_prefix("data:") {
                    let payload = payload.trim();
                    if payload == "[DONE]" {
                        return SseControl::Done;
                    }
                    let v: Value = serde_json::from_str(payload).unwrap();
                    collected.push_str(
                        v.pointer("/choices/0/delta/content")
                            .unwrap()
                            .as_str()
                            .unwrap(),
                    );
                }
                SseControl::Continue
            });
            if finished {
                done_seen = true;
                break;
            }
        }
        assert!(done_seen, "stream must terminate on [DONE]");
        assert_eq!(collected, "Certainly");
    }

    /// End-to-end over a real socket: the whole `chat_stream_timed_with_timeout`
    /// path, not just the fold. Proves the lifted calls actually reach the
    /// caller through `StreamStats`, which is what the agent loop reads.
    #[test]
    fn chat_stream_surfaces_tool_calls_over_a_socket() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = Client::new(listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            // Drain the ENTIRE request — head AND body — before responding.
            //
            // The sibling socket tests here stop at `\r\n\r\n` because their
            // requests are bodyless GETs. This one POSTs JSON, so stopping at the
            // head leaves the body sitting unread in the receive queue. Closing a
            // socket with unread data is what makes Windows send an abortive RST
            // instead of a FIN, and the client then fails the read with
            // `os error 10054` rather than seeing a clean end of stream. It is a
            // race — whether the reset beats the client's last read — so it passed
            // on one CI run and failed on the next with no code change.
            //
            // A single `read` is also wrong on its own terms: it can return a
            // partial request regardless of buffer size.
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            let mut expected_len: Option<usize> = None;
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if expected_len.is_none() {
                    if let Some(head_end) = find(&request, b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
                        let body_len = head
                            .split("content-length:")
                            .nth(1)
                            .and_then(|rest| rest.split("\r\n").next())
                            .and_then(|value| value.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        expected_len = Some(head_end + 4 + body_len);
                    }
                }
                if expected_len.is_some_and(|need| request.len() >= need) {
                    break;
                }
            }
            let body = concat!(
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,",
                "\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"list_dir\",",
                "\"arguments\":\"{\\\"path\\\":\\\".\\\"}\"}}]}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: [DONE]\n\n",
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = std::io::Write::write_all(&mut stream, response.as_bytes());
            let _ = std::io::Write::flush(&mut stream);
        });

        let mut forwarded = String::new();
        let stats = client
            .chat_stream_timed_with_timeout(
                &json!({"stream": true}),
                &AtomicBool::new(false),
                Some(Duration::from_secs(5)),
                |delta| forwarded.push_str(delta),
            )
            .unwrap();
        server.join().unwrap();

        assert_eq!(stats.end, StreamEnd::Done);
        assert_eq!(forwarded, "", "a tool turn forwards no content deltas");
        assert_eq!(stats.deltas, 0);
        assert_eq!(
            stats.tool_calls,
            vec![ToolCallOut {
                name: "list_dir".into(),
                arguments: r#"{"path":"."}"#.into(),
            }],
        );
    }

    #[test]
    fn sse_header_read_honors_model_step_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (release_tx, release_rx) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let (_stream, _) = listener.accept().unwrap();
            let _ = release_rx.recv_timeout(Duration::from_secs(1));
        });
        let stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_millis(10)))
            .unwrap();
        let mut reader = SseReader::new(stream);
        let error = reader
            .read_headers(
                &AtomicBool::new(false),
                Some(std::time::Instant::now() + Duration::from_millis(30)),
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("chat stream exceeded its model-step deadline"));
        let _ = release_tx.send(());
        server.join().unwrap();
    }

    #[test]
    fn generation_preflight_control_honors_cancellation_and_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = Client::new(listener.local_addr().unwrap());
        let cancelled = client
            .generation_preflight_with_control(
                &json!({}),
                &AtomicBool::new(true),
                Duration::from_secs(1),
            )
            .unwrap_err();
        assert!(cancelled.to_string().contains("cancelled"));

        let server = std::thread::spawn(move || {
            let (_stream, _) = listener.accept().unwrap();
            std::thread::park_timeout(Duration::from_secs(1));
        });
        let error = client
            .generation_preflight_with_control(
                &json!({}),
                &AtomicBool::new(false),
                Duration::from_millis(30),
            )
            .unwrap_err();
        assert!(error.to_string().contains("deadline"));
        server.thread().unpark();
        server.join().unwrap();
    }

    #[test]
    fn parses_content_length_and_chunked_bodies() {
        let cl = b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{\"ok\":true}";
        let (status, value) = parse_http_response(cl).unwrap();
        assert_eq!(status, 200);
        assert_eq!(value["ok"], true);

        let chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"a\":\r\n5\r\ntrue}\r\n0\r\n\r\n";
        let (status, value) = parse_http_response(chunked).unwrap();
        assert_eq!(status, 200);
        assert_eq!(value["a"], true);
    }

    #[test]
    fn supported_predicate_reads_the_ledger_status() {
        let supported = CompatRow {
            id: "tinyllama_1_1b_chat_q8_0".into(),
            family: "llama_bpe_decoder".into(),
            quantization: "Q8_0".into(),
            status: "supported_exact_row_smoke".into(),
            tool_capable: false,
        };
        let planned = CompatRow {
            id: "qwen2_5_7b_instruct_q8_0".into(),
            family: "qwen2".into(),
            quantization: "Q8_0".into(),
            status: "planned".into(),
            tool_capable: false,
        };
        assert!(supported.is_supported());
        assert!(!planned.is_supported());
    }
}
