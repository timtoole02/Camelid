//! Audit events for the agent tool loop (Task 3).
//!
//! Every executed tool is bracketed by two events — `agent.tool_call` before and
//! `agent.tool_result` after — carrying the tool name, the approval tier that was
//! applied, a **digest** of the arguments (a SHA-256 hash, never the raw args, so
//! secrets in tool inputs are not leaked to a sink), the outcome, and the wall
//! duration. Events are delivered through a pluggable [`AuditSink`]:
//!
//! - [`NoopSink`] — the default; emits nothing (an unconfigured deployment never
//!   errors and never pays a cost).
//! - [`WebhookSink`] — POSTs each event as JSON to a configured URL on a
//!   background thread. Delivery is **non-blocking**: the agent loop hands the
//!   event to a bounded channel and moves on; if the channel is full the event is
//!   **dropped** rather than stalling the loop (Decision: liveness of the agent
//!   loop outranks audit completeness; drops are the documented failure mode).
//! - [`InMemorySink`] — collects events in memory (tests / embedding).
//!
//! No endpoint is hardcoded anywhere; the URL comes from config or the
//! `CAMELID_AUDIT_WEBHOOK` environment variable (see [`sink_from_config`]). The
//! event schema is documented in `AUDIT_EVENTS.md`.

use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use super::tools::ToolOutcome;

/// Which half of the bracket an event is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditKind {
    /// Emitted immediately before a tool executes.
    ToolCall,
    /// Emitted immediately after a tool executes.
    ToolResult,
}

impl AuditKind {
    /// The dotted event name carried in the payload.
    pub fn event_name(self) -> &'static str {
        match self {
            AuditKind::ToolCall => "agent.tool_call",
            AuditKind::ToolResult => "agent.tool_result",
        }
    }
}

/// The outcome recorded on a `tool_result` event (never the raw output text —
/// that can itself contain secrets or injected content).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditOutcome {
    Ok,
    Error,
}

impl AuditOutcome {
    fn label(self) -> &'static str {
        match self {
            AuditOutcome::Ok => "ok",
            AuditOutcome::Error => "error",
        }
    }
}

/// One audit event. `outcome`/`duration_ms` are present only on `ToolResult`.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    pub kind: AuditKind,
    pub timestamp_unix_ms: u128,
    pub tool: String,
    /// The approval tier that was applied (`auto` / `confirm` / `deny`).
    pub tier: &'static str,
    /// `sha256:<hex>` over the canonical JSON of the tool arguments. A digest,
    /// not the arguments, so secrets in args never reach a sink.
    pub args_digest: String,
    pub outcome: Option<AuditOutcome>,
    pub duration_ms: Option<u64>,
    /// How many images the tool result carried. A count, never the bytes — the
    /// same posture as `outcome`, since image data from an untrusted tool is
    /// exactly the kind of payload this trail must record without transmitting.
    /// `None` on `tool_call`.
    pub images: Option<usize>,
}

impl AuditEvent {
    /// The `agent.tool_call` event (pre-execution).
    pub fn call(tool: &str, tier: &'static str, args_digest: String) -> Self {
        Self {
            kind: AuditKind::ToolCall,
            timestamp_unix_ms: now_ms(),
            tool: tool.to_string(),
            tier,
            args_digest,
            outcome: None,
            duration_ms: None,
            images: None,
        }
    }

    /// The `agent.tool_result` event (post-execution).
    pub fn result(
        tool: &str,
        tier: &'static str,
        args_digest: String,
        outcome: &ToolOutcome,
        duration: Duration,
    ) -> Self {
        Self {
            kind: AuditKind::ToolResult,
            timestamp_unix_ms: now_ms(),
            tool: tool.to_string(),
            tier,
            args_digest,
            outcome: Some(if outcome.is_err() {
                AuditOutcome::Error
            } else {
                AuditOutcome::Ok
            }),
            duration_ms: Some(duration.as_millis() as u64),
            images: Some(outcome.images().len()),
        }
    }

    pub fn event_name(&self) -> &'static str {
        self.kind.event_name()
    }

    /// The wire payload (stable shape — see `AUDIT_EVENTS.md`).
    pub fn to_json(&self) -> Value {
        json!({
            "event": self.event_name(),
            "timestamp_unix_ms": self.timestamp_unix_ms,
            "tool": self.tool,
            "approval_tier": self.tier,
            "args_digest": self.args_digest,
            "outcome": self.outcome.map(AuditOutcome::label),
            "duration_ms": self.duration_ms,
            "images": self.images,
        })
    }
}

/// A SHA-256 digest of the tool arguments, tagged `sha256:`. Hashing the canonical
/// JSON keeps the audit trail correlatable without ever recording the raw args.
pub fn digest_args(args: &Value) -> String {
    use sha2::{Digest, Sha256};
    // serde_json::Value orders object keys deterministically, so the canonical
    // bytes — and therefore the digest — are stable across runs.
    let canonical = serde_json::to_vec(args).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(&canonical);
    format!("sha256:{:x}", hasher.finalize())
}

/// The pluggable audit destination. `emit` must never block the agent loop.
pub trait AuditSink: Send + Sync {
    fn emit(&self, event: &AuditEvent);
}

/// The default: emit nothing. An unconfigured deployment audits nothing rather
/// than erroring.
pub struct NoopSink;

impl AuditSink for NoopSink {
    fn emit(&self, _event: &AuditEvent) {}
}

/// Collects events in memory (tests / in-process consumers). Cheap to clone — all
/// clones share the same buffer. Part of the public sink surface even though the
/// shipped binary only wires the no-op and webhook sinks.
#[derive(Default, Clone)]
#[allow(dead_code)]
pub struct InMemorySink {
    events: Arc<Mutex<Vec<AuditEvent>>>,
}

#[allow(dead_code)]
impl InMemorySink {
    pub fn events(&self) -> Vec<AuditEvent> {
        self.events.lock().expect("audit buffer poisoned").clone()
    }
}

impl AuditSink for InMemorySink {
    fn emit(&self, event: &AuditEvent) {
        self.events
            .lock()
            .expect("audit buffer poisoned")
            .push(event.clone());
    }
}

/// Bounded backlog the webhook worker may fall behind by before events drop.
const WEBHOOK_QUEUE_CAP: usize = 256;

/// POSTs events as JSON to a configured URL on a background thread. `emit` is
/// non-blocking and drops on backpressure (a full queue) rather than stalling the
/// agent loop. No endpoint is hardcoded — the URL is supplied by config/env.
pub struct WebhookSink {
    tx: SyncSender<AuditEvent>,
}

impl WebhookSink {
    pub fn new(url: String) -> Self {
        let (tx, rx) = sync_channel::<AuditEvent>(WEBHOOK_QUEUE_CAP);
        // One worker drains the queue and POSTs serially; the loop never waits on
        // network I/O. The thread ends when the last sender (this struct) drops.
        std::thread::Builder::new()
            .name("camelid-audit-webhook".into())
            .spawn(move || {
                for event in rx {
                    post_json(&url, &event.to_json().to_string());
                }
            })
            .expect("spawn audit webhook worker");
        Self { tx }
    }
}

impl AuditSink for WebhookSink {
    fn emit(&self, event: &AuditEvent) {
        // try_send never blocks: if the worker is behind and the queue is full,
        // the event is dropped (documented backpressure behavior).
        let _ = self.tx.try_send(event.clone());
    }
}

/// Build the configured sink. Precedence: an explicit URL, else
/// `CAMELID_AUDIT_WEBHOOK`, else the no-op sink. An empty/whitespace URL counts
/// as unconfigured. Never errors when unconfigured.
pub fn sink_from_config(explicit_url: Option<&str>) -> Box<dyn AuditSink> {
    let url = explicit_url
        .map(str::to_string)
        .or_else(|| std::env::var("CAMELID_AUDIT_WEBHOOK").ok())
        .map(|u| u.trim().to_string())
        .filter(|u| !u.is_empty());
    match url {
        Some(u) => Box::new(WebhookSink::new(u)),
        None => Box::new(NoopSink),
    }
}

fn post_json(url: &str, body: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    // Reuse curl (already a runtime dependency for `pull` and `http_fetch`); body
    // is piped via stdin so it never appears in the process table.
    let mut child = match Command::new("curl")
        .args([
            "-sS",
            "--max-time",
            "5",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            url,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return, // curl missing → silently skip; audit must not crash the agent
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(body.as_bytes());
    }
    let _ = child.wait();
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_a_hash_not_the_raw_args() {
        let d = digest_args(&json!({"path": "secret.txt", "token": "hunter2"}));
        assert!(d.starts_with("sha256:"));
        assert!(!d.contains("hunter2"));
        assert!(!d.contains("secret.txt"));
        // Deterministic for equal args.
        assert_eq!(
            d,
            digest_args(&json!({"path": "secret.txt", "token": "hunter2"}))
        );
    }

    #[test]
    fn noop_sink_emits_nothing_observable() {
        let s = NoopSink;
        s.emit(&AuditEvent::call(
            "read_file",
            "auto",
            digest_args(&json!({})),
        ));
        // No panic, no state — the point is it never errors.
    }

    #[test]
    fn in_memory_sink_records_payload_shape() {
        let s = InMemorySink::default();
        s.emit(&AuditEvent::call(
            "run_shell",
            "confirm",
            "sha256:abc".into(),
        ));
        let evs = s.events();
        assert_eq!(evs.len(), 1);
        let p = evs[0].to_json();
        assert_eq!(p["event"], "agent.tool_call");
        assert_eq!(p["tool"], "run_shell");
        assert_eq!(p["approval_tier"], "confirm");
        assert_eq!(p["outcome"], Value::Null); // tool_call has no outcome yet
        assert_eq!(p["duration_ms"], Value::Null);
    }
}
