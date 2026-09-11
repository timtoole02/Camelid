//! Forwarding a placed request to the node that will serve it.
//!
//! Placement ([`super::policy`]) chooses a node; this sends the request there and
//! brings the answer back. The whole point of the fabric is that this happens
//! once per request rather than once per token, so a node's own generation loop
//! never waits on the network.
//!
//! There are two shapes of answer. [`forward`] reads a complete JSON body and is
//! what the one-shot CLI wants. [`forward_streaming`] reads only the response
//! head and then relays the body as it arrives, which is what a real client
//! asking for `stream: true` needs; it never parses the event payload, so it
//! cannot mangle a field it does not know about.

use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::http::{self, HttpError};
use super::node::NodeSpec;
use super::transport::NodeTransport;
use super::Cancel;

/// Generation can legitimately take minutes, so this is far longer than a probe
/// budget. It exists to bound a wedged node, not to bound a slow model.
pub const DEFAULT_FORWARD_TIMEOUT: Duration = Duration::from_secs(300);

/// A completion body is far larger than a health body but still bounded.
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// The node's answer, tagged with which node produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forwarded {
    pub label: String,
    /// The node's HTTP status. A node answering 503 is a real answer, not a
    /// forwarding failure, so it arrives here rather than as an error.
    pub status: u16,
    pub body: Value,
    pub elapsed: Duration,
}

impl Forwarded {
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// The engine's typed code for a request its bounded queue turned away.
///
/// This mirrors the literal `crate::api` uses when it maps its queue-full post
/// error onto the error envelope. It is duplicated rather than shared because
/// that module's bytes are pinned by the model-qualification provenance check,
/// and one constant is not worth reopening it.
///
/// Nothing in this crate can catch the two drifting apart — a test would only
/// be comparing this literal with itself. Only a real engine refusing a real
/// request can, which is what this was measured against; if the engine ever
/// renames the code, a saturated fabric silently stops handing work on.
pub const ENGINE_QUEUE_FULL_CODE: &str = "engine_queue_full";

/// An answer a node gave, which the fabric looks inside before it treats the
/// request as finished.
pub trait NodeAnswer {
    /// The node turned this request away at its queue boundary, so it never
    /// started the work.
    ///
    /// This is the same property [`ForwardError::node_never_received_it`]
    /// carries for a failure — "another node can safely be asked instead" —
    /// except that it arrives as a *successful* forward, because a node that
    /// answers 503 has answered. Nothing else about a 503 implies it: a node
    /// with no model loaded also refuses with one, and re-sending that would
    /// just spend another node's time on the same refusal.
    fn refused_for_backpressure(&self) -> bool;
}

impl NodeAnswer for Forwarded {
    fn refused_for_backpressure(&self) -> bool {
        self.status == 503
            && self.body.pointer("/error/code").and_then(Value::as_str)
                == Some(ENGINE_QUEUE_FULL_CODE)
    }
}

impl NodeAnswer for StreamOutcome {
    fn refused_for_backpressure(&self) -> bool {
        match self {
            // A refusal arrives buffered, before one event has been relayed, so
            // the request can still be placed somewhere else.
            Self::Buffered(answer) => answer.refused_for_backpressure(),
            // Once the node is streaming it has started generating, and the
            // head is already on its way to the client.
            Self::Streaming(_) => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardError {
    /// The node was never reached, so it cannot have run the request.
    ///
    /// Held apart from [`ForwardError::Transport`] because that is the whole
    /// difference between "safe to send somewhere else" and "may already be
    /// generating"; see [`ForwardError::node_never_received_it`].
    Unreachable { label: String, detail: String },
    /// The request never completed against that node, and may have reached it.
    Transport { label: String, detail: String },
    /// The node answered, but not with JSON.
    Json { label: String, detail: String },
    /// Refused before sending; see [`reject_streaming`].
    Unsupported(String),
    /// Nobody is waiting for this answer any more, so it was not finished.
    ///
    /// Held apart from every other variant because it is not a fault of the
    /// node at all: the observation that placed the request stays good, and
    /// re-placing it would only produce an answer for a caller that has gone.
    Cancelled { label: String },
}

impl ForwardError {
    /// Whether the node provably never received the request.
    ///
    /// Only a caller that has said nothing to its own client on the strength of
    /// this attempt may act on it: re-sending is safe because the node cannot
    /// have started, not because the request is idempotent.
    pub fn node_never_received_it(&self) -> bool {
        matches!(self, Self::Unreachable { .. })
    }

    /// Which node failed, when the failure is attributable to one.
    pub fn label(&self) -> Option<&str> {
        match self {
            Self::Unreachable { label, .. }
            | Self::Transport { label, .. }
            | Self::Json { label, .. }
            | Self::Cancelled { label } => Some(label),
            Self::Unsupported(_) => None,
        }
    }
}

/// Tag an HTTP failure with the node it happened against, keeping the two
/// distinctions the rest of the fabric acts on: whether the request may be
/// re-sent, and whether the node is implicated at all.
fn attribute(label: &str, error: HttpError) -> ForwardError {
    if matches!(error, HttpError::Cancelled) {
        return ForwardError::Cancelled {
            label: label.to_string(),
        };
    }
    let (label, detail) = (label.to_string(), error.to_string());
    if error.peer_never_received_it() {
        ForwardError::Unreachable { label, detail }
    } else {
        ForwardError::Transport { label, detail }
    }
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable { label, detail } => {
                write!(f, "node `{label}` could not be reached: {detail}")
            }
            Self::Transport { label, detail } => {
                write!(f, "node `{label}` did not answer: {detail}")
            }
            Self::Json { label, detail } => {
                write!(f, "node `{label}` returned an unreadable body: {detail}")
            }
            Self::Unsupported(detail) => write!(f, "{detail}"),
            Self::Cancelled { label } => write!(
                f,
                "the client that asked for this request had gone, so node `{label}` was not waited for"
            ),
        }
    }
}

impl std::error::Error for ForwardError {}

/// Refuse a streaming request on the one-shot path. Pure.
///
/// [`forward`] returns one complete JSON body, so it cannot carry a stream.
/// The resident proxy takes the [`forward_streaming`] path instead.
pub fn reject_streaming(body: &Value) -> Result<(), ForwardError> {
    if wants_streaming(body) {
        return Err(ForwardError::Unsupported(
            "`fabric run` returns one complete answer and cannot relay a streaming \
             response; drop `stream: true`, or point a client at `fabric serve`, \
             which does relay it"
                .to_string(),
        ));
    }
    Ok(())
}

/// Build a minimal OpenAI-shaped chat request. Pure.
pub fn chat_request(model: &str, prompt: &str, max_tokens: u32) -> Value {
    json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": max_tokens,
        "stream": false,
    })
}

/// Pull the assistant text out of a chat completion body. Pure.
///
/// Returns `None` rather than a placeholder so a caller can tell "the node
/// answered with no content" from "the node said nothing useful".
pub fn completion_text(body: &Value) -> Option<&str> {
    body.get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()
}

/// Read an engine error message out of a non-2xx body, if it carries one. Pure.
pub fn error_message(body: &Value) -> Option<&str> {
    body.get("error")
        .and_then(|error| error.get("message").or(Some(error)))
        .and_then(Value::as_str)
}

/// Send one request to one node.
///
/// `bearer` is required by any node started with an API key: `/v1/health` is
/// exempt from the server's auth but `/v1/chat/completions` is not, so without
/// it this is the call that comes back 401. It is only ever presented to a
/// Camelid node; see [`forward_with_transport`].
///
/// `cancel` ends the exchange and hangs up on the node, which is what stops it
/// generating; pass [`Cancel::never`] where nothing can ask for that.
pub fn forward(
    spec: &NodeSpec,
    path: &str,
    body: &Value,
    bearer: Option<&str>,
    timeout: Duration,
    cancel: &Cancel,
) -> Result<Forwarded, ForwardError> {
    forward_with_transport(
        spec,
        path,
        body,
        bearer,
        timeout,
        cancel,
        &NodeTransport::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_with_transport(
    spec: &NodeSpec,
    path: &str,
    body: &Value,
    bearer: Option<&str>,
    timeout: Duration,
    cancel: &Cancel,
    transport: &NodeTransport,
) -> Result<Forwarded, ForwardError> {
    reject_streaming(body)?;
    // Decided here rather than by each caller: every placed request leaves
    // through this function or the streaming one below, and these are the
    // last points that still know which engine the node runs. A caller that
    // forgot — or a token that arrived from `CAMELID_API_KEY` rather than a
    // flag — cannot reach a foreign engine past this line.
    let bearer = spec.engine.fabric_bearer(bearer);

    let encoded = serde_json::to_vec(body).map_err(|error| ForwardError::Json {
        label: spec.label.clone(),
        detail: format!("request body could not be encoded: {error}"),
    })?;

    let started = Instant::now();
    let response = http::request_with_transport(
        &spec.host,
        spec.port,
        "POST",
        path,
        Some(&encoded),
        bearer,
        timeout,
        MAX_RESPONSE_BYTES,
        cancel,
        transport,
    )
    .map_err(|error| attribute(&spec.label, error))?;
    let elapsed = started.elapsed();

    // An empty body is legal for some statuses; represent it as null rather than
    // failing, so the status still reaches the caller.
    let parsed = if response.body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice::<Value>(&response.body).map_err(|error| ForwardError::Json {
            label: spec.label.clone(),
            detail: error.to_string(),
        })?
    };

    Ok(Forwarded {
        label: spec.label.clone(),
        status: response.status,
        body: parsed,
        elapsed,
    })
}

/// Whether the client asked for a server-sent event stream. Pure.
pub fn wants_streaming(body: &Value) -> bool {
    body.get("stream").and_then(Value::as_bool) == Some(true)
}

/// A node's answer to a streaming request, still arriving.
///
/// The payload is relayed verbatim. Nothing here parses server-sent events, so
/// a field this fabric has never heard of reaches the client unaltered.
pub struct Streaming {
    pub label: String,
    pub status: u16,
    /// The node's `Content-Type`, so the client is told what it is reading.
    pub content_type: Option<String>,
    stream: http::ResponseStream,
    started: Instant,
}

impl std::fmt::Debug for Streaming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Streaming")
            .field("label", &self.label)
            .field("status", &self.status)
            .field("content_type", &self.content_type)
            .finish_non_exhaustive()
    }
}

impl Streaming {
    /// The next piece of the body, or `None` at the end of the stream.
    ///
    /// Never `Unreachable`: the node answered to get here, so the request
    /// cannot be sent anywhere else. A failure to read is therefore `Transport`
    /// whatever caused it — except a cancellation, which says nothing about the
    /// node and must not be recorded against it.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, ForwardError> {
        self.stream
            .next_chunk()
            .map_err(|error| after_the_head(&self.label, error))
    }

    /// Time from starting the request through the latest body read.
    pub(crate) fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// Classify a failure raised once a node has already answered with a head.
fn after_the_head(label: &str, error: HttpError) -> ForwardError {
    match error {
        HttpError::Cancelled => ForwardError::Cancelled {
            label: label.to_string(),
        },
        error => ForwardError::Transport {
            label: label.to_string(),
            detail: error.to_string(),
        },
    }
}

/// What a node did with a streaming request.
#[derive(Debug)]
pub enum StreamOutcome {
    /// The node is streaming; relay it.
    Streaming(Streaming),
    /// The node answered with a complete body instead — typically a refusal.
    /// Buffered so the reason reaches the client as a readable answer rather
    /// than as an empty stream.
    Buffered(Forwarded),
}

/// Send a streaming request to one node and read as far as its response head.
///
/// `head_timeout` bounds the wait for that head, which covers the node's whole
/// prefill and can legitimately take minutes. `idle_timeout` then bounds how
/// long the node may send *nothing further*: a healthy stream resets it with
/// every token, so a long generation is never cut short, while a wedged node
/// still fails on schedule.
pub fn forward_streaming(
    spec: &NodeSpec,
    path: &str,
    body: &Value,
    bearer: Option<&str>,
    head_timeout: Duration,
    idle_timeout: Duration,
    cancel: &Cancel,
) -> Result<StreamOutcome, ForwardError> {
    forward_streaming_with_transport(
        spec,
        path,
        body,
        bearer,
        head_timeout,
        idle_timeout,
        cancel,
        &NodeTransport::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_streaming_with_transport(
    spec: &NodeSpec,
    path: &str,
    body: &Value,
    bearer: Option<&str>,
    head_timeout: Duration,
    idle_timeout: Duration,
    cancel: &Cancel,
    transport: &NodeTransport,
) -> Result<StreamOutcome, ForwardError> {
    // See `forward_with_transport`: the same gate, for the same reason.
    let bearer = spec.engine.fabric_bearer(bearer);

    let encoded = serde_json::to_vec(body).map_err(|error| ForwardError::Json {
        label: spec.label.clone(),
        detail: format!("request body could not be encoded: {error}"),
    })?;

    let started = Instant::now();
    let stream = http::open_stream_with_transport(
        &spec.host,
        spec.port,
        "POST",
        path,
        Some(&encoded),
        bearer,
        head_timeout,
        idle_timeout,
        MAX_RESPONSE_BYTES,
        cancel,
        transport,
    )
    .map_err(|error| attribute(&spec.label, error))?;

    let head = stream.head().clone();
    // A node that refused, or answered with something other than a stream, has
    // said something worth reading. Buffer it and hand it back as an answer.
    if !(200..300).contains(&head.status) || !head.is_event_stream() {
        let response = stream
            .into_buffered(MAX_RESPONSE_BYTES)
            .map_err(|error| after_the_head(&spec.label, error))?;
        let parsed = if response.body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice::<Value>(&response.body).map_err(|error| ForwardError::Json {
                label: spec.label.clone(),
                detail: error.to_string(),
            })?
        };
        return Ok(StreamOutcome::Buffered(Forwarded {
            label: spec.label.clone(),
            status: response.status,
            body: parsed,
            elapsed: started.elapsed(),
        }));
    }

    Ok(StreamOutcome::Streaming(Streaming {
        label: spec.label.clone(),
        status: head.status,
        content_type: head.content_type,
        stream,
        started,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(label: &str, port: u16) -> NodeSpec {
        NodeSpec::camelid(label, "127.0.0.1", port)
    }

    fn answered(status: u16, body: Value) -> Forwarded {
        Forwarded {
            label: "node-a".to_string(),
            status,
            body,
            elapsed: Duration::from_millis(1),
        }
    }

    /// The exact envelope a real engine sends when its bounded queue turns a
    /// request away: a 503 whose typed code names the queue, not the model.
    #[test]
    fn a_queue_full_refusal_is_backpressure() {
        let answer = answered(
            503,
            serde_json::json!({
                "error": {
                    "message": "the generation queue is full; retry shortly",
                    "type": "runtime_unavailable",
                    "code": ENGINE_QUEUE_FULL_CODE,
                }
            }),
        );
        assert!(answer.refused_for_backpressure());
    }

    /// The status alone must not decide it. A node with no model loaded also
    /// answers 503, and it will answer the same way however many times it is
    /// asked — re-sending that request costs a second node its time and turns
    /// one refusal into two.
    #[test]
    fn another_kind_of_503_is_not_backpressure() {
        let answer = answered(
            503,
            serde_json::json!({
                "error": {
                    "message": "no model is loaded",
                    "type": "runtime_unavailable",
                    "code": "model_unavailable",
                }
            }),
        );
        assert!(!answer.refused_for_backpressure());
    }

    #[test]
    fn a_body_without_an_error_is_not_backpressure() {
        assert!(!answered(503, Value::Null).refused_for_backpressure());
        assert!(!answered(503, serde_json::json!({ "error": "full" })).refused_for_backpressure());
        assert!(!answered(503, serde_json::json!({})).refused_for_backpressure());
    }

    /// A served answer is never re-placed, whatever it happens to contain.
    #[test]
    fn a_successful_answer_is_not_backpressure() {
        let answer = answered(
            200,
            serde_json::json!({ "error": { "code": ENGINE_QUEUE_FULL_CODE } }),
        );
        assert!(!answer.refused_for_backpressure());
    }

    /// A streaming request refused this way never became a stream: the node
    /// answered with one body, so nothing has been relayed and the request can
    /// still go elsewhere.
    #[test]
    fn a_buffered_refusal_on_a_streaming_request_is_backpressure() {
        let outcome = StreamOutcome::Buffered(answered(
            503,
            serde_json::json!({ "error": { "code": ENGINE_QUEUE_FULL_CODE } }),
        ));
        assert!(outcome.refused_for_backpressure());

        let served = StreamOutcome::Buffered(answered(200, serde_json::json!({ "choices": [] })));
        assert!(!served.refused_for_backpressure());
    }

    #[test]
    fn a_chat_request_carries_the_model_prompt_and_no_streaming() {
        let request = chat_request("llama-3b", "hello", 16);
        assert_eq!(request["model"], "llama-3b");
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"], "hello");
        assert_eq!(request["max_tokens"], 16);
        assert_eq!(request["stream"], false);
    }

    #[test]
    fn a_streaming_request_is_refused_before_a_socket_is_opened() {
        let body = json!({ "model": "m", "stream": true });
        let error = reject_streaming(&body).expect_err("streaming is unsupported");
        assert!(matches!(error, ForwardError::Unsupported(_)));

        // Port 1 is closed; a transport error here would prove we tried to send.
        let error = forward(
            &spec("a", 1),
            "/v1/chat/completions",
            &body,
            None,
            Duration::from_millis(200),
            &Cancel::never(),
        )
        .expect_err("refused");
        assert!(
            matches!(error, ForwardError::Unsupported(_)),
            "must refuse before dialling, got {error:?}"
        );
    }

    #[test]
    fn a_non_streaming_request_is_allowed_through() {
        assert!(reject_streaming(&json!({ "model": "m" })).is_ok());
        assert!(reject_streaming(&json!({ "model": "m", "stream": false })).is_ok());
    }

    #[test]
    fn completion_text_reads_the_first_choice() {
        let body = json!({
            "choices": [{ "message": { "role": "assistant", "content": "hi there" } }]
        });
        assert_eq!(completion_text(&body), Some("hi there"));
    }

    #[test]
    fn completion_text_is_none_rather_than_a_placeholder_when_absent() {
        assert_eq!(completion_text(&json!({})), None);
        assert_eq!(completion_text(&json!({ "choices": [] })), None);
        assert_eq!(completion_text(&json!({ "choices": [{}] })), None);
    }

    #[test]
    fn an_engine_error_message_is_extracted_for_reporting() {
        let body = json!({ "error": { "message": "engine queue full" } });
        assert_eq!(error_message(&body), Some("engine queue full"));

        let flat = json!({ "error": "bad request" });
        assert_eq!(error_message(&flat), Some("bad request"));
    }

    #[test]
    fn a_dead_node_reports_which_node_failed() {
        let error = forward(
            &spec("windows", 1),
            "/v1/chat/completions",
            &chat_request("m", "hi", 4),
            None,
            Duration::from_millis(400),
            &Cancel::never(),
        )
        .expect_err("port 1 is closed");
        match &error {
            ForwardError::Unreachable { label, .. } => assert_eq!(label, "windows"),
            other => panic!("expected Unreachable, got {other:?}"),
        }
        // The message must name the node, or an operator cannot act on it.
        assert!(error.to_string().contains("windows"), "{error}");
        assert_eq!(error.label(), Some("windows"));
    }

    #[test]
    fn a_node_that_was_never_dialled_is_distinguished_from_one_that_was() {
        // The distinction the whole retry path rests on, taken from the real
        // dialling code rather than a hand-built variant.
        let unreached = forward(
            &spec("gone", 1),
            "/v1/chat/completions",
            &chat_request("m", "hi", 4),
            None,
            Duration::from_millis(400),
            &Cancel::never(),
        )
        .expect_err("port 1 is closed");
        assert!(unreached.node_never_received_it(), "{unreached:?}");

        let interrupted = ForwardError::Transport {
            label: "gone".to_string(),
            detail: "connection reset".to_string(),
        };
        assert!(!interrupted.node_never_received_it());

        // A refusal that never named a node cannot be blamed on one either.
        let refused = reject_streaming(&json!({ "model": "m", "stream": true }))
            .expect_err("streaming is unsupported");
        assert!(!refused.node_never_received_it());
        assert_eq!(refused.label(), None);
    }

    #[test]
    fn a_streaming_open_against_a_closed_port_is_also_never_received() {
        let error = forward_streaming(
            &spec("gone", 1),
            "/v1/chat/completions",
            &json!({ "model": "m", "stream": true }),
            None,
            Duration::from_millis(400),
            Duration::from_millis(400),
            &Cancel::never(),
        )
        .expect_err("port 1 is closed");
        assert!(error.node_never_received_it(), "{error:?}");
    }

    /// A client leaving is not the node's doing, and the difference is acted on
    /// twice over: the request is not placed again, and the observation that
    /// chose the node is kept rather than thrown away as proved wrong.
    #[test]
    fn a_request_its_client_gave_up_on_is_not_recorded_against_the_node() {
        let cancel = Cancel::new();
        cancel.cancel();

        let error = forward(
            &spec("windows", 1),
            "/v1/chat/completions",
            &chat_request("m", "hi", 4),
            None,
            Duration::from_millis(400),
            &cancel,
        )
        .expect_err("the client had gone");

        assert!(
            matches!(error, ForwardError::Cancelled { .. }),
            "a client hanging up must not read as a node failure, got {error:?}"
        );
        assert!(
            !error.node_never_received_it(),
            "re-placing it would answer a client that has gone"
        );
        // Named for the same reason every other failure here is: it says which
        // node just got its generation slot back.
        assert_eq!(error.label(), Some("windows"));
        assert!(error.to_string().contains("windows"), "{error}");
    }

    #[test]
    fn a_success_status_is_distinguished_from_an_engine_refusal() {
        let ok = Forwarded {
            label: "a".to_string(),
            status: 200,
            body: Value::Null,
            elapsed: Duration::ZERO,
        };
        let refused = Forwarded {
            status: 503,
            ..ok.clone()
        };
        assert!(ok.is_success());
        assert!(!refused.is_success());
    }
}
