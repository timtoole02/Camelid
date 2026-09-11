//! End-to-end fabric behaviour against stub nodes.
//!
//! The two-machine runs that validated this feature needed real engines, real
//! models and a real network, so they cannot run in CI. These tests stand in
//! two HTTP nodes on loopback that answer `/v1/health` and
//! `/v1/chat/completions` from canned data, which makes the same paths — model
//! scoped placement, affinity, forwarding, failure — reproducible anywhere.
//!
//! They also assert what the fabric *sends*, not only what it does with the
//! reply. Nothing else covers the request side.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use camelid::fabric::{
    forward, parse_node_spec, route, Cancel, DispatchError, Fabric, ForwardError, NodeSpec,
    RouteError, RouteMode, RouteReason, RouteRequest,
};

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const FORWARD_TIMEOUT: Duration = Duration::from_secs(5);

/// One request the stub actually received.
#[derive(Debug, Clone)]
struct Received {
    method: String,
    path: String,
    body: String,
    /// Header names lowercased; values verbatim.
    headers: Vec<(String, String)>,
}

impl Received {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, value)| value.as_str())
    }
}

#[derive(Clone)]
struct StubConfig {
    /// Body served from `/v1/health`.
    health: String,
    /// Body served from `/v1/chat/completions`.
    completion: String,
    completion_status: u16,
    /// When set, every route but `/v1/health` answers 401 unless the request
    /// carries exactly this bearer token — the arrangement a node started with
    /// `CAMELID_API_KEY` actually presents to the fabric.
    required_key: Option<String>,
}

impl StubConfig {
    fn ready(model: &str, in_flight: usize) -> Self {
        Self {
            health: format!(
                r#"{{"ok":true,"generation_ready":true,"active_model_id":"{model}",
                    "backend":"llama","version":"0.5.4",
                    "engine_queued_tasks":0,"engine_queue_depth":{in_flight}}}"#
            ),
            completion: format!(
                r#"{{"choices":[{{"message":{{"role":"assistant","content":"served by {model}"}}}}]}}"#
            ),
            completion_status: 200,
            required_key: None,
        }
    }

    fn not_ready() -> Self {
        Self {
            health: r#"{"ok":true,"generation_ready":false,"active_model_id":null}"#.to_string(),
            completion: "{}".to_string(),
            completion_status: 200,
            required_key: None,
        }
    }

    fn refusing(model: &str) -> Self {
        Self {
            completion: r#"{"error":{"message":"engine queue full"}}"#.to_string(),
            completion_status: 503,
            ..Self::ready(model, 0)
        }
    }

    /// A node with an API key set: health stays open, generation does not.
    fn requiring_key(model: &str, key: &str) -> Self {
        Self {
            required_key: Some(key.to_string()),
            ..Self::ready(model, 0)
        }
    }
}

/// A stand-in Camelid node on loopback.
struct StubNode {
    port: u16,
    shutdown: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Received>>>,
    thread: Option<JoinHandle<()>>,
}

impl StubNode {
    fn start(config: StubConfig) -> Self {
        // Port 0 lets the OS pick, so tests never collide with each other or
        // with a real engine on 8181.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_requests = Arc::clone(&requests);
        let thread = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if thread_shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut stream) = stream else { continue };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                serve_once(&mut stream, &config, &thread_requests);
            }
        });

        Self {
            port,
            shutdown,
            requests,
            thread: Some(thread),
        }
    }

    fn spec(&self, label: &str) -> NodeSpec {
        NodeSpec::camelid(label, "127.0.0.1", self.port)
    }

    fn received(&self) -> Vec<Received> {
        self.requests.lock().expect("stub lock").clone()
    }
}

impl Drop for StubNode {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Unblock the accept() the worker is parked in.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The same stub node over server-authenticated TLS.
struct TlsStubNode {
    port: u16,
    shutdown: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Received>>>,
    thread: Option<JoinHandle<()>>,
    _directory: tempfile::TempDir,
    ca_path: std::path::PathBuf,
}

impl TlsStubNode {
    fn start(config: StubConfig) -> Self {
        let issued = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .expect("generate a node certificate");
        let key = rustls_pki_types::PrivatePkcs8KeyDer::from(issued.key_pair.serialize_der());
        let server = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![issued.cert.der().clone()], key.into())
            .expect("node certificate and key agree");
        let directory = tempfile::tempdir().expect("temp dir");
        let ca_path = directory.path().join("node-ca");
        std::fs::write(&ca_path, rcgen::Certificate::pem(&issued.cert)).expect("write node CA");

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind TLS node");
        let port = listener.local_addr().expect("local addr").port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_shutdown = Arc::clone(&shutdown);
        let thread_requests = Arc::clone(&requests);
        let server = Arc::new(server);
        let thread = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if thread_shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
                let connection = rustls::ServerConnection::new(Arc::clone(&server))
                    .expect("create TLS server connection");
                let mut stream = rustls::StreamOwned::new(connection, stream);
                serve_once(&mut stream, &config, &thread_requests);
            }
        });

        Self {
            port,
            shutdown,
            requests,
            thread: Some(thread),
            _directory: directory,
            ca_path,
        }
    }

    fn spec(&self, label: &str) -> NodeSpec {
        NodeSpec::camelid(label, "localhost", self.port)
    }

    fn received(&self) -> Vec<Received> {
        self.requests.lock().expect("TLS stub lock").clone()
    }
}

impl Drop for TlsStubNode {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(("127.0.0.1", self.port));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn serve_once(
    stream: &mut (impl Read + Write),
    config: &StubConfig,
    requests: &Mutex<Vec<Received>>,
) {
    let Some(received) = read_request(stream) else {
        return;
    };

    // Mirrors `route_requires_auth` in the real server: `/v1/health` is exempt,
    // everything else is gated.
    let authorized = match &config.required_key {
        Some(key) => {
            received.path == "/v1/health"
                || received.header("authorization") == Some(&format!("Bearer {key}"))
        }
        None => true,
    };

    let (status, body) = match received.path.as_str() {
        _ if !authorized => (
            401,
            r#"{"error":{"message":"provide Authorization: Bearer <key> or X-API-Key"}}"#
                .to_string(),
        ),
        "/v1/health" => (200_u16, config.health.clone()),
        "/v1/chat/completions" => (config.completion_status, config.completion.clone()),
        _ => (404, "{}".to_string()),
    };

    // Record only after deciding, so a malformed request never poisons the log.
    requests.lock().expect("stub lock").push(received);

    let response = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn read_request(stream: &mut impl Read) -> Option<Received> {
    let mut raw = Vec::new();
    let mut chunk = [0_u8; 1024];
    // Read until the headers are complete, then until Content-Length is met.
    loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&chunk[..read]);

        let Some(header_end) = find_header_end(&raw) else {
            continue;
        };
        let head = String::from_utf8_lossy(&raw[..header_end]).to_string();
        let expected = content_length(&head).unwrap_or(0);
        if raw.len() >= header_end + 4 + expected {
            let body = String::from_utf8_lossy(&raw[header_end + 4..]).to_string();
            let mut parts = head.lines().next()?.split_whitespace();
            return Some(Received {
                method: parts.next()?.to_string(),
                path: parts.next()?.to_string(),
                body,
                headers: parse_headers(&head),
            });
        }
    }
    None
}

fn find_header_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Every header line after the request line, names lowercased for lookup.
fn parse_headers(head: &str) -> Vec<(String, String)> {
    head.lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect()
}

fn content_length(head: &str) -> Option<usize> {
    head.lines()
        .find(|line| line.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|line| line.split(':').nth(1))
        .and_then(|value| value.trim().parse().ok())
}

/// A port that nothing listens on.
fn dead_spec(label: &str) -> NodeSpec {
    NodeSpec::camelid(label, "127.0.0.1", 9)
}

fn fabric_of(specs: Vec<NodeSpec>) -> Fabric {
    Fabric::new(specs).with_timeout(PROBE_TIMEOUT)
}

#[test]
fn a_two_node_fabric_reports_both_models() {
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let beta = StubNode::start(StubConfig::ready("model-beta", 0));
    let fabric = fabric_of(vec![alpha.spec("alpha"), beta.spec("beta")]);

    let snapshots = fabric.observe();
    assert_eq!(snapshots.len(), 2);
    assert_eq!(snapshots[0].active_model_id(), Some("model-alpha"));
    assert_eq!(snapshots[1].active_model_id(), Some("model-beta"));
}

#[test]
fn placement_is_scoped_to_the_node_actually_serving_the_model() {
    // The live two-machine run proved this; the stubs make it reproducible.
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let beta = StubNode::start(StubConfig::ready("model-beta", 0));
    let fabric = fabric_of(vec![alpha.spec("alpha"), beta.spec("beta")]);
    let snapshots = fabric.observe();

    let to_alpha = route(
        &snapshots,
        &RouteRequest::new(RouteMode::Throughput).with_model(Some("model-alpha")),
    )
    .expect("alpha serves it");
    assert_eq!(to_alpha.label, "alpha");

    let to_beta = route(
        &snapshots,
        &RouteRequest::new(RouteMode::Throughput).with_model(Some("model-beta")),
    )
    .expect("beta serves it");
    assert_eq!(to_beta.label, "beta");
}

#[test]
fn an_unserved_model_is_refused_naming_what_the_fabric_does_serve() {
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let fabric = fabric_of(vec![alpha.spec("alpha")]);

    let error = route(
        &fabric.observe(),
        &RouteRequest::new(RouteMode::Throughput).with_model(Some("model-absent")),
    )
    .expect_err("nothing serves it");

    match error {
        RouteError::ModelUnavailable {
            model,
            serving,
            unobserved,
        } => {
            assert_eq!(model, "model-absent");
            assert_eq!(serving, vec!["model-alpha".to_string()]);
            // Every configured node answered, so this refusal is final.
            assert_eq!(unobserved, 0);
        }
        other => panic!("expected ModelUnavailable, got {other:?}"),
    }
}

#[test]
fn a_node_that_is_not_ready_is_never_placed_on() {
    let warming = StubNode::start(StubConfig::not_ready());
    let ready = StubNode::start(StubConfig::ready("model-alpha", 0));
    let fabric = fabric_of(vec![warming.spec("warming"), ready.spec("ready")]);

    let decision = route(&fabric.observe(), &RouteRequest::new(RouteMode::Throughput))
        .expect("one node can serve");
    assert_eq!(decision.label, "ready");
    assert_eq!(decision.reason, RouteReason::OnlyCandidate);
}

#[test]
fn the_least_loaded_node_wins_and_a_dead_peer_does_not_stall_the_view() {
    let busy = StubNode::start(StubConfig::ready("model-alpha", 5));
    let idle = StubNode::start(StubConfig::ready("model-alpha", 0));
    let fabric = fabric_of(vec![
        dead_spec("dead"),
        busy.spec("busy"),
        idle.spec("idle"),
    ]);

    let snapshots = fabric.observe();
    assert_eq!(snapshots.len(), 3, "every node is reported, dead included");

    let decision =
        route(&snapshots, &RouteRequest::new(RouteMode::Throughput)).expect("two nodes can serve");
    assert_eq!(decision.label, "idle");
    assert_eq!(decision.reason, RouteReason::LeastLoaded);
}

#[test]
fn affinity_holds_on_a_live_node_and_degrades_on_a_dead_one() {
    let warm = StubNode::start(StubConfig::ready("model-alpha", 3));
    let cold = StubNode::start(StubConfig::ready("model-alpha", 0));
    let fabric = fabric_of(vec![warm.spec("warm"), cold.spec("cold")]);
    let snapshots = fabric.observe();

    let held = route(
        &snapshots,
        &RouteRequest::new(RouteMode::Affinity).with_sticky(Some("warm")),
    )
    .expect("warm is alive");
    assert_eq!(held.label, "warm");
    assert_eq!(held.reason, RouteReason::Affinity);
    assert_eq!(held.affinity_lost, None);

    // Same fabric, but the session's node is not in it at all.
    let degraded = route(
        &snapshots,
        &RouteRequest::new(RouteMode::Affinity).with_sticky(Some("vanished")),
    )
    .expect("degrades rather than failing");
    assert_eq!(degraded.label, "cold");
    assert_eq!(degraded.affinity_lost.as_deref(), Some("vanished"));
}

#[test]
fn forwarding_sends_a_well_formed_request_and_returns_the_answer() {
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let spec = alpha.spec("alpha");

    let body = forward::chat_request("model-alpha", "ping", 8);
    let answer = forward::forward(
        &spec,
        "/v1/chat/completions",
        &body,
        None,
        FORWARD_TIMEOUT,
        &Cancel::never(),
    )
    .expect("stub answers");

    assert!(answer.is_success());
    assert_eq!(answer.label, "alpha");
    assert_eq!(
        forward::completion_text(&answer.body),
        Some("served by model-alpha")
    );

    // The request side: a health probe, then the completion carrying our body.
    let seen = alpha.received();
    let completion = seen
        .iter()
        .find(|request| request.path == "/v1/chat/completions")
        .expect("the completion reached the node");
    assert_eq!(completion.method, "POST");

    let sent: serde_json::Value =
        serde_json::from_str(&completion.body).expect("we send valid JSON");
    assert_eq!(sent["model"], "model-alpha");
    assert_eq!(sent["messages"][0]["content"], "ping");
    assert_eq!(sent["stream"], false);
}

#[test]
fn an_engine_refusal_is_the_nodes_answer_not_a_transport_failure() {
    let refusing = StubNode::start(StubConfig::refusing("model-alpha"));
    let spec = refusing.spec("refusing");

    let answer = forward::forward(
        &spec,
        "/v1/chat/completions",
        &forward::chat_request("model-alpha", "ping", 8),
        None,
        FORWARD_TIMEOUT,
        &Cancel::never(),
    )
    .expect("a 503 is an answer, not an error");

    assert!(!answer.is_success());
    assert_eq!(answer.status, 503);
    assert_eq!(
        forward::error_message(&answer.body),
        Some("engine queue full")
    );
}

#[test]
fn dispatch_places_and_sends_in_one_call() {
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 4));
    let beta = StubNode::start(StubConfig::ready("model-beta", 0));
    let fabric = fabric_of(vec![alpha.spec("alpha"), beta.spec("beta")]);

    let dispatched = fabric
        .dispatch(
            "/v1/chat/completions",
            &forward::chat_request("model-beta", "ping", 8),
            &RouteRequest::new(RouteMode::Throughput).with_model(Some("model-beta")),
            FORWARD_TIMEOUT,
            &Cancel::never(),
        )
        .expect("beta serves model-beta");

    assert_eq!(dispatched.decision.label, "beta");
    assert_eq!(dispatched.answer.label, "beta");
    assert_eq!(
        dispatched.attempts, 1,
        "a node that answers first time is one attempt"
    );
    assert_eq!(
        forward::completion_text(&dispatched.answer.body),
        Some("served by model-beta")
    );
    // The request must not have touched the node that does not hold the model.
    assert!(
        alpha
            .received()
            .iter()
            .all(|request| request.path == "/v1/health"),
        "alpha should only have been probed, never sent the completion"
    );
}

#[test]
fn dispatch_refuses_streaming_before_touching_the_network() {
    // Every node is dead: if dispatch probed first this would be a routing
    // failure, so seeing the streaming refusal proves the order.
    let fabric = fabric_of(vec![dead_spec("dead")]);
    let error = fabric
        .dispatch(
            "/v1/chat/completions",
            &serde_json::json!({ "model": "m", "stream": true }),
            &RouteRequest::new(RouteMode::Throughput),
            Duration::from_millis(300),
            &Cancel::never(),
        )
        .expect_err("streaming is unsupported");
    assert!(
        error.to_string().contains("streaming"),
        "unexpected: {error}"
    );
}

#[test]
fn a_fabric_carrying_the_key_is_served_by_an_authenticated_node() {
    let alpha = StubNode::start(StubConfig::requiring_key("model-alpha", "s3cret"));
    let fabric = fabric_of(vec![alpha.spec("alpha")]).with_bearer(Some("s3cret"));

    let dispatched = fabric
        .dispatch(
            "/v1/chat/completions",
            &forward::chat_request("model-alpha", "ping", 8),
            &RouteRequest::new(RouteMode::Throughput),
            FORWARD_TIMEOUT,
            &Cancel::never(),
        )
        .expect("the node accepts our key");

    assert_eq!(dispatched.decision.label, "alpha");
    assert_eq!(dispatched.answer.status, 200);
    assert_eq!(
        forward::completion_text(&dispatched.answer.body),
        Some("served by model-alpha")
    );

    // Every request carried the token, the probe included. Health is exempt from
    // the server's auth today, so sending it there buys nothing now — it means
    // placement keeps working if that exemption is ever tightened.
    let seen = alpha.received();
    assert!(
        seen.iter().any(|request| request.path == "/v1/health"),
        "the node was probed as well as sent to"
    );
    for request in &seen {
        assert_eq!(
            request.header("authorization"),
            Some("Bearer s3cret"),
            "`{}` went out unauthenticated",
            request.path
        );
    }
}

#[test]
fn a_fabric_observes_and_dispatches_through_one_ca_pinned_tls_policy() {
    let alpha = TlsStubNode::start(StubConfig::requiring_key("model-alpha", "s3cret"));
    let fabric = fabric_of(vec![alpha.spec("alpha")])
        .with_bearer(Some("s3cret"))
        .with_node_transport(Some(&alpha.ca_path), false)
        .expect("node TLS policy resolves");

    let snapshots = fabric.observe();
    assert_eq!(snapshots[0].active_model_id(), Some("model-alpha"));

    let dispatched = fabric
        .dispatch(
            "/v1/chat/completions",
            &forward::chat_request("model-alpha", "ping", 8),
            &RouteRequest::new(RouteMode::Throughput),
            FORWARD_TIMEOUT,
            &Cancel::never(),
        )
        .expect("the TLS node serves the request");
    assert_eq!(dispatched.decision.label, "alpha");
    assert_eq!(dispatched.answer.status, 200);
    assert_eq!(
        forward::completion_text(&dispatched.answer.body),
        Some("served by model-alpha")
    );

    let seen = alpha.received();
    assert!(seen.iter().any(|request| request.path == "/v1/health"));
    assert!(seen
        .iter()
        .any(|request| request.path == "/v1/chat/completions"));
    assert!(seen
        .iter()
        .all(|request| { request.header("authorization") == Some("Bearer s3cret") }));
}

#[test]
fn a_node_loaded_later_from_the_file_inherits_the_tls_policy() {
    let original = TlsStubNode::start(StubConfig::ready("model-alpha", 0));
    let joined = TlsStubNode::start(StubConfig::ready("model-alpha", 0));
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("nodes");
    std::fs::write(&path, format!("original=localhost:{}\n", original.port))
        .expect("write initial node file");
    let ca_path = directory.path().join("node-ca-bundle");
    let ca_bundle = format!(
        "{}\n{}",
        std::fs::read_to_string(&original.ca_path).expect("read original CA"),
        std::fs::read_to_string(&joined.ca_path).expect("read joined CA")
    );
    std::fs::write(&ca_path, ca_bundle).expect("write shared CA bundle");

    let fabric = Fabric::from_node_file(path.clone())
        .expect("load node file")
        .with_timeout(PROBE_TIMEOUT)
        .with_node_transport(Some(&ca_path), false)
        .expect("node TLS policy resolves");
    let initial = fabric.observe();
    assert_eq!(initial.len(), 1);
    assert_eq!(initial[0].label(), "original");
    assert!(initial[0].status.is_ready());

    // The shipped watcher checks once per second. Rewrite after that bound so
    // this test exercises a newly loaded spec rather than the startup value.
    std::thread::sleep(Duration::from_millis(1_100));
    std::fs::write(&path, format!("joined-later=localhost:{}\n", joined.port))
        .expect("replace node file with a distinct endpoint");

    let reloaded = fabric.observe();
    assert_eq!(reloaded.len(), 1);
    assert_eq!(reloaded[0].label(), "joined-later");
    assert!(reloaded[0].status.is_ready());
    assert_eq!(
        original
            .received()
            .iter()
            .filter(|request| request.path == "/v1/health")
            .count(),
        1,
        "the original endpoint must not be reused after reload"
    );
    assert_eq!(
        joined
            .received()
            .iter()
            .filter(|request| request.path == "/v1/health")
            .count(),
        1,
        "the newly loaded endpoint must cross the existing TLS policy"
    );
}

#[test]
fn a_missing_or_wrong_key_arrives_as_a_401_answer_not_a_transport_failure() {
    // The defect this flag exists for. `/v1/health` is auth-exempt, so an
    // authenticated node observes as ready and places fine; only the forward is
    // refused. That refusal is the node answering, so it must reach the caller
    // with its status rather than as a `Transport` error naming a dead node.
    let alpha = StubNode::start(StubConfig::requiring_key("model-alpha", "s3cret"));

    let unauthenticated = fabric_of(vec![alpha.spec("alpha")]);
    let snapshots = unauthenticated.observe();
    assert_eq!(
        snapshots[0].active_model_id(),
        Some("model-alpha"),
        "health is exempt, so status still reads the node as ready"
    );
    assert_eq!(
        route(&snapshots, &RouteRequest::new(RouteMode::Throughput))
            .expect("placement succeeds even though the request will not")
            .label,
        "alpha"
    );

    // A wrong key is refused exactly as plainly as no key — and proves the stub
    // compares the token rather than merely noticing one is present.
    for fabric in [
        unauthenticated,
        fabric_of(vec![alpha.spec("alpha")]).with_bearer(Some("wrong-key")),
    ] {
        let (_, answer) = fabric
            .dispatch(
                "/v1/chat/completions",
                &forward::chat_request("model-alpha", "ping", 8),
                &RouteRequest::new(RouteMode::Throughput),
                FORWARD_TIMEOUT,
                &Cancel::never(),
            )
            .map(|dispatched| (dispatched.decision, dispatched.answer))
            .expect("a 401 is the node's answer, not a forwarding failure");

        assert_eq!(answer.status, 401);
        assert!(!answer.is_success());
        assert!(
            forward::error_message(&answer.body).is_some(),
            "the refusal must carry a message an operator can act on"
        );
    }
}

#[test]
fn an_unauthenticated_fabric_sends_no_authorization_header_at_all() {
    // An open node must see exactly the request it saw before the fabric learned
    // about tokens — not a header with an empty credential.
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let fabric = fabric_of(vec![alpha.spec("alpha")]);

    fabric
        .dispatch(
            "/v1/chat/completions",
            &forward::chat_request("model-alpha", "ping", 8),
            &RouteRequest::new(RouteMode::Throughput),
            FORWARD_TIMEOUT,
            &Cancel::never(),
        )
        .expect("an open node answers");

    let seen = alpha.received();
    assert!(!seen.is_empty(), "the node was reached");
    assert!(
        seen.iter()
            .all(|request| request.header("authorization").is_none()),
        "an unconfigured fabric must not send an Authorization header"
    );
}

#[test]
fn an_operator_node_string_drives_a_real_placement() {
    // Proves the parsed CLI form reaches a live node, not just the struct form.
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let spec = parse_node_spec(&format!("alpha=127.0.0.1:{}", alpha.port)).expect("parses");

    let fabric = fabric_of(vec![spec]);
    let decision = route(&fabric.observe(), &RouteRequest::new(RouteMode::Throughput))
        .expect("the parsed node serves");
    assert_eq!(decision.label, "alpha");
}

/// Health probes a node has answered. A freshness bound exists so that this
/// stops growing with the number of client requests.
fn health_probes(node: &StubNode) -> usize {
    node.received()
        .iter()
        .filter(|received| received.path == "/v1/health")
        .count()
}

/// A client hanging up says nothing about the node its request was placed on.
///
/// The fabric drops an observation that has proved wrong, which is what keeps
/// a freshness bound honest. Counting a cancellation as proof would invert
/// that: on a fabric working perfectly, every client that leaves would make the
/// *next* request pay a whole probe round.
#[test]
fn a_cancelled_request_leaves_the_observation_that_placed_it_in_force() {
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let fabric =
        fabric_of(vec![alpha.spec("alpha")]).with_max_observation_age(Duration::from_secs(30));
    let body = forward::chat_request("model-alpha", "ping", 8);
    let request = RouteRequest::new(RouteMode::Throughput);

    fabric
        .dispatch(
            "/v1/chat/completions",
            &body,
            &request,
            FORWARD_TIMEOUT,
            &Cancel::never(),
        )
        .expect("the node serves");
    let after_the_first = health_probes(&alpha);
    assert!(after_the_first > 0, "the fabric observed the node at all");

    let gone = Cancel::new();
    gone.cancel();
    let error = fabric
        .dispatch(
            "/v1/chat/completions",
            &body,
            &request,
            FORWARD_TIMEOUT,
            &gone,
        )
        .expect_err("the client had gone");
    assert!(
        matches!(
            error,
            DispatchError::Forward(ForwardError::Cancelled { .. })
        ),
        "got {error:?}"
    );

    fabric
        .dispatch(
            "/v1/chat/completions",
            &body,
            &request,
            FORWARD_TIMEOUT,
            &Cancel::never(),
        )
        .expect("the node still serves");

    assert_eq!(
        health_probes(&alpha),
        after_the_first,
        "the fabric re-probed because a client hung up"
    );
}
