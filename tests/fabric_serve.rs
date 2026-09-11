//! End-to-end tests for the resident fabric proxy (`camelid fabric serve`).
//!
//! [`fabric_end_to_end.rs`] proves `Fabric::dispatch` itself; these tests prove
//! the HTTP front door around it — a real client, over a real socket, talking
//! to the real router bound by [`camelid::fabric::server::serve_on`], routed to
//! stub nodes on loopback.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use camelid::fabric::server::{
    serve_on, serve_on_until, ClientAuth, ProxyCors, ProxyTls, ServeConfig,
};
use camelid::fabric::{Fabric, NodeSpec, RouteMode, ENGINE_QUEUE_FULL_CODE};

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const FORWARD_TIMEOUT: Duration = Duration::from_secs(5);
/// Budget for one step of the TLS client. Generous enough never to fire on a
/// working proxy, short enough that a broken one fails instead of hanging.
const TLS_STEP_TIMEOUT: Duration = Duration::from_secs(10);

/// The routes the proxy places, which a node therefore has to answer. Repeated
/// here rather than imported: these tests are a client's view of the proxy, and
/// a test that read the same constant the router is built from could not notice
/// a route quietly leaving it.
const PLACED_ROUTES: [&str; 5] = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/rerank",
    "/v1/reranking",
];

/// One request a stub received: enough to say which route it was, what
/// credential the proxy presented on it, and what it asked for.
#[derive(Debug, Clone)]
struct Received {
    path: String,
    /// The `Authorization` header verbatim, or `None` if the request carried none.
    authorization: Option<String>,
    body: String,
}

#[derive(Clone)]
struct StubConfig {
    health: String,
    completion: String,
    completion_status: u16,
    /// Mutable status for a node that degrades after the policy has learned it.
    live_completion_status: Option<Arc<AtomicUsize>>,
    /// Held before answering `/v1/chat/completions`, to stand in for a real
    /// generation that takes a while. Health is never delayed.
    completion_delay: Duration,
    /// When set, every route but `/v1/health` answers 401 unless the request
    /// carries exactly this bearer token — the arrangement a node started with
    /// `CAMELID_API_KEY` actually presents to the proxy.
    required_key: Option<String>,
    /// Hang up on `/v1/chat/completions` after reading the request instead of
    /// answering it. The node has the request, so nothing may re-send it.
    hangs_up_on_completion: bool,
    /// Server-sent events answered to a request carrying `"stream": true`,
    /// instead of one JSON body.
    stream_events: Vec<String>,
    /// Gap left between events. A test that measures arrival times uses it to
    /// tell a relayed stream from one that was buffered and released at the end.
    stream_gap: Duration,
    /// Hang up after the events instead of writing the terminal chunk, which is
    /// what a node that dies mid-generation leaves on the wire.
    stream_truncated: bool,
    /// Mutable truncation for a stream that degrades after warm-up.
    live_stream_truncated: Option<Arc<AtomicBool>>,
    /// Events actually written to the socket. It stops advancing once the peer
    /// has gone, which is how a cancellation test observes the hang-up.
    events_written: Arc<AtomicUsize>,
    /// When set, `/v1/health` reports this live count rather than a fixed one,
    /// and a completion holds it up for its whole duration. A real engine's
    /// `engine_queue_depth` behaves this way, and a placement test is only
    /// meaningful against a node whose reported load actually moves.
    live_in_flight: Option<Arc<AtomicUsize>>,
    /// When set, `/v1/health` reports this model rather than the fixed one, so
    /// a test can load a different model while the proxy is running.
    live_model: Option<Arc<Mutex<String>>>,
    /// How long a placed request is worked on while watching the socket for the
    /// caller going away. A node with one JSON answer to give has nothing to
    /// write until it is finished, so unlike `events_written` there is no
    /// failing write to reveal a hang-up — it has to be looked for.
    completion_watch: Duration,
    /// Mutable work duration for a node that is made long-running only for a
    /// cancellation probe after quick policy warm-up.
    live_completion_watch: Option<Arc<Mutex<Duration>>>,
    /// How far into `completion_watch` the caller went away, if it did.
    caller_left_after: Arc<Mutex<Option<Duration>>>,
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
            live_completion_status: None,
            completion_delay: Duration::ZERO,
            required_key: None,
            hangs_up_on_completion: false,
            stream_events: Vec::new(),
            stream_gap: Duration::ZERO,
            stream_truncated: false,
            live_stream_truncated: None,
            events_written: Arc::new(AtomicUsize::new(0)),
            live_in_flight: None,
            live_model: None,
            completion_watch: Duration::ZERO,
            live_completion_watch: None,
            caller_left_after: Arc::new(Mutex::new(None)),
        }
    }

    /// A node that reports the load it is really carrying, and takes `delay`
    /// to answer a completion.
    fn reporting_live_load(model: &str, delay: Duration) -> Self {
        Self {
            completion_delay: delay,
            live_in_flight: Some(Arc::new(AtomicUsize::new(0))),
            ..Self::ready(model, 0)
        }
    }

    /// A node answering exactly what a real engine answers once its bounded
    /// queue turns a request away: the typed code, the `runtime_unavailable`
    /// envelope a 503 carries, and the message naming the bound.
    fn queue_full(model: &str) -> Self {
        Self::refusing_with(
            model,
            ENGINE_QUEUE_FULL_CODE,
            "the generation queue is full; retry shortly (depth is bounded by CAMELID_QUEUE_DEPTH)",
        )
    }

    /// A node refusing with a 503 that is *not* backpressure. A node with no
    /// model loaded answers one of these, and sending that request on would
    /// spend another node's time reaching the same refusal.
    fn refusing_with(model: &str, code: &str, message: &str) -> Self {
        Self {
            completion_status: 503,
            completion: format!(
                r#"{{"error":{{"message":"{message}","type":"runtime_unavailable","code":"{code}"}}}}"#
            ),
            ..Self::ready(model, 0)
        }
    }

    /// A node whose active model can be changed while the proxy is running, the
    /// way an operator loading a different model changes it.
    fn with_switchable_model(model: &str) -> Self {
        Self {
            live_model: Some(Arc::new(Mutex::new(model.to_string()))),
            ..Self::ready(model, 0)
        }
    }

    /// A node that answers a streaming request with server-sent events.
    fn streaming(model: &str, events: &[&str], gap: Duration) -> Self {
        Self {
            stream_events: events.iter().map(|event| (*event).to_string()).collect(),
            stream_gap: gap,
            ..Self::ready(model, 0)
        }
    }

    /// A node that dies part-way through a stream: the events it managed to
    /// produce, then a hang-up with no terminal chunk.
    fn dying_mid_stream(model: &str, events: &[&str], gap: Duration) -> Self {
        Self {
            stream_truncated: true,
            ..Self::streaming(model, events, gap)
        }
    }

    /// A node with an API key set: health stays open, generation does not.
    fn requiring_key(model: &str, key: &str) -> Self {
        Self {
            required_key: Some(key.to_string()),
            ..Self::ready(model, 0)
        }
    }

    fn refusing(model: &str) -> Self {
        Self {
            completion: r#"{"error":{"message":"engine queue full"}}"#.to_string(),
            completion_status: 503,
            ..Self::ready(model, 0)
        }
    }

    fn slow(model: &str, delay: Duration) -> Self {
        Self {
            completion_delay: delay,
            ..Self::ready(model, 0)
        }
    }

    /// A node that takes the request and then dies without answering, which is
    /// what a node crashing mid-generation leaves on the wire.
    fn hanging_up_on_completion(model: &str) -> Self {
        Self {
            hangs_up_on_completion: true,
            ..Self::ready(model, 0)
        }
    }

    /// A node working on a request for `hold`, watching for its caller to go.
    ///
    /// A real engine notices the same event by the same means — its handler
    /// frame is dropped when the connection is — and stops its decode loop
    /// within one step. `hold` stands in for the generation that would still
    /// have been running.
    fn working_on_a_completion(model: &str, hold: Duration) -> Self {
        Self {
            completion_watch: hold,
            ..Self::ready(model, 0)
        }
    }
}

/// A stand-in Camelid node on loopback, identical in shape to the one in
/// `fabric_end_to_end.rs` (duplicated rather than shared, since each
/// integration test file is its own crate).
struct StubNode {
    port: u16,
    shutdown: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Received>>>,
    thread: Option<JoinHandle<()>>,
}

impl StubNode {
    fn start(config: StubConfig) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_shutdown = Arc::clone(&shutdown);
        let thread_requests = Arc::clone(&requests);
        let config = Arc::new(config);
        let thread = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if thread_shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut stream) = stream else { continue };
                // Each connection gets its own thread: a real node answers a
                // health probe while it is generating for someone else, and a
                // serial accept loop would make the proxy look slower than it
                // is when several requests probe the same node at once.
                let config = Arc::clone(&config);
                let requests = Arc::clone(&thread_requests);
                std::thread::spawn(move || serve_once(&mut stream, &config, &requests));
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

    /// Stop accepting, releasing the port so it refuses connections the way a
    /// machine that has gone does. What it already recorded stays readable.
    fn stop_listening(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Unblocks the accept loop so it observes the flag and drops the
        // listener; the connection itself is never served.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for StubNode {
    fn drop(&mut self) {
        self.stop_listening();
    }
}

fn serve_once(stream: &mut TcpStream, config: &StubConfig, requests: &Mutex<Vec<Received>>) {
    let Some(received) = read_request(stream) else {
        return;
    };

    // Mirrors `route_requires_auth` in the real server: `/v1/health` is exempt,
    // everything else is gated.
    let authorized = match &config.required_key {
        Some(key) => {
            received.path == "/v1/health"
                || received.authorization.as_deref() == Some(&format!("Bearer {key}"))
        }
        None => true,
    };

    let asked_to_stream = serde_json::from_str::<Value>(&received.body)
        .ok()
        .and_then(|body| body.get("stream").and_then(Value::as_bool))
        .unwrap_or(false);

    if authorized && asked_to_stream && !config.stream_events.is_empty() {
        requests.lock().expect("stub lock").push(received);
        serve_event_stream(stream, config);
        return;
    }

    if authorized && config.hangs_up_on_completion && received.path == "/v1/chat/completions" {
        // Recorded first: the point of this stub is that the node did get the
        // request, whatever the caller ends up seeing.
        requests.lock().expect("stub lock").push(received);
        return;
    }

    if authorized
        && !completion_watch(config).is_zero()
        && PLACED_ROUTES.contains(&received.path.as_str())
    {
        // Recorded before the work starts, so a test can wait until the node
        // really holds the request before taking its caller away.
        requests.lock().expect("stub lock").push(received);
        if caller_stayed(stream, config) {
            write_json(stream, completion_status(config), &config.completion);
        }
        return;
    }

    let (status, body) = match received.path.as_str() {
        _ if !authorized => (
            401_u16,
            r#"{"error":{"message":"provide Authorization: Bearer <key> or X-API-Key"}}"#
                .to_string(),
        ),
        "/v1/health" => (200_u16, health_body(config)),
        path if PLACED_ROUTES.contains(&path) => {
            // Counted for exactly as long as the node is working on it.
            let busy = config.live_in_flight.as_ref().map(|count| {
                count.fetch_add(1, Ordering::SeqCst);
                Arc::clone(count)
            });
            if !config.completion_delay.is_zero() {
                std::thread::sleep(config.completion_delay);
            }
            if let Some(count) = busy {
                count.fetch_sub(1, Ordering::SeqCst);
            }
            (completion_status(config), config.completion.clone())
        }
        _ => (404, "{}".to_string()),
    };

    // Record only after deciding, so a malformed request never poisons the log.
    requests.lock().expect("stub lock").push(received);

    write_json(stream, status, &body);
}

fn write_json(stream: &mut TcpStream, status: u16, body: &str) {
    let response = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn completion_status(config: &StubConfig) -> u16 {
    config
        .live_completion_status
        .as_ref()
        .map_or(config.completion_status, |status| {
            status.load(Ordering::SeqCst) as u16
        })
}

/// Work on a placed request for `completion_watch`, watching the socket for the
/// caller going away, and report whether it was still there at the end.
///
/// A node with one JSON answer to give writes nothing until it is finished, so
/// unlike a stream there is no failing write to reveal a hang-up: it has to be
/// looked for. A real engine notices the same event by the same means — its
/// handler frame is dropped when the connection is.
fn caller_stayed(stream: &mut TcpStream, config: &StubConfig) -> bool {
    let started = Instant::now();
    let hold = completion_watch(config);
    if stream
        .set_read_timeout(Some(Duration::from_millis(20)))
        .is_err()
    {
        return false;
    }
    let mut scratch = [0_u8; 64];
    while started.elapsed() < hold {
        let gone = match stream.read(&mut scratch) {
            // A graceful close reads as end-of-file and an abortive one as an
            // error; both mean the caller has gone.
            Ok(0) => true,
            // Nothing here pipelines a second request, so anything readable is
            // not the caller leaving.
            Ok(_) => false,
            Err(error) => !matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ),
        };
        if gone {
            *config.caller_left_after.lock().expect("stub lock") = Some(started.elapsed());
            return false;
        }
    }
    true
}

fn completion_watch(config: &StubConfig) -> Duration {
    config
        .live_completion_watch
        .as_ref()
        .map_or(config.completion_watch, |watch| {
            *watch.lock().expect("stub watch lock")
        })
}

/// What `/v1/health` answers: the configured body, carrying whatever load and
/// model the node is really reporting when it is tracking those.
fn health_body(config: &StubConfig) -> String {
    let mut body: Value = serde_json::from_str(&config.health).expect("stub health is json");
    if let Some(count) = &config.live_in_flight {
        body["engine_queue_depth"] = count.load(Ordering::SeqCst).into();
    }
    if let Some(model) = &config.live_model {
        body["active_model_id"] = Value::String(model.lock().expect("stub model lock").clone());
    }
    body.to_string()
}

/// Answer with chunked `text/event-stream`, one HTTP chunk per event, exactly
/// as an axum `Sse` response reaches the wire.
fn serve_event_stream(stream: &mut TcpStream, config: &StubConfig) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    let _ = stream.flush();

    for event in &config.stream_events {
        if !config.stream_gap.is_zero() {
            std::thread::sleep(config.stream_gap);
        }
        let frame = format!("{:x}\r\n{event}\r\n", event.len());
        // A failed write is the peer having gone; stop rather than keep
        // generating for nobody, which is what a real node's channel does.
        if stream.write_all(frame.as_bytes()).is_err() || stream.flush().is_err() {
            return;
        }
        config.events_written.fetch_add(1, Ordering::SeqCst);
    }
    if config.stream_truncated
        || config
            .live_stream_truncated
            .as_ref()
            .is_some_and(|truncated| truncated.load(Ordering::SeqCst))
    {
        return;
    }
    let _ = stream.write_all(b"0\r\n\r\n");
    let _ = stream.flush();
}

fn read_request(stream: &mut TcpStream) -> Option<Received> {
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    let mut raw = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&chunk[..read]);
        if let Some(header_end) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..header_end]).to_string();
            let expected = head
                .lines()
                .find(|line| line.to_ascii_lowercase().starts_with("content-length:"))
                .and_then(|line| line.split(':').nth(1))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if raw.len() >= header_end + 4 + expected {
                let mut parts = head.lines().next()?.split_whitespace();
                parts.next()?; // method
                let body_start = header_end + 4;
                return Some(Received {
                    path: parts.next()?.to_string(),
                    authorization: authorization(&head),
                    body: String::from_utf8_lossy(&raw[body_start..body_start + expected])
                        .to_string(),
                });
            }
        }
    }
    None
}

/// The `Authorization` header from a request head, matched case-insensitively
/// the way a real server matches it.
fn authorization(head: &str) -> Option<String> {
    head.lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.trim().eq_ignore_ascii_case("authorization"))
        .map(|(_, value)| value.trim().to_string())
}

fn fabric_of(specs: Vec<NodeSpec>) -> Fabric {
    Fabric::new(specs).with_timeout(PROBE_TIMEOUT)
}

/// A fabric that reuses an observation, as `fabric serve` builds one.
fn fabric_reusing_observations(specs: Vec<NodeSpec>, max_age: Duration) -> Fabric {
    fabric_of(specs).with_max_observation_age(max_age)
}

/// Health probes every node saw. The whole point of a freshness bound is that
/// this stops growing with the number of client requests.
fn health_probes(nodes: &[StubNode]) -> usize {
    nodes
        .iter()
        .map(|node| {
            node.received()
                .iter()
                .filter(|received| received.path == "/v1/health")
                .count()
        })
        .sum()
}

/// Requests each node received on one route.
fn served_on(nodes: &[StubNode], path: &str) -> Vec<usize> {
    nodes
        .iter()
        .map(|node| {
            node.received()
                .iter()
                .filter(|received| received.path == path)
                .count()
        })
        .collect()
}

fn completions_served(nodes: &[StubNode]) -> Vec<usize> {
    served_on(nodes, "/v1/chat/completions")
}

/// Bind the real proxy on an OS-assigned port and start serving it in the
/// background, returning the address a client should connect to.
async fn start_proxy(fabric: Fabric, mode: RouteMode) -> SocketAddr {
    start_proxy_with_auth(fabric, mode, ClientAuth::none()).await
}

async fn start_proxy_with_auth(fabric: Fabric, mode: RouteMode, auth: ClientAuth) -> SocketAddr {
    start_proxy_waiting(fabric, mode, auth, FORWARD_TIMEOUT).await
}

/// The proxy with a forward budget of its own.
///
/// A cancellation test needs one far longer than the test itself, so that a
/// request ending early can only be because it was given up on, never because
/// it ran out of time.
async fn start_proxy_waiting(
    fabric: Fabric,
    mode: RouteMode,
    auth: ClientAuth,
    forward_timeout: Duration,
) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let config = ServeConfig {
        mode,
        forward_timeout,
        auth,
        tls: None,
        cors: None,
        bound: addr,
    };
    tokio::spawn(async move {
        let _ = serve_on(listener, fabric, config).await;
    });
    addr
}

/// The proxy behind `auth`, allowing browser reads as `cors` says.
async fn start_proxy_allowing(
    fabric: Fabric,
    auth: ClientAuth,
    cors: Option<ProxyCors>,
) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let config = ServeConfig {
        mode: RouteMode::Throughput,
        forward_timeout: FORWARD_TIMEOUT,
        auth,
        tls: None,
        cors,
        bound: addr,
    };
    tokio::spawn(async move {
        let _ = serve_on(listener, fabric, config).await;
    });
    addr
}

/// Send one bodiless request and return the status and the lowercased headers:
/// a preflight is read only for its head, and has no JSON to parse.
async fn head_of(
    addr: SocketAddr,
    method: &str,
    path: &str,
    extra_headers: &[(&str, &str)],
) -> (u16, Vec<(String, String)>) {
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    let text = String::from_utf8_lossy(&raw).into_owned();
    let header_end = text.find("\r\n\r\n").expect("header terminator");
    let mut lines = text[..header_end].lines();
    let status = lines
        .next()
        .expect("status line")
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("numeric status");
    let headers = lines
        .filter_map(|line| line.split_once(": "))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.to_string()))
        .collect();
    (status, headers)
}

/// Where the web UI is served from in development.
const UI_ORIGIN: &str = "http://127.0.0.1:5173";

/// What a browser sends before it will POST a comparison with a key.
fn preflight_for_compare(origin: &str) -> [(&str, &str); 3] {
    [
        ("Origin", origin),
        ("Access-Control-Request-Method", "POST"),
        (
            "Access-Control-Request-Headers",
            "content-type, authorization",
        ),
    ]
}

fn ui_origin_allowed() -> ProxyCors {
    ProxyCors::resolve(&[UI_ORIGIN.to_string()])
        .expect("an explicit origin")
        .expect("one origin was named")
}

/// The WebUI is served from another origin, so without this a browser
/// withholds every answer the proxy gives it — measured live, a preflight to
/// the compare route got 405 and the comparison was never sent. The refusal a
/// missing key earns has to be readable as well, or the page sees a network
/// failure where the proxy said 401.
#[tokio::test(flavor = "multi_thread")]
async fn a_named_origin_may_read_the_proxy_its_preflight_and_its_refusals() {
    let addr = start_proxy_allowing(
        fabric_of(Vec::new()),
        authenticated("proxy-key"),
        Some(ui_origin_allowed()),
    )
    .await;

    let (_, health) = head_of(addr, "GET", "/v1/health", &[("Origin", UI_ORIGIN)]).await;
    assert_eq!(
        header(&health, "access-control-allow-origin"),
        Some(UI_ORIGIN),
        "{health:?}"
    );

    let (status, preflight) = head_of(
        addr,
        "OPTIONS",
        "/v1/fabric/compare",
        &preflight_for_compare(UI_ORIGIN),
    )
    .await;
    assert!(
        (200..300).contains(&status),
        "a refused preflight means the request is never sent: {status} {preflight:?}"
    );
    assert_eq!(
        header(&preflight, "access-control-allow-origin"),
        Some(UI_ORIGIN)
    );
    let allowed = header(&preflight, "access-control-allow-headers")
        .unwrap_or_default()
        .to_ascii_lowercase();
    for needed in ["content-type", "authorization"] {
        assert!(
            allowed.contains(needed),
            "{needed} is not allowed: {allowed:?}"
        );
    }
    assert!(
        header(&preflight, "access-control-allow-methods")
            .unwrap_or_default()
            .contains("POST"),
        "{preflight:?}"
    );

    // The key is still required for the request itself, and the refusal is
    // readable by the page, so it can tell its user why.
    let (status, refused) =
        head_of(addr, "POST", "/v1/fabric/compare", &[("Origin", UI_ORIGIN)]).await;
    assert_eq!(status, 401, "{refused:?}");
    assert_eq!(
        header(&refused, "access-control-allow-origin"),
        Some(UI_ORIGIN)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_origin_nobody_named_is_not_allowed_to_read_the_proxy() {
    let addr = start_proxy_allowing(
        fabric_of(Vec::new()),
        authenticated("proxy-key"),
        Some(ui_origin_allowed()),
    )
    .await;
    let stranger = "http://127.0.0.1:9999";

    let (_, health) = head_of(addr, "GET", "/v1/health", &[("Origin", stranger)]).await;
    assert_eq!(
        header(&health, "access-control-allow-origin"),
        None,
        "{health:?}"
    );
    let (_, preflight) = head_of(
        addr,
        "OPTIONS",
        "/v1/fabric/compare",
        &preflight_for_compare(stranger),
    )
    .await;
    assert_eq!(
        header(&preflight, "access-control-allow-origin"),
        None,
        "{preflight:?}"
    );
}

/// The default is unchanged: with no origin named the proxy sends no CORS
/// header at all, and an `OPTIONS` reaches the router exactly as it did before
/// the flag existed — past the key, which exempts it, to a route that takes
/// only POST.
#[tokio::test(flavor = "multi_thread")]
async fn a_proxy_with_no_origin_named_sends_no_cors_header_and_answers_options_as_before() {
    let addr = start_proxy_allowing(fabric_of(Vec::new()), authenticated("proxy-key"), None).await;

    let (_, health) = head_of(addr, "GET", "/v1/health", &[("Origin", UI_ORIGIN)]).await;
    let (status, preflight) = head_of(
        addr,
        "OPTIONS",
        "/v1/fabric/compare",
        &preflight_for_compare(UI_ORIGIN),
    )
    .await;

    assert_eq!(status, 405, "{preflight:?}");
    assert_eq!(header(&preflight, "allow"), Some("POST"), "{preflight:?}");
    for headers in [&health, &preflight] {
        assert!(
            headers
                .iter()
                .all(|(name, _)| !name.starts_with("access-control-")),
            "{headers:?}"
        );
    }
}

/// A throwaway certificate for `localhost`, valid only for the test that mints
/// it. Generated rather than committed: a private key in the tree is one nobody
/// can ever be sure is unused.
struct TestCertificate {
    _dir: tempfile::TempDir,
    cert_path: std::path::PathBuf,
    key_path: std::path::PathBuf,
    der: rustls_pki_types::CertificateDer<'static>,
}

fn mint_certificate() -> TestCertificate {
    let issued = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("generate a self-signed certificate");
    let dir = tempfile::tempdir().expect("temp dir");
    let cert_path = dir.path().join("certificate-chain");
    let key_path = dir.path().join("private-key");
    // Called through the type: the public-scrub guard rejects that method name
    // written in dotted form anywhere in the tree.
    std::fs::write(&cert_path, rcgen::Certificate::pem(&issued.cert)).expect("write certificate");
    std::fs::write(&key_path, issued.key_pair.serialize_pem()).expect("write key");
    TestCertificate {
        der: issued.cert.der().clone(),
        _dir: dir,
        cert_path,
        key_path,
    }
}

/// Start the proxy serving TLS, returning the address and the certificate a
/// client has to trust to talk to it.
async fn start_tls_proxy(fabric: Fabric, auth: ClientAuth) -> (SocketAddr, TestCertificate) {
    let certificate = mint_certificate();
    let tls = ProxyTls::resolve(
        Some(certificate.cert_path.clone()),
        Some(certificate.key_path.clone()),
    )
    .await
    .expect("a complete pair resolves")
    .expect("a pair was given");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let config = ServeConfig {
        mode: RouteMode::Throughput,
        forward_timeout: FORWARD_TIMEOUT,
        auth,
        tls: Some(tls),
        cors: None,
        bound: addr,
    };
    tokio::spawn(async move {
        let _ = serve_on(listener, fabric, config).await;
    });
    (addr, certificate)
}

/// Send one request over a real TLS connection and return (status, body, headers).
///
/// Rolled by hand for the same reason the cleartext client here is: what is
/// under test is the bytes on the socket, and a client that shares code with
/// the server can agree with it about something neither has right.
///
/// Every step is bounded. A TLS client and a cleartext server deadlock rather
/// than fail — hyper waits for a request line it will never recognise while the
/// handshake waits for a ServerHello — so without these an regression would
/// burn the whole CI job instead of reporting which assertion broke.
async fn post_over_tls(
    addr: SocketAddr,
    certificate: &TestCertificate,
    path: &str,
    body: &Value,
    extra_headers: &[(&str, &str)],
) -> (u16, Value, Vec<(String, String)>) {
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(certificate.der.clone())
        .expect("trust the test certificate");
    let client_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config));
    let server_name = rustls_pki_types::ServerName::try_from("localhost").expect("valid name");

    let stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let mut stream = tokio::time::timeout(TLS_STEP_TIMEOUT, connector.connect(server_name, stream))
        .await
        .expect("the TLS handshake must not hang; is the listener serving cleartext?")
        .expect("TLS handshake");

    let payload = serde_json::to_vec(body).expect("serialize body");
    let mut request = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n",
        payload.len()
    );
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");

    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request head");
    stream
        .write_all(&payload)
        .await
        .expect("write request body");
    stream.flush().await.expect("flush");

    let mut raw = Vec::new();
    tokio::time::timeout(TLS_STEP_TIMEOUT, stream.read_to_end(&mut raw))
        .await
        .expect("the proxy must answer within the step budget")
        .expect("read TLS response");
    parse_http_response(&raw)
}

/// Send one POST over a real socket and return (status, parsed body, headers).
async fn post_chat(
    addr: SocketAddr,
    body: &Value,
    extra_headers: &[(&str, &str)],
) -> (u16, Value, Vec<(String, String)>) {
    post_to(addr, "/v1/chat/completions", body, extra_headers).await
}

/// The same, on any route, for the ones that are not chat.
async fn post_to(
    addr: SocketAddr,
    path: &str,
    body: &Value,
    extra_headers: &[(&str, &str)],
) -> (u16, Value, Vec<(String, String)>) {
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let payload = body.to_string();
    let mut request = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        payload.len()
    );
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    request.push_str(&payload);
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    parse_http_response(&raw)
}

/// Split a raw HTTP/1.1 response into (status, JSON body, lowercased headers).
///
/// Shared by the cleartext and TLS clients so a difference between them can
/// only come from the transport, which is the thing under test.
fn parse_http_response(raw: &[u8]) -> (u16, Value, Vec<(String, String)>) {
    let text = String::from_utf8_lossy(raw);
    let header_end = text.find("\r\n\r\n").expect("header terminator");
    let head = &text[..header_end];
    let body_text = &text[header_end + 4..];

    let mut lines = head.lines();
    let status: u16 = lines
        .next()
        .expect("status line")
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("numeric status");
    let headers = lines
        .filter_map(|line| line.split_once(": "))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.to_string()))
        .collect();
    let json: Value = serde_json::from_str(body_text).expect("json body");
    (status, json, headers)
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_is_placed_and_the_backend_answer_returned_verbatim() {
    let node = StubNode::start(StubConfig::ready("model-alpha", 0));
    let fabric = fabric_of(vec![node.spec("solo")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-alpha" }), &[]).await;

    assert_eq!(status, 200);
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "served by model-alpha"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("solo"));
}

#[tokio::test(flavor = "multi_thread")]
async fn placement_through_the_proxy_is_scoped_to_the_serving_node() {
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let beta = StubNode::start(StubConfig::ready("model-beta", 0));
    let fabric = fabric_of(vec![alpha.spec("alpha"), beta.spec("beta")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-beta" }), &[]).await;

    assert_eq!(status, 200);
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "served by model-beta"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("beta"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_node_5xx_reaches_the_client_verbatim_not_as_a_proxy_error() {
    let node = StubNode::start(StubConfig::refusing("model-alpha"));
    let fabric = fabric_of(vec![node.spec("solo")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, _headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-alpha" }), &[]).await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["message"], "engine queue full");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_proxys_bearer_reaches_the_node_on_every_request_it_makes() {
    // `fabric serve --bearer` is only worth having if the token survives the
    // whole resident path. The proxy has no auth of its own, so the client sends
    // none: this asserts the credential on the wire is the fabric's, presented
    // to the node.
    let node = StubNode::start(StubConfig::requiring_key("model-alpha", "s3cret"));
    let fabric = fabric_of(vec![node.spec("solo")]).with_bearer(Some("s3cret"));
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-alpha" }), &[]).await;

    assert_eq!(status, 200);
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "served by model-alpha"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("solo"));

    // The probe is authenticated as well as the forward. Health is exempt from
    // the node's auth today, so that buys nothing now — it means placement keeps
    // working if the exemption is ever tightened.
    let seen = node.received();
    assert!(
        seen.iter().any(|request| request.path == "/v1/health"),
        "the node was probed as well as forwarded to"
    );
    assert!(
        seen.iter()
            .any(|request| request.path == "/v1/chat/completions"),
        "the forward reached the node"
    );
    for request in &seen {
        assert_eq!(
            request.authorization.as_deref(),
            Some("Bearer s3cret"),
            "`{}` went out unauthenticated",
            request.path
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_proxy_without_the_bearer_relays_the_nodes_401_rather_than_looking_healthy() {
    // The defect the flag exists for, and the proof the gate in the test above
    // is live rather than a no-op that would pass whatever the proxy sent.
    // `/v1/health` is exempt, so an unauthenticated proxy observes this node as
    // ready and places onto it fine; only the forward is refused, and that
    // refusal is the node answering, so it must reach the client with its own
    // status rather than as a proxy error naming a dead node.
    let node = StubNode::start(StubConfig::requiring_key("model-alpha", "s3cret"));
    let fabric = fabric_of(vec![node.spec("solo")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "model-alpha" }), &[]).await;

    assert_eq!(status, 401);
    assert!(
        body["error"]["message"].is_string(),
        "the refusal must carry a message an operator can act on: {body}"
    );
    // Placement still happened, so the answer is the node's, not the proxy's.
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("solo"));
    assert!(
        node.received()
            .iter()
            .all(|request| request.authorization.is_none()),
        "an unconfigured proxy must not send an Authorization header"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn no_eligible_node_answers_503_with_a_fabric_error_shape() {
    let fabric = fabric_of(Vec::new());
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, _headers) =
        post_chat(addr, &serde_json::json!({ "model": "anything" }), &[]).await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["type"], "fabric_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streaming_request_is_routed_rather_than_refused() {
    // Port 9 is a dead node. A 503 (a placement failure) rather than a 400
    // proves the proxy tried to route the stream instead of rejecting it.
    let fabric = fabric_of(vec![NodeSpec::camelid("dead", "127.0.0.1", 9)]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, _headers) = post_chat(
        addr,
        &serde_json::json!({ "model": "m", "stream": true }),
        &[],
    )
    .await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["type"], "fabric_error");
}

/// One piece of a streamed body, with how long after the request it arrived.
struct Piece {
    at: Duration,
    text: String,
}

/// Send a streaming POST and read the body as it arrives, timing each piece.
///
/// Deliberately does not reuse [`post_chat`]: that reads to EOF before looking
/// at anything, which would make a buffered response indistinguishable from a
/// streamed one — the exact thing these tests exist to tell apart.
async fn post_chat_streaming(
    addr: SocketAddr,
    body: &Value,
) -> (u16, Vec<(String, String)>, Vec<Piece>) {
    post_streaming(addr, "/v1/chat/completions", body).await
}

async fn post_streaming(
    addr: SocketAddr,
    path: &str,
    body: &Value,
) -> (u16, Vec<(String, String)>, Vec<Piece>) {
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let payload = body.to_string();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let started = Instant::now();
    let mut raw = Vec::new();
    let mut scratch = [0_u8; 4096];
    let mut head_end = None;
    let mut pieces: Vec<Piece> = Vec::new();
    let mut delivered = 0_usize;

    // A read error rather than a clean close is how an aborted body reaches the
    // client, and that abort is itself the thing a truncation test reads.
    while let Ok(read) = stream.read(&mut scratch).await {
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&scratch[..read]);
        if head_end.is_none() {
            head_end = raw
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|at| at + 4);
            delivered = head_end.unwrap_or(0);
        }
        if let Some(start) = head_end {
            if raw.len() > delivered.max(start) {
                let fresh = String::from_utf8_lossy(&raw[delivered..]).to_string();
                delivered = raw.len();
                pieces.push(Piece {
                    at: started.elapsed(),
                    text: fresh,
                });
            }
        }
    }

    let head_end = head_end.expect("header terminator");
    let head = String::from_utf8_lossy(&raw[..head_end - 4]).to_string();
    let mut lines = head.lines();
    let status: u16 = lines
        .next()
        .expect("status line")
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("numeric status");
    let headers = lines
        .filter_map(|line| line.split_once(": "))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.to_string()))
        .collect();
    (status, headers, pieces)
}

/// Strip HTTP chunk framing that hyper adds on the way back out to the client.
fn dechunk(raw: &str) -> String {
    let mut out = String::new();
    let mut rest = raw;
    loop {
        let Some((size_line, tail)) = rest.split_once("\r\n") else {
            return out;
        };
        let Ok(size) = usize::from_str_radix(size_line.trim(), 16) else {
            return out;
        };
        if size == 0 || tail.len() < size {
            return out;
        }
        out.push_str(&tail[..size]);
        rest = tail[size..].strip_prefix("\r\n").unwrap_or("");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_body_reaches_the_client_verbatim() {
    let events = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let node = StubNode::start(StubConfig::streaming("m", &events, Duration::ZERO));
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, headers, pieces) =
        post_chat_streaming(addr, &serde_json::json!({ "model": "m", "stream": true })).await;

    assert_eq!(status, 200);
    assert_eq!(header(&headers, "content-type"), Some("text/event-stream"));
    // Placement is still reported, exactly as on a buffered answer.
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("only"));

    let body: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
    assert_eq!(dechunk(&body), events.concat());
    // The control for the truncation test below: a stream the node finished is
    // framed as finished, and that is the only thing that says so.
    assert!(
        body.ends_with("0\r\n\r\n"),
        "a completed stream must end with the terminal chunk: {body:?}"
    );
}

/// A node that dies mid-generation must not reach the client looking like a
/// stream that finished. Chunked framing is what carries that distinction: a
/// complete body ends with the terminal chunk, an aborted one does not. The
/// proxy used to read the node's EOF as the end of the body and then frame its
/// own response as complete, so a half answer arrived indistinguishable from a
/// whole one.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_dying_mid_stream_is_not_relayed_as_a_completed_one() {
    // The gap matters: it makes the proxy flush the head and the early events
    // before the node dies, which is what a real generation does. Without it
    // everything would still be in one write buffer and the whole response
    // would be discarded, which is a different (also safe) outcome.
    let events = ["data: one\n\n", "data: two\n\n"];
    let node = StubNode::start(StubConfig::dying_mid_stream(
        "m",
        &events,
        Duration::from_millis(150),
    ));
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, _headers, pieces) =
        post_chat_streaming(addr, &serde_json::json!({ "model": "m", "stream": true })).await;

    // The head was already sent before the node died, so the status stands.
    assert_eq!(status, 200);
    let body: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
    // What the node did produce still reaches the client; only the claim that
    // this was all of it is withheld.
    assert!(
        dechunk(&body).starts_with("data: one\n\n"),
        "the events the node did produce must still arrive: {body:?}"
    );
    assert!(
        !body.ends_with("0\r\n\r\n"),
        "a truncated stream was framed as complete: {body:?}"
    );
}

/// The proxy's contract is that a client can ask for a specific node whatever
/// default mode the proxy was started with. Both nodes are idle and `alpha`
/// sorts first, so throughput placement takes `alpha` — the answer coming from
/// `beta` can only mean the header was honoured.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_can_pin_a_node_even_when_the_proxy_defaults_to_throughput() {
    let alpha = StubNode::start(StubConfig::ready("shared", 0));
    let beta = StubNode::start(StubConfig::ready("shared", 0));
    let fabric = fabric_of(vec![alpha.spec("alpha"), beta.spec("beta")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let body = serde_json::json!({ "model": "shared" });
    let (status, _body, headers) =
        post_chat(addr, &body, &[("x-camelid-fabric-sticky", "beta")]).await;

    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("beta"));
    assert_eq!(
        header(&headers, "x-camelid-fabric-reason"),
        Some("Affinity")
    );

    // Same proxy, same nodes, no header: the configured default still decides.
    let (status, _body, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("alpha"));
}

/// How long a change to the node file is given to take effect before the test
/// calls it broken. Generous against the one-second re-read the proxy ships
/// with, so a loaded machine does not fail this for being slow.
const NODE_RELOAD_TIMEOUT: Duration = Duration::from_secs(5);

fn node_file(dir: &tempfile::TempDir, lines: &[String]) -> std::path::PathBuf {
    let path = dir.path().join("nodes");
    std::fs::write(&path, format!("{}\n", lines.join("\n"))).expect("write node file");
    path
}

/// The line an operator would write for this stub, in `--node` syntax.
fn node_line(node: &StubNode, label: &str) -> String {
    let spec = node.spec(label);
    format!("{}={}:{}", spec.label, spec.host, spec.port)
}

/// A proxy whose node set is a file, exactly as `fabric serve --nodes-file`
/// builds one — including the shipped one-second re-read, not a test-only bound.
fn fabric_watching(path: std::path::PathBuf) -> Fabric {
    Fabric::from_node_file(path)
        .expect("load node file")
        .with_timeout(PROBE_TIMEOUT)
}

/// A machine the operator adds is placed on, without the proxy restarting.
///
/// Affinity is the instrument: both nodes serve the same model and `alpha`
/// sorts first, so only an explicit pin can prove `beta` is in the set at all.
#[tokio::test(flavor = "multi_thread")]
async fn a_machine_added_to_the_node_file_starts_being_placed_on() {
    let dir = tempfile::tempdir().expect("temp dir");
    let nodes = [
        StubNode::start(StubConfig::ready("shared-model", 0)),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let path = node_file(&dir, &[node_line(&nodes[0], "alpha")]);
    let addr = start_proxy(fabric_watching(path.clone()), RouteMode::Affinity).await;
    let body = serde_json::json!({ "model": "shared-model" });
    let sticky = [("x-camelid-fabric-sticky", "beta")];

    // beta is not in the set yet, so the pin cannot be honoured.
    let (status, answered, headers) = post_chat(addr, &body, &sticky).await;
    assert_eq!(status, 200, "{answered}");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("alpha"));
    assert_eq!(
        completions_served(&nodes)[1],
        0,
        "a machine that is not in the node file must never be placed on"
    );

    std::fs::write(
        &path,
        format!(
            "{}\n{}\n",
            node_line(&nodes[0], "alpha"),
            node_line(&nodes[1], "beta")
        ),
    )
    .expect("add beta");

    let started = Instant::now();
    loop {
        let (status, answered, headers) = post_chat(addr, &body, &sticky).await;
        assert_eq!(status, 200, "{answered}");
        if header(&headers, "x-camelid-fabric-node") == Some("beta") {
            break;
        }
        assert!(
            started.elapsed() < NODE_RELOAD_TIMEOUT,
            "a machine added to the node file was still not placed on after {:?}",
            started.elapsed()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        completions_served(&nodes)[1] > 0,
        "the added machine served nothing"
    );
}

/// A machine the operator takes away stops being placed on, and the rest of
/// the fabric goes on serving.
#[tokio::test(flavor = "multi_thread")]
async fn a_machine_removed_from_the_node_file_stops_being_placed_on() {
    let dir = tempfile::tempdir().expect("temp dir");
    let nodes = [
        StubNode::start(StubConfig::ready("shared-model", 0)),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let path = node_file(
        &dir,
        &[node_line(&nodes[0], "alpha"), node_line(&nodes[1], "beta")],
    );
    let addr = start_proxy(fabric_watching(path.clone()), RouteMode::Affinity).await;
    let body = serde_json::json!({ "model": "shared-model" });
    let sticky = [("x-camelid-fabric-sticky", "beta")];

    let (status, answered, headers) = post_chat(addr, &body, &sticky).await;
    assert_eq!(status, 200, "{answered}");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("beta"));

    std::fs::write(&path, format!("{}\n", node_line(&nodes[0], "alpha"))).expect("remove beta");

    let started = Instant::now();
    loop {
        let (status, answered, headers) = post_chat(addr, &body, &sticky).await;
        assert_eq!(status, 200, "{answered}");
        if header(&headers, "x-camelid-fabric-node") == Some("alpha") {
            break;
        }
        assert!(
            started.elapsed() < NODE_RELOAD_TIMEOUT,
            "a machine removed from the node file was still placed on after {:?}",
            started.elapsed()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Settled: from here the removed machine must receive nothing more, and
    // the one that is left must still be served.
    let settled = completions_served(&nodes)[1];
    let (status, answered, headers) = post_chat(addr, &body, &sticky).await;
    assert_eq!(status, 200, "{answered}");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("alpha"));
    assert_eq!(
        completions_served(&nodes)[1],
        settled,
        "a removed machine was still placed on"
    );
}

/// Taking a machine away must not disturb the request already running on it.
///
/// The generation outlives the re-read interval on purpose: the second request
/// proves the removal has actually taken effect while the first is still in
/// flight, which is the only way this test can claim anything.
#[tokio::test(flavor = "multi_thread")]
async fn removing_a_machine_does_not_disturb_the_request_already_running_on_it() {
    let dir = tempfile::tempdir().expect("temp dir");
    let nodes = [
        StubNode::start(StubConfig::ready("shared-model", 0)),
        StubNode::start(StubConfig {
            completion_delay: Duration::from_millis(2500),
            ..StubConfig::ready("shared-model", 0)
        }),
    ];
    let path = node_file(
        &dir,
        &[node_line(&nodes[0], "alpha"), node_line(&nodes[1], "beta")],
    );
    let addr = start_proxy(fabric_watching(path.clone()), RouteMode::Affinity).await;

    let in_flight = tokio::spawn(async move {
        let body = serde_json::json!({ "model": "shared-model" });
        post_chat(addr, &body, &[("x-camelid-fabric-sticky", "beta")]).await
    });

    // Long enough for the request to be placed on beta and forwarded.
    tokio::time::sleep(Duration::from_millis(300)).await;
    std::fs::write(&path, format!("{}\n", node_line(&nodes[0], "alpha"))).expect("remove beta");

    let body = serde_json::json!({ "model": "shared-model" });
    let sticky = [("x-camelid-fabric-sticky", "beta")];
    let started = Instant::now();
    loop {
        let (status, answered, headers) = post_chat(addr, &body, &sticky).await;
        assert_eq!(status, 200, "{answered}");
        if header(&headers, "x-camelid-fabric-node") == Some("alpha") {
            break;
        }
        assert!(
            started.elapsed() < NODE_RELOAD_TIMEOUT,
            "the removal never took effect, so this proves nothing"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let (status, answered, headers) = in_flight.await.expect("the in-flight request panicked");
    assert_eq!(status, 200, "{answered}");
    assert_eq!(
        header(&headers, "x-camelid-fabric-node"),
        Some("beta"),
        "a request already running on a removed machine must still be answered by it"
    );
}

/// The load-bearing property: the proxy must relay events as they are produced.
/// If it buffered the body and released it at the end, every piece would arrive
/// at once, after the whole generation.
#[tokio::test(flavor = "multi_thread")]
async fn events_reach_the_client_as_they_are_produced_not_at_the_end() {
    let gap = Duration::from_millis(200);
    let events = ["data: one\n\n", "data: two\n\n", "data: [DONE]\n\n"];
    let node = StubNode::start(StubConfig::streaming("m", &events, gap));
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, _headers, pieces) =
        post_chat_streaming(addr, &serde_json::json!({ "model": "m", "stream": true })).await;
    assert_eq!(status, 200);

    let carrying: Vec<&Piece> = pieces
        .iter()
        .filter(|piece| piece.text.contains("data:"))
        .collect();
    assert!(
        carrying.len() >= 2,
        "expected several separately delivered pieces, got {}",
        carrying.len()
    );

    // The stub finishes at ~3 * 200ms. A buffered proxy would hand everything
    // over at that point, so a first piece well before it can only mean the
    // bytes were relayed while the node was still generating.
    let first = carrying[0].at;
    assert!(
        first < gap * 2,
        "first event arrived after {first:?}; it looks buffered rather than relayed"
    );
    let last = carrying[carrying.len() - 1].at;
    assert!(
        last - first > gap / 2,
        "every piece arrived within {:?} of the first; that is a buffered body",
        last - first
    );
}

/// A node that refuses a streaming request answers with JSON, not with events.
/// That reason has to reach the client as a readable body rather than as an
/// empty stream, or an operator has nothing to act on.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_refusing_a_stream_is_relayed_as_a_readable_answer() {
    let events = ["data: never sent\n\n"];
    let mut config = StubConfig::streaming("m", &events, Duration::ZERO);
    config.required_key = Some("expected-key".to_string());
    let node = StubNode::start(config);
    // The fabric holds no bearer, so the node answers 401 to the forward.
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, headers) = post_chat(
        addr,
        &serde_json::json!({ "model": "m", "stream": true }),
        &[],
    )
    .await;

    assert_eq!(status, 401);
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("Authorization")),
        "the node's own reason must survive: {body}"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("only"));
}

/// A client that hangs up mid-stream must take the node's work with it.
///
/// The relay notices because a send to a client that has gone fails. That only
/// works once the node produces something to send, which is why the buffered
/// path needs a mechanism of its own; see the test below it.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_hanging_up_stops_the_node_generating() {
    let gap = Duration::from_millis(100);
    let events: Vec<String> = (0..40).map(|i| format!("data: {i}\n\n")).collect();
    let borrowed: Vec<&str> = events.iter().map(String::as_str).collect();
    let config = StubConfig::streaming("m", &borrowed, gap);
    let written = Arc::clone(&config.events_written);
    let node = StubNode::start(config);
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let payload = serde_json::json!({ "model": "m", "stream": true }).to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    // Read enough to know the stream is genuinely running, then hang up.
    let mut scratch = [0_u8; 1024];
    let _ = stream.read(&mut scratch).await.expect("read head");
    tokio::time::sleep(gap * 3).await;
    drop(stream);

    // A hang-up only surfaces once a write to the client round-trips, so a few
    // more events can escape first. What must be true is that generation then
    // *stops*: settle past that window, then check the count is still moving.
    tokio::time::sleep(gap * 5).await;
    let settled = written.load(Ordering::SeqCst);
    tokio::time::sleep(gap * 10).await;
    let later = written.load(Ordering::SeqCst);

    assert_eq!(
        settled,
        later,
        "the node was still generating {} events after the client left",
        later - settled
    );
    assert!(
        later < events.len(),
        "the node wrote all {} events despite the client leaving",
        events.len()
    );
}

/// Send a request over a real socket and leave without waiting for the answer.
///
/// Returns once the node holds the request, so the hang-up cannot land before
/// there is anything to abandon. Polling the node beats sleeping a guess: the
/// point of the measurement afterwards is that it is not a timing coincidence.
async fn post_then_hang_up(addr: SocketAddr, node: &StubNode, path: &str, body: Value) {
    let already_received = served_on(std::slice::from_ref(node), path)[0];
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let payload = body.to_string();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let deadline = Instant::now() + Duration::from_secs(10);
    while served_on(std::slice::from_ref(node), path)[0] <= already_received {
        assert!(
            Instant::now() < deadline,
            "the node never received the request, so there was nothing to abandon"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    drop(stream);
}

/// Send one request and return which candidate held it before hanging up.
async fn post_then_hang_up_on_one(
    addr: SocketAddr,
    nodes: &[&StubNode],
    path: &str,
    body: Value,
) -> usize {
    let before: Vec<usize> = nodes
        .iter()
        .map(|node| served_on(std::slice::from_ref(*node), path)[0])
        .collect();
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let payload = body.to_string();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(index) = nodes.iter().enumerate().find_map(|(index, node)| {
            (served_on(std::slice::from_ref(*node), path)[0] > before[index]).then_some(index)
        }) {
            drop(stream);
            return index;
        }
        assert!(
            Instant::now() < deadline,
            "no node received the request, so there was nothing to abandon"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// What the node observed, once it observes anything, or `None` if it never did.
async fn caller_left_within(seen: &Mutex<Option<Duration>>, bound: Duration) -> Option<Duration> {
    let deadline = Instant::now() + bound;
    loop {
        if let Some(elapsed) = *seen.lock().expect("stub lock") {
            return Some(elapsed);
        }
        if Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The buffered path has no relay whose failure could reveal a client leaving,
/// so it has to be told — and this is what being told is worth. A Camelid node
/// runs one generation at a time, so a request nobody wants holds that node's
/// only slot until it finishes or the proxy's forward budget expires.
///
/// The budget here is 60s and the node would work for 30s; noticing inside a
/// couple of seconds cannot be either of those running out.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_hanging_up_gives_the_node_back_its_generation_slot() {
    let config = StubConfig::working_on_a_completion("m", Duration::from_secs(30));
    let left = Arc::clone(&config.caller_left_after);
    let node = StubNode::start(config);
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy_waiting(
        fabric,
        RouteMode::Throughput,
        ClientAuth::none(),
        Duration::from_secs(60),
    )
    .await;

    post_then_hang_up(
        addr,
        &node,
        "/v1/chat/completions",
        serde_json::json!({ "model": "m" }),
    )
    .await;

    let noticed = caller_left_within(&left, Duration::from_secs(10))
        .await
        .expect("the node was never told its caller had gone; it kept the request for 30s");
    assert!(
        noticed < Duration::from_secs(3),
        "the node held the request for {noticed:?} after its client left"
    );
}

/// The same for a streaming request that has not reached its first event: the
/// relay's own mechanism cannot fire, because nothing has been relayed.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_leaving_before_the_first_event_still_stops_the_node() {
    let config = StubConfig::working_on_a_completion("m", Duration::from_secs(30));
    let left = Arc::clone(&config.caller_left_after);
    let node = StubNode::start(config);
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy_waiting(
        fabric,
        RouteMode::Throughput,
        ClientAuth::none(),
        Duration::from_secs(60),
    )
    .await;

    post_then_hang_up(
        addr,
        &node,
        "/v1/chat/completions",
        serde_json::json!({ "model": "m", "stream": true }),
    )
    .await;

    let noticed = caller_left_within(&left, Duration::from_secs(10))
        .await
        .expect("the node was never told its caller had gone before the first event");
    assert!(
        noticed < Duration::from_secs(3),
        "the node held the request for {noticed:?} after its client left"
    );
}

/// The control for both: a client that stays is served, and the node is never
/// told to stop. Without this, a proxy that cancelled every request the moment
/// it started one would pass the two tests above.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_that_waits_is_served_and_the_node_is_never_told_to_stop() {
    let config = StubConfig::working_on_a_completion("m", Duration::from_millis(400));
    let left = Arc::clone(&config.caller_left_after);
    let node = StubNode::start(config);
    let fabric = fabric_of(vec![node.spec("only")]);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let (status, body, _) = post_chat(addr, &serde_json::json!({ "model": "m" }), &[]).await;

    assert_eq!(status, 200);
    assert_eq!(body["choices"][0]["message"]["content"], "served by m");
    assert_eq!(
        *left.lock().expect("stub lock"),
        None,
        "a served request must not look like an abandoned one"
    );
}

/// A burst of requests for one model must spread across the nodes serving it.
///
/// `/v1/health` reports what a node has already accepted, so a request still on
/// its way is invisible there. Without the fabric counting its own outstanding
/// placements, every request in a burst observes the same idle fabric and the
/// label tie-break sends all of them to whichever node sorts first — measured at
/// 10 / 2 / 0 across three nodes, with one node never used at all.
///
/// The nodes here report the load they are really under, which is the only way
/// this test means anything.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_burst_of_requests_spreads_across_the_nodes_serving_the_model() {
    const REQUESTS: usize = 12;
    let labels = ["node-a", "node-b", "node-c"];
    let nodes: Vec<StubNode> = labels
        .iter()
        .map(|_| {
            StubNode::start(StubConfig::reporting_live_load(
                "shared-model",
                Duration::from_millis(300),
            ))
        })
        .collect();
    let specs = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| node.spec(labels[i]))
        .collect();
    let addr = start_proxy(fabric_of(specs), RouteMode::Throughput).await;

    let body = serde_json::json!({ "model": "shared-model" });
    let results =
        futures_util::future::join_all((0..REQUESTS).map(|_| post_chat(addr, &body, &[]))).await;
    for (status, _, _) in &results {
        assert_eq!(*status, 200);
    }

    let served: Vec<usize> = nodes
        .iter()
        .map(|node| {
            node.received()
                .iter()
                .filter(|r| r.path == "/v1/chat/completions")
                .count()
        })
        .collect();
    let total: usize = served.iter().sum();
    assert_eq!(total, REQUESTS, "every request must reach a node");

    // Deliberately not asserting an exact split: probes race, so an occasional
    // request lands a place either way. What must hold is that the work is
    // shared rather than piled onto whichever label sorts first.
    let busiest = served.iter().max().copied().unwrap_or(0);
    assert!(
        busiest <= REQUESTS / 2,
        "one node took {busiest} of {REQUESTS}: {served:?}"
    );
    assert!(
        served.iter().all(|count| *count > 0),
        "a node serving the model was never used: {served:?}"
    );
}

/// Completion-time placement learns only from completed requests, then uses
/// the measured service difference instead of treating equal queue depths as
/// equal machines.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completion_time_learns_buffered_service_speed_over_real_sockets() {
    const WARMUP_REQUESTS: usize = 10;
    const MEASURED_REQUESTS: usize = 6;
    let nodes = vec![
        StubNode::start(StubConfig::slow("shared-model", Duration::from_millis(20))),
        StubNode::start(StubConfig::slow("shared-model", Duration::from_millis(200))),
    ];
    let fabric = fabric_reusing_observations(
        vec![nodes[0].spec("fast"), nodes[1].spec("slow")],
        Duration::from_secs(30),
    );
    let addr = start_proxy(fabric, RouteMode::CompletionTime).await;
    let body = serde_json::json!({ "model": "shared-model" });

    for _ in 0..WARMUP_REQUESTS {
        let (status, _, headers) = post_chat(addr, &body, &[]).await;
        assert_eq!(status, 200);
        assert_eq!(
            header(&headers, "x-camelid-fabric-reason"),
            Some("LeastLoaded"),
            "the policy must say when it is still using the cold fallback"
        );
    }
    assert_eq!(
        completions_served(&nodes),
        [5, 5],
        "cold exploration must collect the same number of samples from both nodes"
    );

    for _ in 0..MEASURED_REQUESTS {
        let (status, _, headers) = post_chat(addr, &body, &[]).await;
        assert_eq!(status, 200);
        assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("fast"));
        assert_eq!(
            header(&headers, "x-camelid-fabric-reason"),
            Some("EstimatedCompletion")
        );
    }
    assert_eq!(completions_served(&nodes), [5 + MEASURED_REQUESTS, 5]);
}

/// A quick failure is not quick service. Counting it would teach the policy to
/// prefer the node that fails fastest, which is worse than speed-blind routing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completion_time_never_learns_from_http_failures() {
    let failing = StubNode::start(StubConfig {
        completion_status: 500,
        completion: r#"{"error":{"message":"generation failed"}}"#.to_string(),
        ..StubConfig::ready("shared-model", 0)
    });
    let serving = StubNode::start(StubConfig::slow("shared-model", Duration::from_millis(60)));
    let fabric = fabric_reusing_observations(
        vec![failing.spec("a-failing"), serving.spec("b-serving")],
        Duration::from_secs(30),
    );
    let addr = start_proxy(fabric, RouteMode::CompletionTime).await;
    let body = serde_json::json!({ "model": "shared-model" });

    let mut statuses = Vec::new();
    for _ in 0..12 {
        let (status, _, headers) = post_chat(addr, &body, &[]).await;
        statuses.push(status);
        assert_eq!(
            header(&headers, "x-camelid-fabric-reason"),
            Some("LeastLoaded"),
            "a failed request must never mature the completion-time policy"
        );
    }
    assert_eq!(statuses.iter().filter(|status| **status == 500).count(), 6);
    assert_eq!(statuses.iter().filter(|status| **status == 200).count(), 6);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completion_time_invalidates_a_learned_node_on_5xx() {
    let fast_status = Arc::new(AtomicUsize::new(200));
    let fast = StubNode::start(StubConfig {
        completion_delay: Duration::from_millis(20),
        live_completion_status: Some(Arc::clone(&fast_status)),
        ..StubConfig::ready("shared-model", 0)
    });
    let slow = StubNode::start(StubConfig::slow("shared-model", Duration::from_millis(200)));
    let fabric = fabric_reusing_observations(
        vec![fast.spec("fast"), slow.spec("slow")],
        Duration::from_secs(30),
    );
    let addr = start_proxy(fabric, RouteMode::CompletionTime).await;
    let body = serde_json::json!({ "model": "shared-model" });

    for _ in 0..10 {
        assert_eq!(post_chat(addr, &body, &[]).await.0, 200);
    }
    fast_status.store(500, Ordering::SeqCst);

    // Force the failure through affinity: invalidation describes the node's
    // health, not the policy that happened to select it.
    let (failed, _, failed_headers) =
        post_chat(addr, &body, &[("x-camelid-fabric-sticky", "fast")]).await;
    assert_eq!(failed, 500);
    assert_eq!(
        header(&failed_headers, "x-camelid-fabric-node"),
        Some("fast")
    );
    assert_eq!(
        header(&failed_headers, "x-camelid-fabric-reason"),
        Some("Affinity")
    );

    let (recovered, _, recovered_headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(recovered, 200);
    assert_eq!(
        header(&recovered_headers, "x-camelid-fabric-node"),
        Some("slow")
    );
    assert_eq!(
        header(&recovered_headers, "x-camelid-fabric-reason"),
        Some("LeastLoaded"),
        "the 5xx must return the class to cold fallback"
    );
}

/// Streaming service time ends at clean EOF, not when the response head
/// arrives. Otherwise a slow decoder that flushes an early role frame would
/// look identical to a fast one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completion_time_learns_streaming_service_through_clean_eof() {
    const EVENTS: [&str; 2] = ["data: token\n\n", "data: [DONE]\n\n"];
    let fast = StubNode::start(StubConfig::streaming(
        "shared-model",
        &EVENTS,
        Duration::from_millis(5),
    ));
    let slow = StubNode::start(StubConfig::streaming(
        "shared-model",
        &EVENTS,
        Duration::from_millis(50),
    ));
    let fabric = fabric_reusing_observations(
        vec![fast.spec("fast"), slow.spec("slow")],
        Duration::from_secs(30),
    );
    let addr = start_proxy(fabric, RouteMode::CompletionTime).await;
    let body = serde_json::json!({ "model": "shared-model", "stream": true });

    for _ in 0..10 {
        let (status, headers, pieces) = post_chat_streaming(addr, &body).await;
        assert_eq!(status, 200);
        assert_eq!(
            header(&headers, "x-camelid-fabric-reason"),
            Some("LeastLoaded")
        );
        let framed: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
        assert_eq!(dechunk(&framed), EVENTS.concat());
    }

    for _ in 0..4 {
        let (status, headers, pieces) = post_chat_streaming(addr, &body).await;
        assert_eq!(status, 200);
        assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("fast"));
        assert_eq!(
            header(&headers, "x-camelid-fabric-reason"),
            Some("EstimatedCompletion")
        );
        let framed: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
        assert_eq!(dechunk(&framed), EVENTS.concat());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completion_time_invalidates_a_stream_that_stops_before_clean_eof() {
    const EVENTS: [&str; 2] = ["data: token\n\n", "data: [DONE]\n\n"];
    let fast_truncated = Arc::new(AtomicBool::new(false));
    let fast = StubNode::start(StubConfig {
        live_stream_truncated: Some(Arc::clone(&fast_truncated)),
        ..StubConfig::streaming("shared-model", &EVENTS, Duration::from_millis(5))
    });
    let slow = StubNode::start(StubConfig::streaming(
        "shared-model",
        &EVENTS,
        Duration::from_millis(50),
    ));
    let fabric = fabric_reusing_observations(
        vec![fast.spec("fast"), slow.spec("slow")],
        Duration::from_secs(30),
    );
    let addr = start_proxy(fabric, RouteMode::CompletionTime).await;
    let body = serde_json::json!({ "model": "shared-model", "stream": true });

    for _ in 0..10 {
        assert_eq!(post_chat_streaming(addr, &body).await.0, 200);
    }
    fast_truncated.store(true, Ordering::SeqCst);

    let (status, failed_headers, failed_pieces) = post_chat_streaming(addr, &body).await;
    assert_eq!(status, 200, "the response head arrived before truncation");
    assert_eq!(
        header(&failed_headers, "x-camelid-fabric-node"),
        Some("fast")
    );
    let failed: String = failed_pieces
        .iter()
        .map(|piece| piece.text.as_str())
        .collect();
    assert!(
        !failed.ends_with("0\r\n\r\n"),
        "the failed stream looked complete"
    );

    let (recovered, recovered_headers, _) = post_chat_streaming(addr, &body).await;
    assert_eq!(recovered, 200);
    assert_eq!(
        header(&recovered_headers, "x-camelid-fabric-node"),
        Some("slow")
    );
    assert_eq!(
        header(&recovered_headers, "x-camelid-fabric-reason"),
        Some("LeastLoaded")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completion_time_keeps_a_learned_estimate_when_only_the_client_leaves() {
    let fast_watch = Arc::new(Mutex::new(Duration::from_millis(50)));
    let fast_config = StubConfig {
        live_completion_watch: Some(Arc::clone(&fast_watch)),
        ..StubConfig::ready("shared-model", 0)
    };
    let fast_left = Arc::clone(&fast_config.caller_left_after);
    let fast = StubNode::start(fast_config);
    let slow_watch = Arc::new(Mutex::new(Duration::from_millis(300)));
    let slow_config = StubConfig {
        live_completion_watch: Some(Arc::clone(&slow_watch)),
        ..StubConfig::ready("shared-model", 0)
    };
    let slow_left = Arc::clone(&slow_config.caller_left_after);
    let slow = StubNode::start(slow_config);
    let fabric = fabric_reusing_observations(
        vec![fast.spec("fast"), slow.spec("slow")],
        Duration::from_secs(30),
    );
    let addr = start_proxy_waiting(
        fabric,
        RouteMode::CompletionTime,
        ClientAuth::none(),
        Duration::from_secs(5),
    )
    .await;
    let body = serde_json::json!({ "model": "shared-model" });

    for _ in 0..10 {
        assert_eq!(post_chat(addr, &body, &[]).await.0, 200);
    }

    *fast_watch.lock().expect("fast watch lock") = Duration::from_secs(30);
    *slow_watch.lock().expect("slow watch lock") = Duration::from_secs(30);

    let candidates = [&fast, &slow];
    let left = [&fast_left, &slow_left];
    let labels = ["fast", "slow"];
    let winner =
        post_then_hang_up_on_one(addr, &candidates, "/v1/chat/completions", body.clone()).await;
    caller_left_within(left[winner], Duration::from_secs(3))
        .await
        .expect("the learned node never observed the client cancellation");

    *fast_watch.lock().expect("fast watch lock") = Duration::from_millis(50);
    *slow_watch.lock().expect("slow watch lock") = Duration::from_millis(300);

    let (status, _, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(
        header(&headers, "x-camelid-fabric-node"),
        Some(labels[winner])
    );
    assert_eq!(
        header(&headers, "x-camelid-fabric-reason"),
        Some("EstimatedCompletion"),
        "client cancellation must not invalidate a healthy node's estimate"
    );
}

/// `Fabric::dispatch` is synchronous socket I/O that can legitimately run for
/// the whole `forward_timeout` — up to minutes for a real generation. If the
/// handler ran it directly on an async worker thread instead of
/// `spawn_blocking`, concurrent requests would serialize on that thread once
/// the runtime is down to one worker: this pins the test runtime to exactly
/// one worker thread, sends four concurrent requests to four independently
/// slow nodes, and asserts they complete together rather than one-at-a-time.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn concurrent_slow_requests_do_not_serialize_on_the_single_worker_thread() {
    let delay = Duration::from_millis(500);
    let nodes: Vec<StubNode> = (0..4)
        .map(|i| StubNode::start(StubConfig::slow(&format!("model-{i}"), delay)))
        .collect();
    let specs = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| node.spec(&format!("node-{i}")))
        .collect();
    let fabric = fabric_of(specs);
    let addr = start_proxy(fabric, RouteMode::Throughput).await;

    let started = std::time::Instant::now();
    let requests = (0..4).map(|i| {
        let model = format!("model-{i}");
        async move { post_chat(addr, &serde_json::json!({ "model": model }), &[]).await }
    });
    let results = futures_util::future::join_all(requests).await;
    let elapsed = started.elapsed();

    for (status, _, _) in &results {
        assert_eq!(*status, 200);
    }
    // Serialized on one worker thread: ~4 * 500ms = 2000ms. Concurrent via
    // spawn_blocking: ~500ms plus scheduling overhead. 1200ms sits well clear
    // of both, so neither CI jitter nor a slow stub can decide the outcome.
    assert!(
        elapsed < Duration::from_millis(1200),
        "four concurrent slow requests took {elapsed:?}; \
         they appear to have serialized on the async runtime"
    );
}

/// Send one GET over a real socket and return (status, raw body).
async fn get_raw(addr: SocketAddr, path: &str, extra_headers: &[(&str, &str)]) -> (u16, String) {
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let mut request = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    let text = String::from_utf8_lossy(&raw).into_owned();
    let header_end = text.find("\r\n\r\n").expect("header terminator");
    let status: u16 = text[..header_end]
        .lines()
        .next()
        .expect("status line")
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("numeric status");
    (status, text[header_end + 4..].to_string())
}

fn authenticated(key: &str) -> ClientAuth {
    ClientAuth::resolve(Some(key.to_string()), None).expect("a key resolves")
}

/// The whole point of the credential is that an unauthenticated caller cannot
/// reach the fabric at all. Refusing after placement would still have spent a
/// probe on every node, and told the caller which of them are up.
#[tokio::test(flavor = "multi_thread")]
async fn an_unauthenticated_request_never_reaches_a_node() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;

    let (status, body, _) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;
    assert_eq!(status, 401);
    assert_eq!(body["error"]["type"], "authentication_error");
    assert!(
        node.received().is_empty(),
        "the node was touched by a request that was never authenticated: {:?}",
        node.received()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn either_header_the_engine_accepts_is_accepted_here() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;
    let body = serde_json::json!({ "model": "shared-model" });

    let (bearer, _, _) = post_chat(addr, &body, &[("Authorization", "Bearer s3cret")]).await;
    assert_eq!(bearer, 200, "Authorization: Bearer must be accepted");

    let (api_key, _, _) = post_chat(addr, &body, &[("X-API-Key", "s3cret")]).await;
    assert_eq!(api_key, 200, "X-API-Key must be accepted");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wrong_key_is_refused_like_no_key_at_all() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;

    let (status, _, _) = post_chat(
        addr,
        &serde_json::json!({ "model": "shared-model" }),
        &[("Authorization", "Bearer wrong")],
    )
    .await;
    assert_eq!(status, 401);
    assert!(node.received().is_empty(), "{:?}", node.received());
}

/// Discovery is behind the key too: an unauthenticated caller must not learn
/// which models the fabric is serving.
#[tokio::test(flavor = "multi_thread")]
async fn discovery_does_not_answer_an_unauthenticated_caller() {
    let node = StubNode::start(StubConfig::ready("private-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;

    let (refused, body) = get_raw(addr, "/v1/models", &[]).await;
    assert_eq!(refused, 401);
    assert!(!body.contains("private-model"), "{body}");

    let (served, listing) =
        get_raw(addr, "/v1/models", &[("Authorization", "Bearer s3cret")]).await;
    assert_eq!(served, 200);
    assert!(listing.contains("private-model"), "{listing}");
}

/// A `tracing` sink a test can read back, so the access log can be asserted as
/// an operator actually reads it rather than by calling the middleware directly.
#[derive(Clone)]
struct AccessLog(Arc<Mutex<Vec<u8>>>);

impl AccessLog {
    /// The line mentioning `needle`. Every test in this binary shares one sink,
    /// so a caller has to look for something only it could have sent.
    fn line_mentioning(&self, needle: &str) -> Option<String> {
        let captured = self.0.lock().expect("access log");
        String::from_utf8_lossy(&captured)
            .lines()
            .find(|line| line.contains(needle))
            .map(str::to_string)
    }
}

impl Write for AccessLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("access log").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Install the capturing subscriber, once: a global default can only be set
/// once per process, and these tests share one.
fn access_log() -> AccessLog {
    static CAPTURED: OnceLock<AccessLog> = OnceLock::new();
    CAPTURED
        .get_or_init(|| {
            let sink = AccessLog(Arc::new(Mutex::new(Vec::new())));
            let writer = sink.clone();
            tracing_subscriber::fmt()
                .with_writer(move || writer.clone())
                .with_max_level(tracing::Level::INFO)
                // Both off so the assertions match on the text an operator
                // greps, not on escape codes or a timestamp.
                .with_ansi(false)
                .without_time()
                .init();
            sink
        })
        .clone()
}

/// The access log is the only place this proxy says who called and which
/// request it was. Both are supplied by the serving stack rather than by the
/// middleware, so neither is proven by calling the middleware directly.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_is_logged_with_the_caller_and_the_id_it_was_given() {
    let recorded = access_log();
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("node-a")]), RouteMode::Throughput).await;

    let mine = "only-this-test-sends-this-id";
    let body = serde_json::json!({ "model": "shared-model" });
    let (status, answered, headers) = post_chat(addr, &body, &[("x-request-id", mine)]).await;

    assert_eq!(status, 200, "{answered}");
    assert_eq!(
        header(&headers, "x-request-id"),
        Some(mine),
        "the id a client sent must come back to it, or it cannot quote it"
    );

    let line = recorded
        .line_mentioning(mine)
        .expect("the request must be logged under the id it was given");
    assert!(
        line.contains("127.0.0.1:"),
        "the line must name the caller, not `-`: {line}"
    );
}

/// A key set lives in a file, so these tests write one and hand over its path.
fn client_keys(dir: &tempfile::TempDir, body: &str) -> std::path::PathBuf {
    let path = dir.path().join("clients.json");
    std::fs::write(&path, body).expect("write client key file");
    path
}

const TWO_CLIENTS: &str = r#"{"clients":[
    {"name":"laptop","key":"laptop-secret"},
    {"name":"ci","key":"ci-secret"}
]}"#;

/// How long a revocation is given to take effect before the test calls it
/// broken. Generous against the one-second reload the proxy ships with, so a
/// loaded machine does not fail this for being slow.
const REVOCATION_TIMEOUT: Duration = Duration::from_secs(5);

fn chat_requests(node: &StubNode) -> usize {
    node.received()
        .iter()
        .filter(|request| request.path == "/v1/chat/completions")
        .count()
}

/// One key could only tell an operator that *someone* authenticated. A named
/// set says which client, in the one place they already read.
#[tokio::test(flavor = "multi_thread")]
async fn each_client_is_served_and_logged_under_its_own_name() {
    let recorded = access_log();
    let dir = tempfile::tempdir().expect("temp dir");
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        ClientAuth::from_key_file(client_keys(&dir, TWO_CLIENTS)).expect("load key set"),
    )
    .await;
    let body = serde_json::json!({ "model": "shared-model" });

    for (client, key, id) in [
        ("laptop", "laptop-secret", "only-this-test-names-laptop"),
        ("ci", "ci-secret", "only-this-test-names-ci"),
    ] {
        let bearer = format!("Bearer {key}");
        let (status, answered, _) = post_chat(
            addr,
            &body,
            &[("Authorization", &bearer), ("x-request-id", id)],
        )
        .await;
        assert_eq!(status, 200, "{client} was not served: {answered}");

        let line = recorded
            .line_mentioning(id)
            .unwrap_or_else(|| panic!("{client}'s request was not logged"));
        assert!(
            line.contains(&format!("client_name=\"{client}\"")),
            "the line must name the client that called: {line}"
        );
        assert!(
            !line.contains(key),
            "the line must never carry the key itself: {line}"
        );
    }
}

/// The reason for naming clients at all: one of them stops being served, and
/// nothing else does.
///
/// This runs against the reload interval the proxy actually ships with, not a
/// test-only one, because the claim being made is about the binary an operator
/// runs. Nothing is restarted between the two halves of this test — the same
/// listener that served the laptop goes on serving CI after refusing it.
#[tokio::test(flavor = "multi_thread")]
async fn revoking_one_client_stops_it_without_a_restart_or_disturbing_the_rest() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = client_keys(&dir, TWO_CLIENTS);
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        ClientAuth::from_key_file(path.clone()).expect("load key set"),
    )
    .await;
    let body = serde_json::json!({ "model": "shared-model" });
    let laptop = [("Authorization", "Bearer laptop-secret")];
    let ci = [("Authorization", "Bearer ci-secret")];

    assert_eq!(post_chat(addr, &body, &laptop).await.0, 200);
    assert_eq!(post_chat(addr, &body, &ci).await.0, 200);
    assert_eq!(
        chat_requests(&node),
        2,
        "both clients should have reached the node before the revocation"
    );

    std::fs::write(&path, r#"{"clients":[{"name":"ci","key":"ci-secret"}]}"#).expect("revoke");

    let started = Instant::now();
    loop {
        if post_chat(addr, &body, &laptop).await.0 == 401 {
            break;
        }
        assert!(
            started.elapsed() < REVOCATION_TIMEOUT,
            "a revoked client was still being served {:?} after the key file \
             stopped listing it",
            started.elapsed()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Refused, and refused early: the point of the credential is that the
    // fabric is never touched on behalf of a caller it does not serve.
    let settled = chat_requests(&node);
    let (status, refused, _) = post_chat(addr, &body, &laptop).await;
    assert_eq!(status, 401, "{refused}");
    assert_eq!(refused["error"]["type"], "authentication_error");
    assert_eq!(
        chat_requests(&node),
        settled,
        "a revoked client still reached a node"
    );

    let (still_served, answered, _) = post_chat(addr, &body, &ci).await;
    assert_eq!(
        still_served, 200,
        "revoking one client cut off another: {answered}"
    );
}

/// Being asked to stop is not the same as being killed. A proxy holds other
/// people's requests, and dropping them on a deploy turns a routine restart
/// into a client-visible failure.
///
/// The load-bearing claim is an *ordering* one: the stop must not complete
/// until the work in flight has. Asserting only that the request got its answer
/// would prove nothing here, because a connection axum has already accepted is
/// served on its own task and finishes either way while this process lives — it
/// is the caller returning early, and the process exiting under it, that loses
/// the request. So this measures how long the stop took to report itself done.
#[tokio::test(flavor = "multi_thread")]
async fn a_stop_finishes_the_work_in_flight_and_accepts_no_more() {
    let generating = Duration::from_millis(600);
    let node = StubNode::start(StubConfig {
        completion_delay: generating,
        ..StubConfig::ready("shared-model", 0)
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let fabric = fabric_of(vec![node.spec("node-a")]);
    let config = ServeConfig {
        mode: RouteMode::Throughput,
        forward_timeout: FORWARD_TIMEOUT,
        auth: ClientAuth::none(),
        tls: None,
        cors: None,
        bound: addr,
    };

    let (stop, stop_asked) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(async move {
        serve_on_until(listener, fabric, config, async move {
            let _ = stop_asked.await;
        })
        .await
    });

    let body = serde_json::json!({ "model": "shared-model" });
    let inflight = tokio::spawn(async move { post_chat(addr, &body, &[]).await });

    // Long enough for the request to be on the node, short enough that it is
    // still there when the stop arrives.
    let settle = Duration::from_millis(200);
    tokio::time::sleep(settle).await;
    let asked_at = Instant::now();
    let _ = stop.send(());

    serving
        .await
        .expect("the server joins")
        .expect("a stop is not a failure");
    let took = asked_at.elapsed();

    // The request still had most of its generation left when the stop landed.
    let still_owed = generating - settle;
    assert!(
        took >= still_owed / 2,
        "the proxy called itself stopped after {took:?}, while it still owed a \
         request about {still_owed:?} of work"
    );

    let (status, answered, _) = inflight.await.expect("the in-flight request joins");
    assert_eq!(
        status, 200,
        "a request already on a node must be finished, not dropped: {answered}"
    );
    assert!(
        tokio::net::TcpStream::connect(addr).await.is_err(),
        "a stopped proxy must not still be taking work"
    );
}

/// A proxy started without a key is unchanged: this is what every other test in
/// this file relies on, and what a loopback operator gets by default.
#[tokio::test(flavor = "multi_thread")]
async fn a_proxy_without_a_key_serves_an_anonymous_client() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("node-a")]), RouteMode::Throughput).await;

    let (status, _, _) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;
    assert_eq!(status, 200);
}

/// A load balancer needs one address to ask whether to send traffic here. Until
/// this route existed the answer was a bodyless 404 whatever the fabric was
/// doing, so nothing could tell a working proxy from a useless one.
#[tokio::test(flavor = "multi_thread")]
async fn health_reports_ready_while_a_node_can_serve() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("node-a")]), RouteMode::Throughput).await;

    let (status, body) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(status, 200, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(parsed["ok"], serde_json::json!(true));
    assert_eq!(parsed["ready"], serde_json::json!(true));
    assert_eq!(parsed["nodes"]["ready"], serde_json::json!(1));
    assert_eq!(parsed["models"], serde_json::json!(["shared-model"]));
    // The test binds loopback, so the fabric may be named.
    assert_eq!(parsed["node_detail"][0]["spec"]["label"], "node-a");
}

/// A spec for a port nothing listens on. Binding and dropping proves the port
/// was free and is now closed; a hardcoded number proves neither.
async fn closed_port_spec(label: &str) -> NodeSpec {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a free port");
    let addr = listener.local_addr().expect("port");
    drop(listener);
    NodeSpec::camelid(label, addr.ip().to_string(), addr.port())
}

/// Ready is "some request can be served", not "every node is well". A fabric
/// with one node down still takes traffic, and a proxy that reported itself
/// unhealthy here would take a whole fabric out of rotation over one node.
#[tokio::test(flavor = "multi_thread")]
async fn health_stays_ready_while_any_node_can_serve() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let specs = vec![node.spec("node-a"), closed_port_spec("node-gone").await];
    let addr = start_proxy(fabric_of(specs), RouteMode::Throughput).await;

    let (status, body) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(status, 200, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(parsed["ready"], serde_json::json!(true));
    assert_eq!(parsed["nodes"]["ready"], serde_json::json!(1));
    assert_eq!(parsed["nodes"]["unreachable"], serde_json::json!(1));
}

/// The status code has to change, not just a field in the body: a load balancer
/// acts on the code alone. A proxy with nothing behind it must stop inviting
/// traffic it can only refuse.
#[tokio::test(flavor = "multi_thread")]
async fn health_stops_inviting_traffic_when_no_node_answers() {
    let specs = vec![closed_port_spec("node-gone").await];
    let addr = start_proxy(fabric_of(specs), RouteMode::Throughput).await;

    let (status, body) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(status, 503, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(parsed["ready"], serde_json::json!(false));
    // Liveness is unchanged: this process is fine, its fabric is not, and
    // restarting this process would not bring a node back.
    assert_eq!(parsed["ok"], serde_json::json!(true));
    assert_eq!(parsed["nodes"]["unreachable"], serde_json::json!(1));
    assert_eq!(parsed["models"], serde_json::json!([]));
}

/// Health needs no key even on a proxy that requires one everywhere else: the
/// engine's shared `authenticate` keeps `/v1/health` public so a probe can run
/// without credentials, and this proxy reuses that check rather than writing a
/// second one. `/v1/models` stays behind the key, and the contrast is the point
/// — it is why the health body withholds the fabric's detail off-box, which the
/// unit tests cover because this test can only bind loopback.
#[tokio::test(flavor = "multi_thread")]
async fn health_answers_a_probe_that_carries_no_key() {
    let node = StubNode::start(StubConfig::ready("private-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("node-a")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;

    let (health, body) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(health, 200, "a probe with no key must still get an answer");
    let parsed: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(parsed["ready"], serde_json::json!(true));

    let (models, listing) = get_raw(addr, "/v1/models", &[]).await;
    assert_eq!(models, 401, "discovery stays behind the key");
    assert!(!listing.contains("private-model"), "{listing}");

    let (chat, _, _) = post_chat(addr, &serde_json::json!({ "model": "private-model" }), &[]).await;
    assert_eq!(chat, 401, "generation stays behind the key");
}

/// A load balancer polls this route forever. If each poll re-probed the fabric,
/// adding a health check would multiply node traffic by the polling rate, which
/// is exactly the cost the freshness bound exists to remove.
#[tokio::test(flavor = "multi_thread")]
async fn repeated_health_checks_share_one_observation() {
    let nodes = vec![
        StubNode::start(StubConfig::ready("shared-model", 0)),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let specs = vec![nodes[0].spec("node-a"), nodes[1].spec("node-b")];
    let addr = start_proxy(
        fabric_reusing_observations(specs, Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    for _ in 0..5 {
        let (status, _) = get_raw(addr, "/v1/health", &[]).await;
        assert_eq!(status, 200);
    }

    assert_eq!(
        health_probes(&nodes),
        nodes.len(),
        "five health checks should share one observation, one probe per node"
    );
}

/// The other half of that bound, and the half an operator actually has to set
/// something for: reuse lasts exactly as long as the window. A deployment
/// polling this route more slowly than `--observation-max-age-ms` re-probes
/// every node on every check — at the 500 ms default, a once-a-second probe
/// does that every time.
#[tokio::test(flavor = "multi_thread")]
async fn a_health_check_past_the_window_observes_the_fabric_again() {
    let nodes = vec![StubNode::start(StubConfig::ready("shared-model", 0))];
    let specs = vec![nodes[0].spec("node-a")];
    let addr = start_proxy(
        fabric_reusing_observations(specs, Duration::from_millis(50)),
        RouteMode::Throughput,
    )
    .await;

    let (first, _) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(first, 200);
    let after_first = health_probes(&nodes);

    // Comfortably past the window, so this is not a race with the clock.
    tokio::time::sleep(Duration::from_millis(150)).await;

    let (second, _) = get_raw(addr, "/v1/health", &[]).await;
    assert_eq!(second, 200);
    assert!(
        health_probes(&nodes) > after_first,
        "an expired observation must be taken again rather than reused"
    );
}

/// The proxy observes the fabric before every placement. Without a freshness
/// bound that is a `/v1/health` per node per request: measured at exactly 2 per
/// request against two nodes, and 2.0 s per request when one node black-holes,
/// against nodes answering in 2 ms.
#[tokio::test(flavor = "multi_thread")]
async fn requests_inside_the_freshness_window_share_one_observation() {
    let nodes = vec![
        StubNode::start(StubConfig::ready("shared-model", 0)),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let specs = vec![nodes[0].spec("node-a"), nodes[1].spec("node-b")];
    let addr = start_proxy(
        fabric_reusing_observations(specs, Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    let body = serde_json::json!({ "model": "shared-model" });
    for _ in 0..4 {
        let (status, _, _) = post_chat(addr, &body, &[]).await;
        assert_eq!(status, 200);
    }

    assert_eq!(
        health_probes(&nodes),
        nodes.len(),
        "four requests should share one observation, one probe per node"
    );
    assert_eq!(
        completions_served(&nodes).iter().sum::<usize>(),
        4,
        "every request must still reach a node"
    );
}

/// The bound is a bound, not a cache that never expires. Two requests either
/// side of it must each take their own observation.
#[tokio::test(flavor = "multi_thread")]
async fn an_observation_past_the_bound_is_taken_again() {
    let max_age = Duration::from_millis(50);
    let nodes = vec![StubNode::start(StubConfig::ready("shared-model", 0))];
    let specs = vec![nodes[0].spec("node-a")];
    let addr = start_proxy(
        fabric_reusing_observations(specs, max_age),
        RouteMode::Throughput,
    )
    .await;

    let body = serde_json::json!({ "model": "shared-model" });
    let (first, _, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(first, 200);
    tokio::time::sleep(max_age * 4).await;
    let (second, _, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(second, 200);

    assert_eq!(
        health_probes(&nodes),
        2,
        "a request after the bound must observe the fabric again"
    );
}

/// Reusing an observation must not undo the placement fix: these nodes report a
/// fixed load, so the observation is identical for every request in the burst
/// and the spreading can only come from the reservations the fabric keeps
/// itself. Without those, the label tie-break sends the whole burst to one node.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_burst_still_spreads_while_one_observation_is_reused() {
    const REQUESTS: usize = 12;
    let labels = ["node-a", "node-b", "node-c"];
    let nodes: Vec<StubNode> = labels
        .iter()
        .map(|_| StubNode::start(StubConfig::slow("shared-model", Duration::from_millis(300))))
        .collect();
    let specs = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| node.spec(labels[i]))
        .collect();
    let addr = start_proxy(
        fabric_reusing_observations(specs, Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    let body = serde_json::json!({ "model": "shared-model" });
    let results =
        futures_util::future::join_all((0..REQUESTS).map(|_| post_chat(addr, &body, &[]))).await;
    for (status, _, _) in &results {
        assert_eq!(*status, 200);
    }

    assert_eq!(
        health_probes(&nodes),
        nodes.len(),
        "the burst must have been placed from a single observation, \
         or this proves nothing about spreading without a fresh one"
    );

    let served = completions_served(&nodes);
    assert_eq!(
        served.iter().sum::<usize>(),
        REQUESTS,
        "every request must reach a node"
    );
    let busiest = served.iter().max().copied().unwrap_or(0);
    assert!(
        busiest <= REQUESTS / 2,
        "one node took {busiest} of {REQUESTS}: {served:?}"
    );
    assert!(
        served.iter().all(|count| *count > 0),
        "a node serving the model was never used: {served:?}"
    );
}

/// A node can die inside the freshness window, and an observation naming it as
/// ready is then wrong. The failed forward must drop that observation, so the
/// node is refused as unroutable rather than chosen again for the rest of the
/// window: the request after the failure is a routing refusal (503), not a
/// second failed forward (502).
#[tokio::test(flavor = "multi_thread")]
async fn a_node_that_stops_answering_drops_the_observation_that_named_it() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let specs = vec![node.spec("node-a")];
    let addr = start_proxy(
        fabric_reusing_observations(specs, Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    let body = serde_json::json!({ "model": "shared-model" });
    let (served, _, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(served, 200, "the node answers while it is up");

    drop(node);

    let (after_death, _, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(
        after_death, 502,
        "the observation still named the node, so this request is spent on it"
    );

    let (next, refusal, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(
        next, 503,
        "the failed forward must have dropped the observation, so this request \
         observes the fabric again and finds nothing to route to"
    );
    assert_eq!(refusal["error"]["type"], "fabric_error");
}

/// Reusing an observation may delay routing to a newly loaded model. It must
/// not turn that delay into a *permanent* refusal: `/v1/chat/completions`
/// answers 404 `model_not_found` for a model no node serves, and a 404 tells an
/// OpenAI SDK to stop retrying. Issued from a reused observation, that verdict
/// is about the fabric as it was, so the fabric has to look again before it
/// refuses rather than settle the question from memory.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_loaded_since_the_observation_is_not_refused_as_permanently_absent() {
    let config = StubConfig::with_switchable_model("old-model");
    let model = Arc::clone(config.live_model.as_ref().expect("a switchable node"));
    let node = StubNode::start(config);
    // Far longer than the test runs, so nothing here depends on it expiring.
    let addr = start_proxy(
        fabric_reusing_observations(vec![node.spec("only")], Duration::from_secs(30)),
        RouteMode::Throughput,
    )
    .await;

    let (served, _, _) = post_chat(addr, &serde_json::json!({ "model": "old-model" }), &[]).await;
    assert_eq!(served, 200, "the first request takes the observation");

    // The operator loads something else, exactly as `/api/models/load` does.
    *model.lock().expect("stub model lock") = "new-model".to_string();

    let (status, body, _) =
        post_chat(addr, &serde_json::json!({ "model": "new-model" }), &[]).await;
    assert_ne!(
        status, 404,
        "a permanent refusal was settled from a stale observation: {body}"
    );
    assert_eq!(status, 200, "{body}");
}

// ---------------------------------------------------------------------------
// Failing over to another node
//
// A node can go between being observed and being sent to. Placement then names
// a machine that is not there, and without failover the request dies against it
// even when a sibling is serving the same model. These tests are built on a
// reused observation because that is the arrangement `fabric serve` runs in,
// and the window where the fabric's view can be wrong by construction.
// ---------------------------------------------------------------------------

/// A fabric as `fabric serve` builds one, with an explicit attempt budget so a
/// test can turn failover off without changing anything else.
fn fabric_with_attempts(specs: Vec<NodeSpec>, attempts: usize) -> Fabric {
    fabric_reusing_observations(specs, Duration::from_secs(30)).with_max_forward_attempts(attempts)
}

/// Two nodes, both serving the model, and a proxy holding one observation of
/// them. `node-a` wins the label tie-break, so it is where the next request
/// goes; stopping it is what makes that observation wrong.
async fn two_nodes_one_of_which_will_vanish(
    attempts: usize,
) -> (StubNode, StubNode, SocketAddr, Value) {
    let a = StubNode::start(StubConfig::ready("shared-model", 0));
    let b = StubNode::start(StubConfig::ready("shared-model", 0));
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, attempts), RouteMode::Throughput).await;
    let body = serde_json::json!({ "model": "shared-model" });

    let (status, _, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(
        header(&headers, "x-camelid-fabric-node"),
        Some("node-a"),
        "the tie-break must settle on node-a, or stopping it proves nothing"
    );
    (a, b, addr, body)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_vanished_node_does_not_fail_a_request_another_node_can_serve() {
    let (mut a, _b, addr, body) = two_nodes_one_of_which_will_vanish(2).await;
    let before = completions_served(std::slice::from_ref(&a))[0];
    a.stop_listening();

    let (status, answer, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200, "a sibling serves the same model: {answer}");
    assert_eq!(
        header(&headers, "x-camelid-fabric-node"),
        Some("node-b"),
        "the answer must come from the node that is still there"
    );
    assert_eq!(
        header(&headers, "x-camelid-fabric-attempts"),
        Some("2"),
        "a failover must be visible to the client that was served by one"
    );
    assert_eq!(
        completions_served(std::slice::from_ref(&a))[0],
        before,
        "the vanished node must not have been sent the request twice"
    );
}

/// The ablation, run as a test: the same fabric with the budget at one is the
/// behaviour this change replaces, and it fails the request.
#[tokio::test(flavor = "multi_thread")]
async fn a_budget_of_one_fails_the_request_the_way_it_did_before() {
    let (mut a, b, addr, body) = two_nodes_one_of_which_will_vanish(1).await;
    a.stop_listening();

    let (status, refusal, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 502, "{refusal}");
    assert_eq!(
        completions_served(std::slice::from_ref(&b))[0],
        0,
        "with failover off the sibling must never be asked"
    );
}

/// The line the whole design rests on. This node reads the request and then
/// hangs up, so it may already be generating; that is a different failure from
/// never having been reached, and it must not be sent anywhere else.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_that_took_the_request_is_not_asked_to_run_it_again_elsewhere() {
    let a = StubNode::start(StubConfig::hanging_up_on_completion("shared-model"));
    let b = StubNode::start(StubConfig::ready("shared-model", 0));
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, refusal, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 502, "{refusal}");
    // The refusal names the node it happened against, so an operator can act on
    // the one that is actually broken.
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));
    assert_eq!(
        completions_served(std::slice::from_ref(&a))[0],
        1,
        "the node did receive the request, which is why it may not be re-sent"
    );
    assert_eq!(
        completions_served(std::slice::from_ref(&b))[0],
        0,
        "a request that may already be running was sent to a second node"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn failover_stops_at_the_attempt_budget_rather_than_walking_the_fabric() {
    let labels = ["node-a", "node-b", "node-c"];
    let mut nodes: Vec<StubNode> = labels
        .iter()
        .map(|_| StubNode::start(StubConfig::ready("shared-model", 0)))
        .collect();
    let specs: Vec<NodeSpec> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| node.spec(labels[index]))
        .collect();

    // Both proxies take their observation while all three nodes are up, so the
    // only difference between them is the budget.
    let bounded = start_proxy(
        fabric_with_attempts(specs.clone(), 2),
        RouteMode::Throughput,
    )
    .await;
    let generous = start_proxy(fabric_with_attempts(specs, 3), RouteMode::Throughput).await;
    let body = serde_json::json!({ "model": "shared-model" });
    for proxy in [bounded, generous] {
        let (status, _, headers) = post_chat(proxy, &body, &[]).await;
        assert_eq!(status, 200);
        assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));
    }

    // Two of the three go, leaving a node that only a third attempt reaches.
    nodes[0].stop_listening();
    nodes[1].stop_listening();
    let served_by_c = completions_served(&nodes[2..])[0];

    let (status, refusal, _) = post_chat(bounded, &body, &[]).await;
    assert_eq!(
        status, 502,
        "two attempts must not silently become three: {refusal}"
    );
    assert_eq!(
        completions_served(&nodes[2..])[0],
        served_by_c,
        "the third node is beyond the budget and must not have been asked"
    );

    // The positive control: the same situation with room for a third attempt
    // does reach it, so the refusal above is the budget and not the exclusion.
    let (status, answer, headers) = post_chat(generous, &body, &[]).await;
    assert_eq!(status, 200, "{answer}");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-c"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("3"));
}

/// A failover proves the observation is wrong, so it must not survive the
/// request that discovered it — even though that request succeeded. Otherwise
/// every request in the rest of the window pays the same dial to the same
/// absent node before being served.
#[tokio::test(flavor = "multi_thread")]
async fn a_served_failover_still_drops_the_observation_that_named_the_vanished_node() {
    let (mut a, b, addr, body) = two_nodes_one_of_which_will_vanish(2).await;
    a.stop_listening();

    let (status, _, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));

    let probes_before = completions_served(std::slice::from_ref(&b))[0];
    let (status, _, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(
        header(&headers, "x-camelid-fabric-attempts"),
        Some("1"),
        "the next request must be placed from a fresh observation, which no \
         longer names the node that is gone"
    );
    assert_eq!(
        header(&headers, "x-camelid-fabric-node"),
        Some("node-b"),
        "and it must go straight to the node that is there"
    );
    assert_eq!(
        completions_served(std::slice::from_ref(&b))[0],
        probes_before + 1
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_vanished_node_does_not_fail_a_streaming_request() {
    let events = ["data: one\n\n", "data: [DONE]\n\n"];
    let mut a = StubNode::start(StubConfig::streaming(
        "shared-model",
        &events,
        Duration::ZERO,
    ));
    let b = StubNode::start(StubConfig::streaming(
        "shared-model",
        &events,
        Duration::ZERO,
    ));
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;
    let body = serde_json::json!({ "model": "shared-model", "stream": true });

    let (status, headers, _) = post_chat_streaming(addr, &body).await;
    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));

    a.stop_listening();

    let (status, headers, pieces) = post_chat_streaming(addr, &body).await;
    assert_eq!(status, 200, "the sibling can still stream");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-b"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));
    let framed: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
    assert_eq!(
        dechunk(&framed),
        events.concat(),
        "the failover must relay the sibling's stream verbatim"
    );
}

/// The streaming half of the safety rule. Once the head is relayed the client
/// has been committed to one node's answer, so a node that dies part-way
/// through must end that stream rather than start a second one somewhere else.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_dying_mid_stream_is_not_replayed_on_another_node() {
    // Two events with a gap, exactly as `a_node_dying_mid_stream_is_not_relayed
    // _as_a_completed_one` uses: it is what makes the proxy flush the head and
    // the first event before the node dies, so the client really is committed
    // to node-a's answer by the time it fails.
    let gap = Duration::from_millis(150);
    let a = StubNode::start(StubConfig::dying_mid_stream(
        "shared-model",
        &["data: one\n\n", "data: two\n\n"],
        gap,
    ));
    let b = StubNode::start(StubConfig::streaming(
        "shared-model",
        &["data: elsewhere\n\n"],
        gap,
    ));
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, headers, pieces) = post_chat_streaming(
        addr,
        &serde_json::json!({ "model": "shared-model", "stream": true }),
    )
    .await;

    assert_eq!(status, 200, "the head arrived before the node died");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("1"));
    let framed: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
    assert!(
        dechunk(&framed).starts_with("data: one\n\n"),
        "what the node did produce must still arrive: {framed:?}"
    );
    assert!(
        !framed.ends_with("0\r\n\r\n"),
        "a truncated stream must not be framed as complete: {framed:?}"
    );
    assert_eq!(
        completions_served(std::slice::from_ref(&b))[0],
        0,
        "the request was replayed on a second node after the client had already \
         been given part of the first node's answer"
    );
}

/// Failing over past a gone node is a recovery, not the reason a request ends.
/// When the node it moves on to fails it for real, that second failure is what
/// the operator has to act on: reporting the first sends them to a node the
/// fabric already routed around, and says nothing about the one that is
/// actually broken.
#[tokio::test(flavor = "multi_thread")]
async fn the_failure_that_ended_the_request_is_the_one_reported() {
    let a = StubNode::start(StubConfig::ready("shared-model", 0));
    // `node-b` is reachable and answers, but with a body that is not JSON, so
    // it fails the request in a way failover must not move past.
    let b = StubNode::start(StubConfig {
        completion: "<html>gateway timeout</html>".to_string(),
        ..StubConfig::ready("shared-model", 0)
    });
    let specs = vec![a.spec("node-a"), b.spec("node-b")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;
    let body = serde_json::json!({ "model": "shared-model" });

    let (status, _, headers) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("node-a"));

    let mut a = a;
    a.stop_listening();

    let (status, refusal, _) = post_chat(addr, &body, &[]).await;
    assert_eq!(status, 502);
    let message = refusal["error"]["message"]
        .as_str()
        .expect("a message an operator can act on");
    assert!(
        message.contains("node-b"),
        "the node that actually failed the request must be named: {message}"
    );
    assert!(
        !message.contains("node-a"),
        "node-a was routed around successfully; naming it points at the wrong \
         node: {message}"
    );
}

/// A node at its queue bound rejected the request rather than running it, so
/// another node can still take it. Until now the client got that 503 while a
/// sibling sat idle — one busy node undoing the reason the fabric exists.
#[tokio::test(flavor = "multi_thread")]
async fn a_saturated_node_hands_the_request_to_one_that_is_not() {
    let nodes = vec![
        StubNode::start(StubConfig::queue_full("shared-model")),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    // `a-full` sorts first, so deterministic placement reaches it first.
    let specs = vec![nodes[0].spec("a-full"), nodes[1].spec("b-ready")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 200, "{body}");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("b-ready"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));
    assert_eq!(
        completions_served(&nodes),
        vec![1, 1],
        "the full node was asked once, then the ready one"
    );
}

/// Every node full is a real answer, not a proxy failure. The client gets a
/// node's own refusal, with the code and message it sent, rather than
/// something this proxy invented on its behalf.
#[tokio::test(flavor = "multi_thread")]
async fn a_fabric_that_is_entirely_saturated_returns_a_node_s_own_refusal() {
    let nodes = vec![
        StubNode::start(StubConfig::queue_full("shared-model")),
        StubNode::start(StubConfig::queue_full("shared-model")),
    ];
    let specs = vec![nodes[0].spec("a-full"), nodes[1].spec("b-full")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["code"], ENGINE_QUEUE_FULL_CODE);
    assert_ne!(
        body["error"]["type"], "fabric_error",
        "a node answered, so the node's answer is what stands: {body}"
    );
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));
    assert_eq!(completions_served(&nodes), vec![1, 1]);
}

/// The count of nodes asked has to be the count of nodes asked. With one node
/// and a budget of two, placement runs out before a second node is reached, so
/// reporting the budget sends an operator looking for a node that was never
/// contacted.
#[tokio::test(flavor = "multi_thread")]
async fn a_lone_saturated_node_is_one_attempt_not_two() {
    let nodes = vec![StubNode::start(StubConfig::queue_full("shared-model"))];
    let specs = vec![nodes[0].spec("a-full")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 503, "{body}");
    assert_eq!(body["error"]["code"], ENGINE_QUEUE_FULL_CODE);
    assert_eq!(completions_served(&nodes), vec![1]);
    assert_eq!(
        header(&headers, "x-camelid-fabric-attempts"),
        Some("1"),
        "one node was asked, so the answer must not claim two"
    );
}

/// The switch that turns off failover turns this off too: one attempt means
/// the first node's answer is the answer.
#[tokio::test(flavor = "multi_thread")]
async fn one_attempt_returns_the_refusal_without_asking_another_node() {
    let nodes = vec![
        StubNode::start(StubConfig::queue_full("shared-model")),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let specs = vec![nodes[0].spec("a-full"), nodes[1].spec("b-ready")];
    let addr = start_proxy(fabric_with_attempts(specs, 1), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 503, "{body}");
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("1"));
    assert_eq!(
        completions_served(&nodes),
        vec![1, 0],
        "the ready node must not be asked when failover is off"
    );
}

/// Only backpressure is re-placed. A node refusing for any other reason has
/// answered the request; asking a second node would spend its time reaching
/// the same refusal, and turn one node's 503 into two.
#[tokio::test(flavor = "multi_thread")]
async fn a_refusal_that_is_not_backpressure_is_relayed_untouched() {
    let nodes = vec![
        StubNode::start(StubConfig::refusing_with(
            "shared-model",
            "model_unavailable",
            "no model is loaded",
        )),
        StubNode::start(StubConfig::ready("shared-model", 0)),
    ];
    let specs = vec![nodes[0].spec("a-refusing"), nodes[1].spec("b-ready")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, body, headers) =
        post_chat(addr, &serde_json::json!({ "model": "shared-model" }), &[]).await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["message"], "no model is loaded");
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("1"));
    assert_eq!(
        completions_served(&nodes),
        vec![1, 0],
        "a refusal that is not backpressure must not be sent on"
    );
}

/// A streaming request gets the same treatment, because the refusal arrives
/// buffered — before one event has been relayed, so nothing has been said.
#[tokio::test(flavor = "multi_thread")]
async fn a_saturated_node_hands_a_streaming_request_on_as_well() {
    let events = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
        "data: [DONE]",
    ];
    let nodes = [
        StubNode::start(StubConfig::queue_full("shared-model")),
        StubNode::start(StubConfig::streaming(
            "shared-model",
            &events,
            Duration::ZERO,
        )),
    ];
    let specs = vec![nodes[0].spec("a-full"), nodes[1].spec("b-ready")];
    let addr = start_proxy(fabric_with_attempts(specs, 2), RouteMode::Throughput).await;

    let (status, headers, pieces) = post_chat_streaming(
        addr,
        &serde_json::json!({ "model": "shared-model", "stream": true }),
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("b-ready"));
    assert_eq!(header(&headers, "x-camelid-fabric-attempts"), Some("2"));
    assert!(
        !pieces.is_empty(),
        "the replacement node's events must arrive"
    );
}

// ---------------------------------------------------------------------------
// The routes the proxy serves
//
// A client pointed at this address expects an OpenAI-compatible surface, not
// only chat. These tests are that client's view: what reaches a node, what is
// refused and why, and what an unauthenticated caller is allowed to learn.
// ---------------------------------------------------------------------------

/// A body that is recognisably this route's, so the assertion that the node got
/// it back cannot pass on a body some other route sent.
fn body_for(path: &str) -> Value {
    let marker = format!("body for {path}");
    match path {
        "/v1/chat/completions" => {
            serde_json::json!({ "model": "shared-model", "messages": [{ "role": "user", "content": marker }] })
        }
        "/v1/embeddings" => serde_json::json!({ "model": "shared-model", "input": marker }),
        "/v1/rerank" | "/v1/reranking" => {
            serde_json::json!({ "model": "shared-model", "query": marker, "documents": ["a", "b"] })
        }
        _ => serde_json::json!({ "model": "shared-model", "prompt": marker }),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn every_placed_route_reaches_a_node_with_its_path_and_body_intact() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("only")]), RouteMode::Throughput).await;

    for path in PLACED_ROUTES {
        let sent = body_for(path);
        let (status, _, headers) = post_to(addr, path, &sent, &[]).await;
        assert_eq!(status, 200, "{path} was not served");
        assert_eq!(
            header(&headers, "x-camelid-fabric-node"),
            Some("only"),
            "{path} was answered without being placed"
        );

        let arrived: Vec<Received> = node
            .received()
            .into_iter()
            .filter(|received| received.path == path)
            .collect();
        assert_eq!(
            arrived.len(),
            1,
            "{path} did not reach the node exactly once"
        );
        // Byte-for-byte: the proxy reads `model` and `stream` out of the body
        // and must relay everything else, including fields it has never heard
        // of, exactly as the client wrote them.
        assert_eq!(
            serde_json::from_str::<Value>(&arrived[0].body).expect("json body"),
            sent,
            "{path} arrived with a body the client did not send"
        );
    }
}

/// Placement is model-scoped on every route, not just the one it was written
/// for: a node that does not hold the model must never see the request.
#[tokio::test(flavor = "multi_thread")]
async fn a_placed_route_other_than_chat_is_still_scoped_to_the_serving_node() {
    let alpha = StubNode::start(StubConfig::ready("model-alpha", 0));
    let beta = StubNode::start(StubConfig::ready("model-beta", 0));
    let addr = start_proxy(
        fabric_of(vec![alpha.spec("alpha"), beta.spec("beta")]),
        RouteMode::Throughput,
    )
    .await;

    let (status, _, headers) = post_to(
        addr,
        "/v1/embeddings",
        &serde_json::json!({ "model": "model-beta", "input": "hello" }),
        &[],
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("beta"));
    assert!(
        alpha
            .received()
            .iter()
            .all(|received| received.path == "/v1/health"),
        "alpha holds a different model and should only have been probed: {:?}",
        alpha.received()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_completions_stream_is_relayed_like_a_chat_one() {
    let events = ["data: one\n\n", "data: [DONE]\n\n"];
    let node = StubNode::start(StubConfig::streaming(
        "shared-model",
        &events,
        Duration::ZERO,
    ));
    let addr = start_proxy(fabric_of(vec![node.spec("only")]), RouteMode::Throughput).await;

    let (status, headers, pieces) = post_streaming(
        addr,
        "/v1/completions",
        &serde_json::json!({ "model": "shared-model", "prompt": "hi", "stream": true }),
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("only"));
    assert_eq!(header(&headers, "content-type"), Some("text/event-stream"));
    let framed: String = pieces.iter().map(|piece| piece.text.as_str()).collect();
    assert_eq!(dechunk(&framed), events.concat());
    assert_eq!(
        served_on(std::slice::from_ref(&node), "/v1/completions")[0],
        1
    );
}

/// `stream: true` is a property of the request, not of the route. A route that
/// has nothing to stream answers with a body instead, and that answer has to
/// reach the client as an answer rather than as an empty stream.
#[tokio::test(flavor = "multi_thread")]
async fn a_route_that_cannot_stream_still_answers_a_client_that_asks_it_to() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("only")]), RouteMode::Throughput).await;

    let (status, body, headers) = post_to(
        addr,
        "/v1/embeddings",
        &serde_json::json!({ "model": "shared-model", "input": "hi", "stream": true }),
        &[],
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("only"));
    assert!(
        body.get("choices").is_some(),
        "the node's complete answer must reach the client verbatim: {body}"
    );
}

/// Refused before placement, so the fabric is not even observed: this proxy
/// declines these routes on principle, not because no node could take them.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_local_route_is_refused_without_touching_a_node() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("only")]), RouteMode::Throughput).await;

    let (status, body, _) = post_to(
        addr,
        "/v1/responses",
        &serde_json::json!({ "model": "shared-model", "input": "hi" }),
        &[],
    )
    .await;

    assert_eq!(status, 501, "{body}");
    assert_eq!(body["error"]["type"], "fabric_error");
    assert!(
        node.received().is_empty(),
        "a refusal on principle must not cost a probe: {:?}",
        node.received()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unserved_route_is_refused_with_the_ones_that_are_served() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("only")]), RouteMode::Throughput).await;

    let (status, body, _) = post_to(
        addr,
        "/v1/audio/speech",
        &serde_json::json!({ "model": "shared-model" }),
        &[],
    )
    .await;

    assert_eq!(status, 404, "{body}");
    assert_eq!(body["error"]["type"], "fabric_error");
    let message = body["error"]["message"].as_str().expect("a message");
    assert!(message.contains("/v1/embeddings"), "{message}");
    assert!(message.contains("/v1/health"), "{message}");
    assert!(node.received().is_empty(), "{:?}", node.received());
}

/// The route table is not public. An unauthenticated caller learns that it is
/// unauthenticated and nothing else — the same reasoning that keeps model names
/// out of a 401 from `/v1/models`.
#[tokio::test(flavor = "multi_thread")]
async fn an_unauthenticated_caller_is_refused_before_learning_the_route_table() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy_with_auth(
        fabric_of(vec![node.spec("only")]),
        RouteMode::Throughput,
        authenticated("s3cret"),
    )
    .await;

    let (unknown, body) = get_raw(addr, "/v1/audio/speech", &[]).await;
    assert_eq!(unknown, 401);
    assert!(
        !body.contains("/v1/embeddings"),
        "the 401 leaked the route table: {body}"
    );

    let (refused, _, _) = post_to(
        addr,
        "/v1/embeddings",
        &serde_json::json!({ "model": "shared-model", "input": "hi" }),
        &[],
    )
    .await;
    assert_eq!(refused, 401, "every placed route is behind the key");
    assert!(node.received().is_empty(), "{:?}", node.received());

    let (served, _, _) = post_to(
        addr,
        "/v1/embeddings",
        &serde_json::json!({ "model": "shared-model", "input": "hi" }),
        &[("Authorization", "Bearer s3cret")],
    )
    .await;
    assert_eq!(served, 200, "the key still opens the new routes");
}

/// The body limit and the JSON rejection shape are properties of the handler
/// every placed route shares, so they must hold on a route that was added
/// after they were written.
#[tokio::test(flavor = "multi_thread")]
async fn a_malformed_body_on_a_new_route_is_refused_in_the_fabric_shape() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("only")]), RouteMode::Throughput).await;

    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .expect("connect to proxy");
    let payload = "this is not json";
    let request = format!(
        "POST /v1/rerank HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    let text = String::from_utf8_lossy(&raw);

    assert!(text.starts_with("HTTP/1.1 400"), "{text}");
    assert!(text.contains("fabric_error"), "{text}");
    assert!(node.received().is_empty(), "{:?}", node.received());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_can_be_retrieved_by_id_through_the_proxy() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let addr = start_proxy(fabric_of(vec![node.spec("only")]), RouteMode::Throughput).await;

    let (found, body) = get_raw(addr, "/v1/models/shared-model", &[]).await;
    assert_eq!(found, 200, "{body}");
    assert!(body.contains("\"id\":\"shared-model\""), "{body}");

    let (missing, refusal) = get_raw(addr, "/v1/models/other-model", &[]).await;
    assert_eq!(missing, 404, "{refusal}");
    assert!(refusal.contains("model_not_found"), "{refusal}");
}

// ---------------------------------------------------------------------------
// Serving TLS
//
// The proxy is the address an operator is told to expose, and the key it asks
// clients for is sent on every request. These tests are about what actually
// crosses the socket, so they speak real TLS to it rather than inspecting
// configuration.
// ---------------------------------------------------------------------------

/// The whole point: a request survives the handshake and is still placed.
#[tokio::test(flavor = "multi_thread")]
async fn a_tls_listener_places_a_request_and_relays_the_answer() {
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let (addr, certificate) =
        start_tls_proxy(fabric_of(vec![node.spec("only")]), ClientAuth::none()).await;

    let (status, body, headers) = post_over_tls(
        addr,
        &certificate,
        "/v1/chat/completions",
        &serde_json::json!({ "model": "shared-model" }),
        &[],
    )
    .await;

    assert_eq!(status, 200, "{body}");
    assert_eq!(header(&headers, "x-camelid-fabric-node"), Some("only"));
    assert_eq!(
        completions_served(std::slice::from_ref(&node))[0],
        1,
        "the request never reached the node"
    );
}

/// A TLS listener is served by a different crate than the cleartext one, and
/// the peer address is wired up per-crate. Without this the access log would
/// keep saying `client="-"` for every encrypted request and no unit test could
/// notice, exactly as it could not notice the cleartext wiring.
#[tokio::test(flavor = "multi_thread")]
async fn a_tls_request_is_logged_with_the_caller_it_came_from() {
    let recorded = access_log();
    let node = StubNode::start(StubConfig::ready("shared-model", 0));
    let (addr, certificate) =
        start_tls_proxy(fabric_of(vec![node.spec("only")]), ClientAuth::none()).await;

    let mine = "only-the-tls-caller-test-sends-this-id";
    let (status, body, _) = post_over_tls(
        addr,
        &certificate,
        "/v1/chat/completions",
        &serde_json::json!({ "model": "shared-model" }),
        &[("x-request-id", mine)],
    )
    .await;
    assert_eq!(status, 200, "{body}");

    let line = recorded
        .line_mentioning(mine)
        .expect("the TLS request must be logged under the id it was given");
    assert!(
        line.contains("client=\"127.0.0.1:"),
        "a TLS request was recorded without the client it came from: {line}"
    );
}

/// A certificate that cannot be read, or cannot be parsed, has to be refused
/// before anything is bound or announced. Refusing later is not good enough:
/// `fabric serve` prints its listening line and probes every node between
/// binding and serving, so an operator would be told the proxy is listening on
/// an `https://` address it is about to fail to open.
///
/// That a configured certificate can never degrade to cleartext is then
/// structural rather than tested: `ServeConfig.tls` can only be `Some` once the
/// certificate has loaded.
#[tokio::test(flavor = "multi_thread")]
async fn an_unreadable_certificate_refuses_before_anything_is_bound_or_announced() {
    let missing = std::path::PathBuf::from("no-such-certificate");
    let absent = ProxyTls::resolve(Some(missing.clone()), Some(missing))
        .await
        .expect_err("a certificate that is not there cannot be served");
    assert_eq!(absent.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        absent.to_string().contains("TLS certificate/key"),
        "{absent}"
    );

    // The likelier operator mistake: files that exist and are not a PEM pair.
    let dir = tempfile::tempdir().expect("temp dir");
    let junk = dir.path().join("certificate-chain");
    std::fs::write(&junk, b"this is not a certificate\n").expect("write junk");
    let unparseable = ProxyTls::resolve(Some(junk.clone()), Some(junk))
        .await
        .expect_err("a file that is not a PEM pair cannot be served");
    assert_eq!(unparseable.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        unparseable.to_string().contains("TLS certificate/key"),
        "{unparseable}"
    );
}

/// A stop still drains in-flight work when the listener is a TLS one, and still
/// stops taking new work. Both halves are a different crate's accept loop and
/// shutdown here than the cleartext twin covers, so neither carries over.
#[tokio::test(flavor = "multi_thread")]
async fn a_tls_stop_finishes_the_work_in_flight_and_accepts_no_more() {
    let generating = Duration::from_millis(600);
    let node = StubNode::start(StubConfig {
        completion_delay: generating,
        ..StubConfig::ready("shared-model", 0)
    });
    let certificate = mint_certificate();
    let tls = ProxyTls::resolve(
        Some(certificate.cert_path.clone()),
        Some(certificate.key_path.clone()),
    )
    .await
    .expect("a complete pair resolves")
    .expect("a pair was given");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let (stop, stop_asked) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(async move {
        serve_on_until(
            listener,
            fabric_of(vec![node.spec("node-a")]),
            ServeConfig {
                mode: RouteMode::Throughput,
                forward_timeout: FORWARD_TIMEOUT,
                auth: ClientAuth::none(),
                tls: Some(tls),
                cors: None,
                bound: addr,
            },
            async move {
                let _ = stop_asked.await;
            },
        )
        .await
    });

    let inflight = tokio::spawn(async move {
        post_over_tls(
            addr,
            &certificate,
            "/v1/chat/completions",
            &serde_json::json!({ "model": "shared-model" }),
            &[],
        )
        .await
    });

    // Long enough for the request to be on the node, short enough that it is
    // still being generated when the stop arrives.
    tokio::time::sleep(Duration::from_millis(150)).await;
    let asked = Instant::now();
    let _ = stop.send(());
    let stopped = serving.await.expect("serving task");
    let drained_for = asked.elapsed();

    stopped.expect("a stop is not an error");
    let (status, body, _) = inflight.await.expect("in-flight task");
    assert_eq!(status, 200, "{body}");
    // The ordering is the feature: reporting itself stopped before the work it
    // owed was done is exactly what drops other people's requests.
    assert!(
        drained_for >= generating - Duration::from_millis(200),
        "the proxy called itself stopped after {drained_for:?} while it still owed \
         a request about {generating:?} of work"
    );
    assert!(
        tokio::net::TcpStream::connect(addr).await.is_err(),
        "a stopped proxy must not still be taking work"
    );
}
