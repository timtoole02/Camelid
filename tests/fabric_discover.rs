//! End-to-end tests for discovery: `camelid fabric discover` and the three
//! routes `fabric serve --discovery` serves.
//!
//! The unit tests prove the rules. These prove the product: real sockets, the
//! real router, the real binary, and stubs that record **every byte they were
//! sent** — because the central claim of this feature is about what is *not*
//! on the wire, and a claim like that cannot rest on application logs.
//!
//! Every scan here passes `loopback: false` and `default_ports: false` and
//! names ephemeral ports. The test host runs real engines, and a suite that
//! swept its own machine's default ports would be neither hermetic nor polite.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use camelid::fabric::server::{serve_on, ClientAuth, DiscoveryConfig, ProxyCors, ServeConfig};
use camelid::fabric::{Fabric, MixedEngines, RouteMode};

/// The node file's own reload bound, plus enough slack to be sure a reload
/// happened rather than to race it.
const RELOAD: Duration = Duration::from_millis(1200);

/// A loopback stub that answers canned bodies and keeps every request byte it
/// was ever sent.
struct Stub {
    port: u16,
    seen: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

impl Stub {
    /// Everything this stub was ever sent, as one string.
    fn bytes(&self) -> String {
        self.seen.lock().expect("seen").join("\n")
    }
}

/// Start a stub answering `answers` (path, status, content-type, body).
fn stub(answers: &'static [(&'static str, u16, &'static str, &'static str)]) -> Stub {
    serving(move |path| {
        answers
            .iter()
            .find(|(candidate, ..)| *candidate == path)
            .map(|(_, status, kind, body)| (*status, *kind, (*body).to_string()))
    })
}

/// Start a stub whose answer for each path is decided by `answer`.
fn serving(
    answer: impl Fn(&str) -> Option<(u16, &'static str, String)> + Send + 'static,
) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
    let port = listener.local_addr().expect("stub addr").port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let recorded = Arc::clone(&seen);
    let stopping = Arc::clone(&stop);

    std::thread::spawn(move || {
        for incoming in listener.incoming() {
            if stopping.load(Ordering::SeqCst) {
                return;
            }
            let Ok(mut stream) = incoming else { return };
            let Some(head) = read_head(&mut stream) else {
                continue;
            };
            recorded.lock().expect("seen").push(head.clone());
            let path = head
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("/")
                .to_string();
            match answer(&path) {
                Some((status, kind, body)) => {
                    let _ = write!(
                        stream,
                        "HTTP/1.1 {status} X\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    );
                }
                None => {
                    let _ = write!(
                        stream,
                        "HTTP/1.1 404 X\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}"
                    );
                }
            }
        }
    });
    Stub { port, seen, stop }
}

/// A stub that accepts and never answers.
fn silent_stub() -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
    let port = listener.local_addr().expect("stub addr").port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let held = Arc::clone(&stop);
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        for incoming in listener.incoming() {
            if held.load(Ordering::SeqCst) {
                return;
            }
            match incoming {
                // Held open, never written to.
                Ok(stream) => kept.push(stream),
                Err(_) => return,
            }
        }
    });
    Stub { port, seen, stop }
}

/// A port nothing is listening on.
fn closed_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);
    port
}

fn read_head(stream: &mut TcpStream) -> Option<String> {
    stream
        .set_read_timeout(Some(Duration::from_millis(1500)))
        .ok()?;
    let mut raw = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&chunk[..read]);
        if raw.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    Some(String::from_utf8_lossy(&raw).into_owned())
}

const CAMELID_HEALTH: &str = r#"{"ok":true,"engine":"camelid","generation_ready":true,"version":"v0.7.2-551","active_model_id":"llama-3.2-1b-instruct","engine_queue_depth":0,"engine_queued_tasks":0}"#;

fn camelid_stub() -> Stub {
    stub(&[("/v1/health", 200, "application/json", CAMELID_HEALTH)])
}

/// A Camelid node started with an API key: 401 on every route but health.
fn keyed_camelid_stub() -> Stub {
    serving(|path| {
        Some(match path {
            "/v1/health" => (200, "application/json", CAMELID_HEALTH.to_string()),
            _ => (401, "application/json", r#"{"error":"unauthorized"}"#.to_string()),
        })
    })
}

fn ollama_stub() -> Stub {
    stub(&[
        ("/api/version", 200, "application/json", r#"{"version":"0.33.2"}"#),
        (
            "/api/tags",
            200,
            "application/json",
            r#"{"models":[{"name":"llama3.2:latest"}]}"#,
        ),
    ])
}

fn html_stub() -> Stub {
    serving(|_| Some((200, "text/html", "<!doctype html><title>files</title>".to_string())))
}

fn proxy_stub() -> Stub {
    stub(&[(
        "/v1/health",
        200,
        "application/json",
        r#"{"ok":true,"service":"camelid-fabric","version":"0.7.3","ready":true}"#,
    )])
}

fn nodes_file(dir: &tempfile::TempDir, body: &str) -> PathBuf {
    let path = dir.path().join("nodes");
    std::fs::write(&path, body).expect("write nodes file");
    path
}

fn hash_of(path: &Path) -> String {
    camelid::fabric::node_file_sha256(path).expect("the file exists")
}

/// The proxy, with discovery on unless `discovery` says otherwise.
async fn start_proxy(
    nodes: &Path,
    auth: ClientAuth,
    bearer: Option<&str>,
    discovery: Option<DiscoveryConfig>,
    cors: Option<ProxyCors>,
) -> SocketAddr {
    let fabric = Fabric::from_node_file(nodes.to_path_buf())
        .expect("the node file loads")
        .with_bearer(bearer)
        .with_timeout(Duration::from_millis(500));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let config = ServeConfig {
        mode: RouteMode::Throughput,
        forward_timeout: Duration::from_secs(5),
        auth,
        tls: None,
        cors,
        mixed: MixedEngines::default(),
        bound: addr,
        discovery,
    };
    tokio::spawn(async move {
        let _ = serve_on(listener, fabric, config).await;
    });
    addr
}

fn discovery_config(nodes: &Path, bearer_configured: bool, cors: &[&str]) -> DiscoveryConfig {
    DiscoveryConfig {
        nodes_file: nodes.to_path_buf(),
        cors_origins: cors
            .iter()
            .map(|origin| (*origin).to_string())
            .collect::<Vec<_>>()
            .into(),
        bearer_configured,
    }
}

/// One request to the proxy, with full control over the head.
async fn request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> (u16, Value) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let mut head = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\n");
    if !headers.iter().any(|(name, _)| name.eq_ignore_ascii_case("host")) {
        head.push_str(&format!("Host: {addr}\r\n"));
    }
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    if let Some(body) = body {
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).await.expect("write head");
    if let Some(body) = body {
        stream.write_all(body.as_bytes()).await.expect("write body");
    }

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    let text = String::from_utf8_lossy(&raw).into_owned();
    let split = text.find("\r\n\r\n").expect("header terminator");
    let status: u16 = text[..split]
        .lines()
        .next()
        .expect("status line")
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("numeric status");
    let body = serde_json::from_str(&text[split + 4..]).unwrap_or(Value::Null);
    (status, body)
}

/// A scan body naming exactly these ports on loopback, and nothing else.
fn scope_over(ports: &[u16]) -> String {
    serde_json::json!({
        "hosts": ["127.0.0.1"],
        "ports": ports,
        "loopback": false,
        "default_ports": false,
    })
    .to_string()
}

const JSON: (&str, &str) = ("content-type", "application/json");
const KEY: &str = "client-key-77c1";
const BEARER: &str = "fabric-secret-3f9a";

fn keyed() -> ClientAuth {
    ClientAuth::resolve(Some(KEY.to_string()), None).expect("a key resolves")
}

fn with_key() -> [(&'static str, &'static str); 2] {
    [JSON, ("authorization", "Bearer client-key-77c1")]
}

// ---- the exit criterion, offline --------------------------------------------

/// The offline form of the R3 exit criterion: a Camelid node, an Ollama node
/// and an unrelated HTTP service are each classified correctly, and the
/// unrelated one offers no way to add it.
#[tokio::test]
async fn a_scan_classifies_camelid_ollama_and_an_unrelated_service() {
    let dir = tempfile::tempdir().expect("temp dir");
    let camelid = camelid_stub();
    let ollama = ollama_stub();
    let html = html_stub();
    let proxy = proxy_stub();
    let silent = silent_stub();
    let closed = closed_port();

    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;

    let ports = [
        camelid.port,
        ollama.port,
        html.port,
        proxy.port,
        silent.port,
        closed,
    ];
    let (status, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON],
        Some(&scope_over(&ports)),
    )
    .await;
    assert_eq!(status, 200, "{body}");

    let kind_on = |port: u16| -> String {
        body["findings"]
            .as_array()
            .expect("findings")
            .iter()
            .find(|finding| finding["port"] == port)
            .map(|finding| finding["classification"]["kind"].as_str().unwrap_or("").to_string())
            .unwrap_or_else(|| "<no finding>".to_string())
    };
    let finding_on = |port: u16| -> Value {
        body["findings"]
            .as_array()
            .expect("findings")
            .iter()
            .find(|finding| finding["port"] == port)
            .cloned()
            .unwrap_or(Value::Null)
    };

    assert_eq!(kind_on(camelid.port), "answers_like");
    assert_eq!(finding_on(camelid.port)["classification"]["engine"], "camelid");
    assert_eq!(kind_on(ollama.port), "answers_like");
    assert_eq!(finding_on(ollama.port)["classification"]["engine"], "ollama");
    assert_eq!(
        finding_on(ollama.port)["classification"]["version"],
        "0.33.2"
    );
    // Never "unknown engine": it speaks HTTP and matched nothing, and that is
    // all that is claimed.
    assert_eq!(kind_on(html.port), "other_http");
    assert_eq!(kind_on(proxy.port), "fabric_proxy");
    assert_eq!(kind_on(silent.port), "silent_after_connect");

    // Only the two engines may be added, and the closed port is counted rather
    // than dropped.
    assert!(!finding_on(camelid.port)["proposal"].is_null());
    assert!(!finding_on(ollama.port)["proposal"].is_null());
    for port in [html.port, proxy.port, silent.port] {
        assert!(
            finding_on(port)["proposal"].is_null(),
            "port {port} was offered as something to add"
        );
        assert!(!finding_on(port)["not_proposed"].is_null(), "port {port}");
    }
    assert_eq!(body["not_listed"]["refused"], 1);
    assert_eq!(
        body["findings"].as_array().expect("findings").len() as u64
            + body["not_listed"]["refused"].as_u64().unwrap_or(0)
            + body["not_listed"]["timed_out"].as_u64().unwrap_or(0)
            + body["not_listed"]["unreachable"].as_u64().unwrap_or(0)
            + body["not_listed"]["other"].as_u64().unwrap_or(0)
            + body["not_scanned"].as_u64().unwrap_or(0),
        body["planned"].as_u64().expect("planned"),
        "every planned probe must be accounted for: {body}"
    );

    // A keyed Camelid still answers like one: a 401 elsewhere is a finished
    // answer, not an unfinished check.
    let keyed = keyed_camelid_stub();
    let (_, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON],
        Some(&scope_over(&[keyed.port])),
    )
    .await;
    let finding = &body["findings"][0];
    assert_eq!(finding["classification"]["kind"], "answers_like");
    assert_eq!(finding["classification"]["engine"], "camelid");
    assert!(
        finding["classification"]["withheld_elsewhere"]
            .as_array()
            .expect("withheld list")
            .len()
            >= 2,
        "{finding}"
    );
}

// ---- credentials ------------------------------------------------------------

/// The central claim of the feature, tested where it can actually fail: across
/// a scan, a successful join, a refused join, and the probes that follow.
#[tokio::test]
async fn no_host_receives_a_credential_it_was_not_declared_to_receive() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let html = html_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));

    let addr = start_proxy(
        &nodes,
        keyed(),
        Some(BEARER),
        Some(discovery_config(&nodes, true, &[])),
        None,
    )
    .await;

    let (status, _) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &with_key(),
        Some(&scope_over(&[ollama.port, html.port])),
    )
    .await;
    assert_eq!(status, 200);

    // A join that succeeds...
    let join = serde_json::json!({
        "label": "found-ollama",
        "engine": "ollama",
        "host": "127.0.0.1",
        "port": ollama.port,
        "base_sha256": hash_of(&nodes),
    })
    .to_string();
    let (status, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover/join",
        &with_key(),
        Some(&join),
    )
    .await;
    assert_eq!(status, 200, "{body}");

    // ...and one that is refused, because the machine is not what was claimed.
    let refused = serde_json::json!({
        "label": "not-studio",
        "engine": "lmstudio",
        "host": "127.0.0.1",
        "port": html.port,
        "base_sha256": hash_of(&nodes),
    })
    .to_string();
    let (status, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover/join",
        &with_key(),
        Some(&refused),
    )
    .await;
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["error"]["code"], "no_longer_answers");

    // Let the proxy re-read the file and probe the node it just gained. The
    // status does not matter here — the probe round it forces does.
    tokio::time::sleep(RELOAD * 4).await;
    let _ = request(addr, "GET", "/v1/health", &with_key(), None).await;

    for (what, stub) in [("ollama", &ollama), ("unrelated", &html)] {
        let seen = stub.bytes().to_ascii_lowercase();
        assert!(
            !seen.contains(BEARER),
            "the {what} stub was sent the fabric's bearer:\n{seen}"
        );
        assert!(
            !seen.contains(KEY),
            "the {what} stub was sent the client's key:\n{seen}"
        );
        assert!(
            !seen.contains("authorization"),
            "the {what} stub was sent an Authorization header at all:\n{seen}"
        );
    }
    assert!(
        ollama.bytes().contains("User-Agent: camelid-fabric-discover/"),
        "a discovery probe must say what it is:\n{}",
        ollama.bytes()
    );
}

/// The paired positive: the warning shown before a join says a Camelid node
/// will be sent the bearer, and it is. If probing ever stopped doing that, the
/// warning would have quietly become false.
#[tokio::test]
async fn a_joined_camelid_node_receives_the_fabric_bearer_as_the_warning_says() {
    let dir = tempfile::tempdir().expect("temp dir");
    let node = camelid_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        Some(BEARER),
        Some(discovery_config(&nodes, true, &[])),
        None,
    )
    .await;

    let (_, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON],
        Some(&scope_over(&[node.port])),
    )
    .await;
    let warnings = body["findings"][0]["proposal"]["warnings"]
        .as_array()
        .expect("warnings")
        .iter()
        .filter_map(|warning| warning.as_str())
        .collect::<Vec<_>>();
    assert!(
        warnings.contains(&"bearer_will_be_sent"),
        "a camelid proposal on a bearer-holding proxy must warn: {warnings:?}"
    );

    let join = serde_json::json!({
        "label": "found-camelid",
        "engine": "camelid",
        "host": "127.0.0.1",
        "port": node.port,
        "base_sha256": hash_of(&nodes),
    })
    .to_string();
    let (status, body) = request(addr, "POST", "/v1/fabric/discover/join", &[JSON], Some(&join)).await;
    assert_eq!(status, 200, "{body}");

    tokio::time::sleep(RELOAD * 4).await;
    let (_, _) = request(addr, "GET", "/v1/health", &[], None).await;
    assert!(
        node.bytes().contains(&format!("Authorization: Bearer {BEARER}")),
        "the warning said this node would be sent the bearer, and it was not:\n{}",
        node.bytes()
    );
}

/// The warning follows two facts — the engine, and whether this proxy holds a
/// bearer at all — and never the engine name alone.
#[tokio::test]
async fn bearer_warning_follows_whether_a_bearer_is_configured() {
    let dir = tempfile::tempdir().expect("temp dir");
    let node = camelid_stub();
    let ollama = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));

    let warnings_from = |body: &Value, port: u16| -> Vec<String> {
        body["findings"]
            .as_array()
            .expect("findings")
            .iter()
            .find(|finding| finding["port"] == port)
            .and_then(|finding| finding["proposal"]["warnings"].as_array())
            .map(|warnings| {
                warnings
                    .iter()
                    .filter_map(|warning| warning.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };

    // No bearer configured: no warning, on any engine.
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;
    let (_, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON],
        Some(&scope_over(&[node.port, ollama.port])),
    )
    .await;
    assert!(
        !warnings_from(&body, node.port).contains(&"bearer_will_be_sent".to_string()),
        "a proxy holding no bearer must not warn that one will be sent"
    );

    // Bearer configured: the Camelid row warns, the Ollama row still does not.
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        Some(BEARER),
        Some(discovery_config(&nodes, true, &[])),
        None,
    )
    .await;
    let (_, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON],
        Some(&scope_over(&[node.port, ollama.port])),
    )
    .await;
    assert!(warnings_from(&body, node.port).contains(&"bearer_will_be_sent".to_string()));
    assert!(
        !warnings_from(&body, ollama.port).contains(&"bearer_will_be_sent".to_string()),
        "an engine that is never shown the bearer must not warn about it"
    );
}

// ---- default off, and unaffected --------------------------------------------

/// Nothing about an existing proxy changes until the flag is passed.
#[tokio::test]
async fn discovery_is_off_unless_asked_for() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(&nodes, ClientAuth::none(), None, None, None).await;

    for (method, path, body) in [
        ("GET", "/v1/fabric/discover", None),
        ("POST", "/v1/fabric/discover", Some(scope_over(&[ollama.port]))),
        ("POST", "/v1/fabric/discover/join", Some("{}".to_string())),
    ] {
        let (status, answer) = request(addr, method, path, &[JSON], body.as_deref()).await;
        assert_eq!(status, 404, "{method} {path}: {answer}");
        assert_eq!(answer["error"]["code"], "discovery_disabled", "{answer}");
    }
    assert_eq!(ollama.bytes(), "", "a disabled proxy connected to something");

    // The health body keeps exactly the keys it had, and the fallback 404 keeps
    // its codeless shape — which is how a page tells this build from an older
    // one that never had these routes. The status is deliberately not asserted:
    // the seeded node is a closed port, so this fabric has nothing ready and
    // answers 503, which is correct and not what this test is about.
    let (_, health) = request(addr, "GET", "/v1/health", &[], None).await;
    let mut keys: Vec<&str> = health.as_object().expect("object").keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["build", "models", "node_detail", "nodes", "ok", "placement", "ready", "service", "version"],
        "the health body gained or lost a key: {health}"
    );

    let (status, unknown) = request(addr, "GET", "/v1/no-such-thing", &[], None).await;
    assert_eq!(status, 404);
    let mut keys: Vec<&str> = unknown["error"]
        .as_object()
        .expect("error object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["message", "type"], "{unknown}");
}

// ---- the route guards -------------------------------------------------------

/// Drive the real router with the peer address chosen, which a socket cannot
/// do: a test that connected over loopback would always look like a local
/// caller, so the guard that matters most could never be exercised.
async fn as_peer(
    nodes: &Path,
    peer: SocketAddr,
    ports: &[u16],
) -> (u16, Value) {
    use tower::ServiceExt;

    let fabric = Fabric::from_node_file(nodes.to_path_buf())
        .expect("the node file loads")
        .with_timeout(Duration::from_millis(500));
    let config = ServeConfig {
        mode: RouteMode::Throughput,
        forward_timeout: Duration::from_secs(5),
        auth: ClientAuth::none(),
        tls: None,
        cors: None,
        mixed: MixedEngines::default(),
        bound: "127.0.0.1:8282".parse().expect("loopback"),
        discovery: Some(discovery_config(nodes, false, &[])),
    };
    let mut built = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/fabric/discover")
        .header("host", "127.0.0.1:8282")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(scope_over(ports)))
        .expect("valid request");
    built
        .extensions_mut()
        .insert(axum::extract::connect_info::ConnectInfo(peer));

    let response = camelid::fabric::server::router(fabric, config)
        .oneshot(built)
        .await
        .expect("the router answers");
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("collect body");
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// A proxy that scanned on anyone's request would be a way into the network
/// behind it.
#[tokio::test]
async fn a_remote_peer_cannot_trigger_a_scan() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));

    let (status, body) = as_peer(
        &nodes,
        SocketAddr::from(([192, 0, 2, 5], 5555)),
        &[ollama.port],
    )
    .await;
    assert_eq!(status, 403, "{body}");
    assert_eq!(body["error"]["code"], "discovery_loopback_only");
    assert_eq!(ollama.bytes(), "", "a remote caller reached a machine on this network");
}

/// On a dual-stack `[::]` bind a real IPv4 loopback client arrives as
/// ::ffff:127.0.0.1, which plain `is_loopback` refuses — so the guard would
/// lock out exactly the caller it exists to admit.
#[tokio::test]
async fn a_mapped_ipv4_loopback_peer_is_loopback() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));

    let mapped: SocketAddr = "[::ffff:127.0.0.1]:5555".parse().expect("mapped loopback");
    let (status, body) = as_peer(&nodes, mapped, &[ollama.port]).await;
    assert_eq!(status, 200, "a mapped IPv4 loopback caller is a local caller: {body}");
}

#[tokio::test]
async fn a_rebound_host_header_is_refused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;

    let (status, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON, ("Host", "attacker.example:8282")],
        Some(&scope_over(&[ollama.port])),
    )
    .await;
    assert_eq!(status, 403, "{body}");
    assert_eq!(body["error"]["code"], "discovery_host_not_loopback");
    assert_eq!(ollama.bytes(), "", "a rebound request reached a machine");

    // ...and a loopback name is accepted.
    let (status, _) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON, ("Host", "localhost:8282")],
        Some(&scope_over(&[ollama.port])),
    )
    .await;
    assert_eq!(status, 200);
}

/// CORS governs reading a reply, never whether the side effect happened, so the
/// origin is checked before the scan and before the write.
#[tokio::test]
async fn an_origin_not_on_the_cors_list_cannot_trigger_a_scan_or_a_join() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let allowed = "http://127.0.0.1:8181";
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[allowed])),
        ProxyCors::resolve(&[allowed.to_string()]).expect("valid origin"),
    )
    .await;
    let before = hash_of(&nodes);

    let (status, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON, ("Origin", "https://evil.example")],
        Some(&scope_over(&[ollama.port])),
    )
    .await;
    assert_eq!(status, 403, "{body}");
    assert_eq!(body["error"]["code"], "discovery_origin_not_allowed");

    let join = serde_json::json!({
        "label": "x", "engine": "ollama", "host": "127.0.0.1",
        "port": ollama.port, "base_sha256": before,
    })
    .to_string();
    let (status, _) = request(
        addr,
        "POST",
        "/v1/fabric/discover/join",
        &[JSON, ("Origin", "https://evil.example")],
        Some(&join),
    )
    .await;
    assert_eq!(status, 403);
    assert_eq!(hash_of(&nodes), before, "a refused origin changed the file");

    // The named origin is served.
    let (status, _) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON, ("Origin", allowed)],
        Some(&scope_over(&[ollama.port])),
    )
    .await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn a_text_plain_join_is_refused_and_writes_nothing() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;
    let before = hash_of(&nodes);

    let join = serde_json::json!({
        "label": "x", "engine": "ollama", "host": "127.0.0.1",
        "port": ollama.port, "base_sha256": before,
    })
    .to_string();
    let (status, _) = request(
        addr,
        "POST",
        "/v1/fabric/discover/join",
        &[("content-type", "text/plain")],
        Some(&join),
    )
    .await;
    assert_eq!(status, 415);
    assert_eq!(hash_of(&nodes), before);
}

#[tokio::test]
async fn a_second_scan_while_one_runs_is_refused() {
    let dir = tempfile::tempdir().expect("temp dir");
    let slow = silent_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;

    let scope = scope_over(&[slow.port]);
    let first = tokio::spawn(async move {
        request(addr, "POST", "/v1/fabric/discover", &[JSON], Some(&scope)).await
    });
    tokio::time::sleep(Duration::from_millis(120)).await;
    let (status, body) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON],
        Some(&scope_over(&[closed])),
    )
    .await;
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["error"]["code"], "scan_in_progress");
    let (status, _) = first.await.expect("first scan");
    assert_eq!(status, 200);
}

// ---- joining ----------------------------------------------------------------

/// A write nobody picks up is a write that did nothing.
#[tokio::test]
async fn a_join_through_the_proxy_appears_in_its_own_health() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;

    let join = serde_json::json!({
        "label": "found-ollama", "engine": "ollama", "host": "127.0.0.1",
        "port": ollama.port, "base_sha256": hash_of(&nodes),
    })
    .to_string();
    let (status, body) = request(addr, "POST", "/v1/fabric/discover/join", &[JSON], Some(&join)).await;
    assert_eq!(status, 200, "{body}");
    assert!(body["appended"].as_str().expect("appended").contains("found-ollama="));

    tokio::time::sleep(RELOAD * 4).await;
    let (_, health) = request(addr, "GET", "/v1/health", &[], None).await;
    let labels: Vec<&str> = health["node_detail"]
        .as_array()
        .expect("node detail")
        .iter()
        .filter_map(|node| node["spec"]["label"].as_str())
        .collect();
    assert!(labels.contains(&"found-ollama"), "{health}");
}

/// The join re-identifies rather than trusting what it was told.
#[tokio::test]
async fn a_join_refuses_an_engine_the_host_no_longer_answers_like() {
    let dir = tempfile::tempdir().expect("temp dir");
    let html = html_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;
    let before = hash_of(&nodes);

    let join = serde_json::json!({
        "label": "studio", "engine": "lmstudio", "host": "127.0.0.1",
        "port": html.port, "base_sha256": before,
    })
    .to_string();
    let (status, body) = request(addr, "POST", "/v1/fabric/discover/join", &[JSON], Some(&join)).await;
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["error"]["code"], "no_longer_answers");
    assert_eq!(hash_of(&nodes), before);
}

/// The re-check happens after the click, so if the host now reaches a
/// different socket than the one that was scanned, what the person confirmed is
/// not what would be written.
#[tokio::test]
async fn a_join_refuses_a_name_that_reaches_another_address() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;
    let before = hash_of(&nodes);

    // It still answers like an Ollama, and it is not the machine that was
    // scanned. Answering correctly is not the same as being the right host.
    let join = serde_json::json!({
        "label": "moved-ollama",
        "engine": "ollama",
        "host": "127.0.0.1",
        "port": ollama.port,
        "base_sha256": before,
        "scanned_address": format!("127.0.0.1:{closed}"),
    })
    .to_string();
    let (status, body) = request(addr, "POST", "/v1/fabric/discover/join", &[JSON], Some(&join)).await;
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["error"]["code"], "name_reaches_another_address");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains(&closed.to_string()),
        "the refusal has to name both addresses: {body}"
    );
    assert_eq!(hash_of(&nodes), before, "a refused join changed the file");
}

/// Two tabs, one file. Without a lock spanning read, compose and rename, one
/// of these lines is silently lost.
#[tokio::test]
async fn two_concurrent_joins_never_lose_a_line() {
    let dir = tempfile::tempdir().expect("temp dir");
    let first = ollama_stub();
    let second = ollama_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;
    let base = hash_of(&nodes);

    let body_for = |label: &str, port: u16| {
        serde_json::json!({
            "label": label, "engine": "ollama", "host": "127.0.0.1",
            "port": port, "base_sha256": base,
        })
        .to_string()
    };
    let a = body_for("one-ollama", first.port);
    let b = body_for("two-ollama", second.port);
    let (left, right) = tokio::join!(
        request(addr, "POST", "/v1/fabric/discover/join", &[JSON], Some(&a)),
        request(addr, "POST", "/v1/fabric/discover/join", &[JSON], Some(&b)),
    );

    let statuses = [left.0, right.0];
    assert!(
        statuses.contains(&200) && statuses.contains(&409),
        "exactly one of two joins on the same base may win: {statuses:?}"
    );
    let text = std::fs::read_to_string(&nodes).expect("read nodes");
    let node_lines = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .count();
    assert_eq!(node_lines, 2, "the file gained more or fewer than one node:\n{text}");
}

// ---- the command line -------------------------------------------------------

/// Discovery never reads the variable that names the fabric's own key, so a
/// shell configured for a node cannot leak one to a stranger.
#[test]
fn the_cli_never_reads_camelid_api_key_for_discovery() {
    let ollama = ollama_stub();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_camelid"))
        .args([
            "fabric",
            "discover",
            "--no-loopback",
            "--no-default-ports",
            "--host",
            "127.0.0.1",
            "--port",
            &ollama.port.to_string(),
        ])
        .env("CAMELID_API_KEY", "env-secret-91")
        .output()
        .expect("run discover");

    assert!(output.status.success(), "{:?}", String::from_utf8_lossy(&output.stderr));
    let seen = ollama.bytes();
    assert!(!seen.contains("env-secret-91"), "the key reached a scanned host:\n{seen}");
    assert!(!seen.to_ascii_lowercase().contains("authorization"), "{seen}");
    let printed = String::from_utf8_lossy(&output.stdout);
    assert!(printed.contains("answers_like"), "{printed}");
    assert!(printed.contains("ollama 0.33.2"), "{printed}");
    assert!(printed.contains("Nothing has been added"), "{printed}");
}

/// A run whose output is piped never writes: it prints the command that would.
#[test]
fn a_non_tty_run_never_writes_without_join() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let nodes = nodes_file(&dir, "local=camelid://127.0.0.1:9\n");
    let before = hash_of(&nodes);

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_camelid"))
        .args([
            "fabric",
            "discover",
            "--no-loopback",
            "--no-default-ports",
            "--host",
            "127.0.0.1",
            "--port",
            &ollama.port.to_string(),
            "--nodes-file",
            nodes.to_str().expect("path"),
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run discover");

    assert!(output.status.success());
    let printed = String::from_utf8_lossy(&output.stdout);
    assert!(printed.contains("--join"), "it must print the command instead:\n{printed}");
    assert!(printed.contains("Nothing has been added"), "{printed}");
    assert_eq!(hash_of(&nodes), before, "a piped run wrote to the nodes file");
}

/// A proxy asked to discover with nowhere legitimate to write is refused at
/// startup, not after it is already listening.
#[test]
fn discover_with_node_instead_of_nodes_file_is_refused_at_startup() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_camelid"))
        .args([
            "fabric",
            "serve",
            "--node",
            "a=127.0.0.1:1",
            "--discovery",
            "--addr",
            "127.0.0.1:0",
        ])
        .output()
        .expect("run serve");
    assert!(!output.status.success());
    let complaint = String::from_utf8_lossy(&output.stderr);
    assert!(
        complaint.contains("writes confirmed machines to the nodes file"),
        "{complaint}"
    );
}

/// One implementation, two front doors. If these ever diverge, a person's
/// terminal and their browser would be describing different networks.
#[tokio::test]
async fn the_route_and_the_cli_print_the_same_discovery() {
    let dir = tempfile::tempdir().expect("temp dir");
    let ollama = ollama_stub();
    let html = html_stub();
    let closed = closed_port();
    let nodes = nodes_file(&dir, &format!("local=camelid://127.0.0.1:{closed}\n"));
    let addr = start_proxy(
        &nodes,
        ClientAuth::none(),
        None,
        Some(discovery_config(&nodes, false, &[])),
        None,
    )
    .await;

    let (_, mut from_route) = request(
        addr,
        "POST",
        "/v1/fabric/discover",
        &[JSON],
        Some(&scope_over(&[ollama.port, html.port])),
    )
    .await;

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_camelid"))
        .args([
            "fabric",
            "discover",
            "--no-loopback",
            "--no-default-ports",
            "--host",
            "127.0.0.1",
            "--port",
            &ollama.port.to_string(),
            "--port",
            &html.port.to_string(),
            "--nodes-file",
            nodes.to_str().expect("path"),
            "--json",
        ])
        .output()
        .expect("run discover");
    let mut from_cli: Value =
        serde_json::from_slice(&output.stdout).expect("the CLI prints the same shape");

    // Everything that legitimately differs between two runs.
    for body in [&mut from_route, &mut from_cli] {
        let object = body.as_object_mut().expect("object");
        object.remove("elapsed_ms");
        object.remove("nodes_file");
        if let Some(findings) = object.get_mut("findings").and_then(Value::as_array_mut) {
            for finding in findings {
                // The proxy proves names; the CLI run above does too, but the
                // resolver's ordering is not a property of the shape.
                finding.as_object_mut().expect("finding").remove("name");
            }
        }
    }
    assert_eq!(
        from_route["findings"], from_cli["findings"],
        "the two front doors describe the same network differently"
    );
    assert_eq!(from_route["scope"]["ports"], from_cli["scope"]["ports"]);
    assert_eq!(from_route["credentials_presented"], "none");
    assert_eq!(from_cli["credentials_presented"], "none");
}
