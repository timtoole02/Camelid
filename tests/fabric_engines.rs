//! A fabric that contains engines it does not place on.
//!
//! P2 and P3 of the local-backend lane make foreign engines **visible** — an
//! operator can see the machine, what it is, and everything it can serve —
//! while keeping them out of placement entirely. These are the properties that
//! combination has to hold, driven against stub servers on loopback so the real
//! probe path, the real HTTP client and the real placement policy all run.
//!
//! The ones that matter most:
//!
//!   * an engine that publishes no queue depth reports **no load**, and that
//!     absence survives all the way onto the wire rather than becoming a zero;
//!   * a model only a foreign node holds is **not advertised**, because the
//!     fabric would refuse to route it;
//!   * an unknown capability is never a `no`, and a measurement is never
//!     credited to a version it was not taken on.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use camelid::fabric::{
    parse_node_spec, probe_node, route, Fabric, NodeEngine, NodeSpec, NodeStatus, Provenance,
    RouteError, RouteMode, RouteRequest,
};

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Answers a fixed set of paths from canned bodies, and 404s everything else.
/// One shape serves every engine, because what distinguishes them is which
/// paths they answer — which is exactly what the adapters encode.
struct StubEngine {
    port: u16,
    shutdown: Arc<AtomicBool>,
    paths: Arc<Mutex<Vec<String>>>,
    thread: Option<JoinHandle<()>>,
}

type Bodies = HashMap<&'static str, String>;

fn model_entries(names: &[&str]) -> String {
    names
        .iter()
        .map(|name| format!(r#"{{"name":"{name}","model":"{name}","size":2019393189}}"#))
        .collect::<Vec<_>>()
        .join(",")
}

/// Ollama: version, everything installed, and what is resident.
fn ollama_bodies(installed: &[&str], resident: &[&str]) -> Bodies {
    HashMap::from([
        ("/api/version", r#"{"version":"0.33.3"}"#.to_string()),
        (
            "/api/tags",
            format!(r#"{{"models":[{}]}}"#, model_entries(installed)),
        ),
        (
            "/api/ps",
            format!(r#"{{"models":[{}]}}"#, model_entries(resident)),
        ),
    ])
}

/// LM Studio: one listing carrying both what is downloaded and what is loaded.
fn lmstudio_bodies(models: &[(&str, &str)]) -> Bodies {
    let entries = models
        .iter()
        .map(|(id, state)| {
            format!(
                r#"{{"id":"{id}","object":"model","type":"llm","arch":"llama",
                     "compatibility_type":"gguf","quantization":"Q4_K_M",
                     "state":"{state}","max_context_length":131072}}"#
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    HashMap::from([(
        "/api/v0/models",
        format!(r#"{{"object":"list","data":[{entries}]}}"#),
    )])
}

/// Ollama with `/api/ps` unavailable: the node can still serve, we just cannot
/// learn which model is warm.
fn ollama_without_ps(installed: &[&str]) -> Bodies {
    let mut bodies = ollama_bodies(installed, &[]);
    bodies.remove("/api/ps");
    bodies
}

impl StubEngine {
    fn start(bodies: Bodies) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let paths = Arc::new(Mutex::new(Vec::new()));

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_paths = Arc::clone(&paths);
        let thread = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if thread_shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut stream) = stream else { continue };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));

                let mut raw = Vec::new();
                let mut buffer = [0_u8; 1024];
                while !raw.windows(4).any(|window| window == b"\r\n\r\n") {
                    match stream.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(read) => raw.extend_from_slice(&buffer[..read]),
                    }
                }
                let request = String::from_utf8_lossy(&raw).to_string();
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("")
                    .to_string();
                thread_paths.lock().expect("paths").push(path.clone());

                let response = match bodies.get(path.as_str()) {
                    Some(body) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    ),
                    None => "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
                };
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });

        Self {
            port,
            shutdown,
            paths,
            thread: Some(thread),
        }
    }

    fn spec(&self, label: &str, engine: &str) -> NodeSpec {
        parse_node_spec(&format!("{label}={engine}://127.0.0.1:{}", self.port))
            .expect("spec parses")
    }

    fn paths(&self) -> Vec<String> {
        self.paths.lock().expect("paths").clone()
    }
}

impl Drop for StubEngine {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Poke the accept loop so it observes the flag and exits.
        let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[test]
fn an_ollama_node_reports_its_engine_version_and_models() {
    let node = StubEngine::start(ollama_bodies(
        &["llama3.2:latest", "qwen3:4b"],
        &["llama3.2:latest"],
    ));
    let snapshot = probe_node(&node.spec("studio", "ollama"), None, PROBE_TIMEOUT);

    let ready = snapshot.status.ready().expect("the stub is serving");
    assert_eq!(ready.engine, NodeEngine::Ollama);
    assert_eq!(ready.version.as_deref(), Some("0.33.3"));
    assert_eq!(
        ready.models,
        vec!["llama3.2:latest".to_string(), "qwen3:4b".to_string()],
        "every installed model is reported, sorted"
    );
    assert_eq!(
        ready.active_model_id.as_deref(),
        Some("llama3.2:latest"),
        "exactly one resident model is the one it is serving"
    );

    // The whole point of the seam.
    assert_eq!(ready.load, None, "Ollama publishes no queue depth");
    assert_eq!(ready.in_flight(), None);
    assert_eq!(
        ready.backend, None,
        "Ollama names no execution lane, so it must not claim one"
    );

    assert_eq!(
        node.paths(),
        vec!["/api/version", "/api/tags", "/api/ps"],
        "the adapter reads exactly the documented endpoints, in order"
    );
}

#[test]
fn an_unreported_load_is_absent_from_the_wire_rather_than_zero() {
    let node = StubEngine::start(ollama_bodies(&["llama3.2:latest"], &[]));
    let snapshot = probe_node(&node.spec("studio", "ollama"), None, PROBE_TIMEOUT);

    let value = serde_json::to_value(&snapshot).expect("serializes");
    let status = &value["status"];
    assert_eq!(status["state"], "ready");
    assert_eq!(status["engine"], "ollama");
    assert!(
        status.get("in_flight").is_none(),
        "a load nobody published must not appear as a number: {status}"
    );
    assert!(status.get("waiting").is_none(), "{status}");
}

#[test]
fn several_resident_models_are_listed_without_picking_one_as_active() {
    let node = StubEngine::start(ollama_bodies(
        &["a:latest", "b:latest"],
        &["a:latest", "b:latest"],
    ));
    let snapshot = probe_node(&node.spec("studio", "ollama"), None, PROBE_TIMEOUT);
    let ready = snapshot.status.ready().expect("serving");

    assert_eq!(ready.models.len(), 2);
    assert_eq!(
        ready.active_model_id, None,
        "with two models resident, naming one of them would be an invention"
    );
}

#[test]
fn a_node_whose_running_models_cannot_be_read_still_serves() {
    let node = StubEngine::start(ollama_without_ps(&["a:latest"]));
    let snapshot = probe_node(&node.spec("studio", "ollama"), None, PROBE_TIMEOUT);
    let ready = snapshot.status.ready().expect("still serving");

    assert_eq!(ready.models, vec!["a:latest".to_string()]);
    assert_eq!(ready.active_model_id, None, "degrades to reporting none");
}

#[test]
fn an_ollama_server_with_no_models_is_not_ready_and_says_why() {
    let node = StubEngine::start(ollama_bodies(&[], &[]));
    let snapshot = probe_node(&node.spec("studio", "ollama"), None, PROBE_TIMEOUT);

    match snapshot.status {
        NodeStatus::NotReady { reason } => {
            assert!(reason.contains("no models installed"), "{reason}");
            assert!(reason.contains("ollama pull"), "{reason}");
        }
        other => panic!("expected a reason an operator can act on, got {other:?}"),
    }
}

#[test]
fn an_address_that_is_not_an_ollama_server_is_unreachable_rather_than_ready() {
    // Port 9 is closed. Reaching for `/api/version` first means a wrong address
    // never gets as far as being reported as a serving node.
    let spec = parse_node_spec("studio=ollama://127.0.0.1:9").expect("spec parses");
    let snapshot = probe_node(&spec, None, Duration::from_millis(500));
    assert!(matches!(snapshot.status, NodeStatus::Unreachable { .. }));
}

#[test]
fn a_healthy_foreign_node_is_never_placed_on() {
    let node = StubEngine::start(ollama_bodies(&["llama3.2:latest"], &["llama3.2:latest"]));
    let snapshot = probe_node(&node.spec("studio", "ollama"), None, PROBE_TIMEOUT);
    assert!(snapshot.status.is_ready(), "the node itself is healthy");
    assert!(
        !snapshot.is_placeable(),
        "healthy is not the same as somewhere work goes"
    );

    let error = route(
        std::slice::from_ref(&snapshot),
        &RouteRequest::new(RouteMode::Throughput),
    )
    .expect_err("a fabric of only foreign nodes can place nothing");

    // Counted apart from unreachable and not-ready: nothing an operator does to
    // that machine will make it eligible, so telling them to fix it would be
    // wrong.
    assert_eq!(
        error,
        RouteError::AllNodesUnavailable {
            unreachable: 0,
            not_ready: 0,
            not_placeable: 1,
        }
    );
    let message = error.to_string();
    assert!(
        message.contains("does not place on"),
        "the refusal has to say why: {message}"
    );
}

#[test]
fn a_model_only_a_foreign_node_holds_is_not_advertised() {
    // Advertising a model the fabric would then refuse to route is worse than
    // not advertising it: `GET /v1/models` is what an SDK reads before asking.
    let foreign = StubEngine::start(ollama_bodies(&["only-here:latest"], &[]));
    let snapshot = probe_node(&foreign.spec("studio", "ollama"), None, PROBE_TIMEOUT);

    assert_eq!(
        snapshot.models(),
        ["only-here:latest".to_string()],
        "the node's own detail still shows what it holds"
    );
    assert!(
        camelid::fabric::servable_models(std::slice::from_ref(&snapshot)).is_empty(),
        "but the fabric advertises nothing it will not route"
    );
}

#[test]
fn a_fabric_may_hold_both_engines_at_once() {
    let foreign = StubEngine::start(ollama_bodies(&["a:latest"], &[]));
    let fabric = Fabric::new(vec![
        NodeSpec::camelid("win", "127.0.0.1", 9),
        foreign.spec("studio", "ollama"),
    ])
    .with_timeout(PROBE_TIMEOUT);

    let snapshots = fabric.observe();
    assert_eq!(snapshots.len(), 2);
    assert_eq!(snapshots[0].engine(), NodeEngine::Camelid);
    assert_eq!(snapshots[1].engine(), NodeEngine::Ollama);
    assert!(
        snapshots[1].status.is_ready(),
        "the foreign node is reported as the healthy machine it is"
    );
}

/* ---- LM Studio ---- */

#[test]
fn an_lm_studio_node_reports_what_it_holds_and_what_is_loaded() {
    let node = StubEngine::start(lmstudio_bodies(&[
        ("qwen2-vl-7b-instruct", "not-loaded"),
        ("meta-llama-3.1-8b-instruct", "loaded"),
    ]));
    let snapshot = probe_node(&node.spec("studio", "lmstudio"), None, PROBE_TIMEOUT);
    let ready = snapshot.status.ready().expect("the stub is serving");

    assert_eq!(ready.engine, NodeEngine::LmStudio);
    assert_eq!(
        ready.models,
        vec![
            "meta-llama-3.1-8b-instruct".to_string(),
            "qwen2-vl-7b-instruct".to_string()
        ],
        "everything downloaded is something it could serve"
    );
    assert_eq!(
        ready.active_model_id.as_deref(),
        Some("meta-llama-3.1-8b-instruct"),
        "exactly one loaded model is the one it is serving"
    );
    assert_eq!(ready.load, None, "LM Studio publishes no queue depth");
    assert_eq!(ready.backend, None);
    assert_eq!(
        ready.version, None,
        "LM Studio has no version endpoint, so no version may be claimed"
    );

    assert_eq!(
        node.paths(),
        vec!["/api/v0/models"],
        "one listing answers both questions"
    );
}

#[test]
fn an_lm_studio_node_holding_nothing_is_not_ready_and_says_why() {
    let node = StubEngine::start(lmstudio_bodies(&[]));
    match probe_node(&node.spec("studio", "lmstudio"), None, PROBE_TIMEOUT).status {
        NodeStatus::NotReady { reason } => {
            assert!(reason.contains("no models downloaded"), "{reason}")
        }
        other => panic!("expected a reason an operator can act on, got {other:?}"),
    }
}

#[test]
fn an_lm_studio_default_port_is_assumed_when_none_is_given() {
    let spec = parse_node_spec("studio=lmstudio://workstation.local").expect("spec parses");
    assert_eq!(spec.engine, NodeEngine::LmStudio);
    assert_eq!(spec.port, 1234);
}

#[test]
fn adding_lm_studio_changed_nothing_about_how_placement_refuses() {
    // The seam's own claim: a second foreign engine is excluded by exactly the
    // path the first one was, with no placement code aware that it exists.
    let node = StubEngine::start(lmstudio_bodies(&[("only-here", "loaded")]));
    let snapshot = probe_node(&node.spec("studio", "lmstudio"), None, PROBE_TIMEOUT);
    assert!(snapshot.status.is_ready());
    assert!(!snapshot.is_placeable());

    let error = route(
        std::slice::from_ref(&snapshot),
        &RouteRequest::new(RouteMode::Throughput),
    )
    .expect_err("a fabric of only foreign nodes can place nothing");
    assert_eq!(
        error,
        RouteError::AllNodesUnavailable {
            unreachable: 0,
            not_ready: 0,
            not_placeable: 1,
        }
    );
    assert!(camelid::fabric::servable_models(std::slice::from_ref(&snapshot)).is_empty());
}

/* ---- capabilities ---- */

#[test]
fn a_node_says_what_its_engine_can_be_asked_and_how_we_know() {
    let node = StubEngine::start(lmstudio_bodies(&[("a", "loaded")]));
    let snapshot = probe_node(&node.spec("studio", "lmstudio"), None, PROBE_TIMEOUT);
    let capabilities = snapshot.capabilities();

    // Declared from the documented API surface.
    assert_eq!(capabilities.load_reporting.supported, Some(false));
    assert_eq!(capabilities.load_reporting.provenance, Provenance::Declared);

    // Never credited on documentation alone, and never measured for a node
    // that cannot tell us which build it is.
    assert_eq!(capabilities.tool_calls.supported, None);
    assert_eq!(capabilities.tool_calls.provenance, Provenance::NotProbed);

    assert_eq!(
        capabilities.placement_blockers().len(),
        3,
        "the operator is told exactly why it is not routed to"
    );
}

#[test]
fn a_measurement_follows_the_version_the_node_actually_reports() {
    // The stub reports 0.33.3. Our tool-call measurement was taken on 0.33.1,
    // so this node inherits nothing from it.
    let node = StubEngine::start(ollama_bodies(&["a:latest"], &[]));
    let snapshot = probe_node(&node.spec("studio", "ollama"), None, PROBE_TIMEOUT);
    assert_eq!(
        snapshot.status.ready().expect("serving").version.as_deref(),
        Some("0.33.3")
    );
    assert_eq!(
        snapshot.capabilities().tool_calls.provenance,
        Provenance::NotProbed,
        "a neighbouring version inherits no measurement"
    );
}
