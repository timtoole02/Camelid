//! Probing a node's `/v1/health`.
//!
//! Everything here turns one HTTP response into a routing fact. The socket work
//! lives in [`super::http`]; [`classify`] is pure and owns the decision about
//! what "ready" means.

use std::time::{Duration, Instant};

use serde::Deserialize;

use super::cancel::Cancel;
use super::engine::NodeEngine;
use super::http::{self, HttpError};
use super::node::{NodeLoad, NodeReady, NodeSnapshot, NodeSpec, NodeStatus};
use super::transport::NodeTransport;

/// Refuse a health body larger than this. A health response is a few KiB;
/// anything at this size means we are not talking to a Camelid engine.
const MAX_HEALTH_BYTES: usize = 1024 * 1024;

/// Default probe budget. Wi-Fi RTT on this fabric was measured at 3-13 ms, so
/// two seconds is generous for a healthy node and still fails a dead one fast.
pub const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeError {
    Transport(String),
    Status(u16),
    Json(String),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(detail) => write!(f, "{detail}"),
            Self::Status(code) => write!(f, "health endpoint answered HTTP {code}"),
            Self::Json(detail) => write!(f, "health payload was not readable: {detail}"),
        }
    }
}

impl From<HttpError> for ProbeError {
    fn from(error: HttpError) -> Self {
        Self::Transport(error.to_string())
    }
}

/// The subset of `/v1/health` the fabric routes on.
///
/// Every field but `ok` defaults, so a node running a newer or older engine that
/// renames an unrelated field still probes successfully instead of dropping out
/// of the fabric.
#[derive(Debug, Clone, Deserialize)]
struct HealthPayload {
    #[serde(default)]
    ok: bool,
    #[serde(default)]
    generation_ready: bool,
    #[serde(default)]
    active_model_id: Option<String>,
    #[serde(default)]
    backend: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    engine_queued_tasks: usize,
    #[serde(default)]
    engine_queue_depth: usize,
}

/// Turn a successful health payload into a routing status. Pure.
fn classify(payload: &HealthPayload) -> NodeStatus {
    if !payload.ok {
        return NodeStatus::NotReady {
            reason: "engine reported not ok".to_string(),
        };
    }
    if !payload.generation_ready {
        let reason = match &payload.active_model_id {
            Some(model) => format!("model `{model}` loaded but not ready to generate"),
            None => "no model loaded".to_string(),
        };
        return NodeStatus::NotReady { reason };
    }
    NodeStatus::Ready(NodeReady {
        engine: NodeEngine::Camelid,
        active_model_id: payload.active_model_id.clone(),
        // A Camelid node serves exactly the model it has loaded.
        models: payload.active_model_id.clone().into_iter().collect(),
        backend: (!payload.backend.is_empty()).then(|| payload.backend.clone()),
        version: (!payload.version.is_empty()).then(|| payload.version.clone()),
        load: Some(NodeLoad {
            // `engine_queue_depth` is a gauge of jobs in flight, not a bound; see
            // the note on `NodeReady`.
            in_flight: payload.engine_queue_depth,
            waiting: payload.engine_queued_tasks,
        }),
    })
}

fn read_health(
    spec: &NodeSpec,
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<HealthPayload, ProbeError> {
    let response = http::request_with_transport(
        &spec.host,
        spec.port,
        "GET",
        "/v1/health",
        None,
        bearer,
        timeout,
        MAX_HEALTH_BYTES,
        // A probe round is already bounded by `timeout` and costs a node a
        // health read, not a generation slot, so there is nothing here worth
        // the noise of making cancellable.
        &Cancel::never(),
        transport,
    )?;
    if response.status != 200 {
        return Err(ProbeError::Status(response.status));
    }
    serde_json::from_slice::<HealthPayload>(&response.body)
        .map_err(|error| ProbeError::Json(error.to_string()))
}

/// Why a probe left a node unroutable. Pure.
///
/// `/v1/health` is exempt from the server's auth today, so a rejected
/// credential cannot reach here yet. If that exemption is ever tightened, the
/// bare status would render as "offline" and send an operator hunting a network
/// fault instead of a missing key, so 401 and 403 say what they mean.
fn unreachable_reason(error: &ProbeError) -> String {
    match error {
        ProbeError::Status(code @ (401 | 403)) => format!(
            "node rejected the fabric's credentials (HTTP {code}); \
             pass --bearer or set CAMELID_API_KEY"
        ),
        other => other.to_string(),
    }
}

/// Probe one node. Never fails: an unreachable node is a routing fact, not an
/// error the caller has to handle separately.
pub fn probe_node(spec: &NodeSpec, bearer: Option<&str>, timeout: Duration) -> NodeSnapshot {
    probe_node_with_transport(spec, bearer, timeout, &NodeTransport::default())
}

pub(crate) fn probe_node_with_transport(
    spec: &NodeSpec,
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> NodeSnapshot {
    let started = Instant::now();
    let status = match spec.engine {
        NodeEngine::Camelid => match read_health(spec, bearer, timeout, transport) {
            Ok(payload) => classify(&payload),
            Err(error) => NodeStatus::Unreachable {
                reason: unreachable_reason(&error),
            },
        },
        NodeEngine::Ollama => super::ollama::probe(spec, timeout, transport),
        NodeEngine::LmStudio => super::lmstudio::probe(spec, timeout, transport),
    };
    // Timed for every outcome: a node that answered "not ready" still told us how
    // long it took to say so, and that is the same fact for both engines.
    let latency = match status {
        NodeStatus::Unreachable { .. } => None,
        _ => Some(started.elapsed()),
    };
    NodeSnapshot {
        spec: spec.clone(),
        status,
        latency,
    }
}

/// Probe every node concurrently, one thread each.
///
/// Sequential probing would make a fabric's status cost the sum of its timeouts,
/// so a single dead node would stall the view of every live one.
pub fn probe_fabric(
    specs: &[NodeSpec],
    bearer: Option<&str>,
    timeout: Duration,
) -> Vec<NodeSnapshot> {
    probe_fabric_with_transport(specs, bearer, timeout, &NodeTransport::default())
}

pub(crate) fn probe_fabric_with_transport(
    specs: &[NodeSpec],
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Vec<NodeSnapshot> {
    if specs.is_empty() {
        return Vec::new();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = specs
            .iter()
            .map(|spec| {
                scope.spawn(move || probe_node_with_transport(spec, bearer, timeout, transport))
            })
            .collect();
        handles
            .into_iter()
            .zip(specs)
            .map(|(handle, spec)| {
                handle.join().unwrap_or_else(|_| NodeSnapshot {
                    spec: spec.clone(),
                    status: NodeStatus::Unreachable {
                        reason: "probe thread panicked".to_string(),
                    },
                    latency: None,
                })
            })
            .collect()
    })
}

/// A fabric observation, and the moment it was taken.
///
/// [`super::Fabric::place`] needs an observation before every request. Probing
/// for each one is exactly right for a CLI invocation, which makes one request
/// and exits. For the resident proxy it is a `/v1/health` per node per request,
/// and while any node is black-holing it is the whole probe budget on every
/// request — measured at 2.0 s each against nodes that answer in 2 ms.
///
/// Reusing an observation trades freshness for that cost, so how stale a reused
/// one may be is a stated bound rather than an implicit one. Pure, so the
/// decision is tested without sockets and without sleeping.
///
/// `taken` is when the probe *completed*. Stamping it at the start would make
/// an observation that waited out a black-holing node be born already expired,
/// which is precisely the case this exists for; the price is that the oldest
/// fact in an observation can be one probe budget older than the bound alone
/// suggests.
#[derive(Debug, Clone)]
pub struct Observation {
    snapshots: Vec<NodeSnapshot>,
    taken: Instant,
}

impl Observation {
    pub fn taken_at(snapshots: Vec<NodeSnapshot>, taken: Instant) -> Self {
        Self { snapshots, taken }
    }

    pub fn snapshots(&self) -> &[NodeSnapshot] {
        &self.snapshots
    }

    /// Whether this observation may still be used at `now`.
    ///
    /// The bound is exclusive, so a `max_age` of zero reuses nothing. That is
    /// what a caller wanting the fabric as it is right now asks for, and it is
    /// the default.
    pub fn is_fresh_at(&self, now: Instant, max_age: Duration) -> bool {
        now.saturating_duration_since(self.taken) < max_age
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(ok: bool, ready: bool, model: Option<&str>) -> HealthPayload {
        HealthPayload {
            ok,
            generation_ready: ready,
            active_model_id: model.map(str::to_string),
            backend: "llama".to_string(),
            version: "0.5.4".to_string(),
            engine_queued_tasks: 1,
            engine_queue_depth: 4,
        }
    }

    #[test]
    fn a_ready_engine_becomes_a_routable_node() {
        let status = classify(&payload(true, true, Some("llama-3b")));
        let ready = status.ready().expect("ready");
        assert_eq!(ready.active_model_id.as_deref(), Some("llama-3b"));
        assert_eq!(ready.in_flight(), Some(4));
        assert_eq!(ready.load.expect("camelid reports load").waiting, 1);
        assert_eq!(ready.engine, NodeEngine::Camelid);
        assert_eq!(ready.models, vec!["llama-3b".to_string()]);
    }

    #[test]
    fn an_idle_engine_is_routable_rather_than_read_as_full() {
        // Regression: an idle node reports 0/0. Reading `engine_queue_depth` as a
        // capacity bound made a healthy idle fabric look saturated.
        let idle = HealthPayload {
            ok: true,
            generation_ready: true,
            active_model_id: Some("llama-3b".to_string()),
            backend: "llama".to_string(),
            version: "0.5.4".to_string(),
            engine_queued_tasks: 0,
            engine_queue_depth: 0,
        };
        let status = classify(&idle);
        assert!(status.is_ready());
        assert_eq!(status.ready().expect("ready").in_flight(), Some(0));
    }

    #[test]
    fn a_loaded_but_unready_engine_names_the_model_it_is_warming() {
        let status = classify(&payload(true, false, Some("llama-3b")));
        match status {
            NodeStatus::NotReady { reason } => assert!(reason.contains("llama-3b"), "{reason}"),
            other => panic!("expected NotReady, got {other:?}"),
        }
    }

    #[test]
    fn an_engine_with_no_model_is_not_ready_and_says_so() {
        match classify(&payload(true, false, None)) {
            NodeStatus::NotReady { reason } => assert_eq!(reason, "no model loaded"),
            other => panic!("expected NotReady, got {other:?}"),
        }
    }

    #[test]
    fn an_engine_reporting_not_ok_is_never_routable() {
        assert!(!classify(&payload(false, true, Some("llama-3b"))).is_ready());
    }

    #[test]
    fn a_health_payload_missing_optional_fields_still_parses() {
        // A node on a different engine version must not drop out of the fabric
        // because an unrelated field was renamed.
        let parsed: HealthPayload =
            serde_json::from_str(r#"{"ok":true,"generation_ready":true}"#).expect("parses");
        assert!(classify(&parsed).is_ready());
    }

    #[test]
    fn probing_an_empty_fabric_does_no_work() {
        assert!(probe_fabric(&[], None, DEFAULT_PROBE_TIMEOUT).is_empty());
    }

    #[test]
    fn an_unroutable_address_becomes_unreachable_not_an_error() {
        // Port 1 on loopback is closed; this exercises the real socket path.
        let spec = NodeSpec::camelid("dead", "127.0.0.1", 1);
        let snapshot = probe_node(&spec, None, Duration::from_millis(500));
        assert!(matches!(snapshot.status, NodeStatus::Unreachable { .. }));
        assert_eq!(snapshot.label(), "dead");
    }

    #[test]
    fn a_rejected_credential_names_the_key_rather_than_reading_as_offline() {
        for code in [401, 403] {
            let reason = unreachable_reason(&ProbeError::Status(code));
            assert!(reason.contains("credentials"), "{reason}");
            assert!(reason.contains("--bearer"), "{reason}");
            assert!(reason.contains(&code.to_string()), "{reason}");
        }
    }

    #[test]
    fn every_other_probe_failure_keeps_its_own_wording() {
        assert_eq!(
            unreachable_reason(&ProbeError::Status(503)),
            "health endpoint answered HTTP 503"
        );
        assert_eq!(
            unreachable_reason(&ProbeError::Transport("cannot connect".to_string())),
            "cannot connect"
        );
    }

    /// Built by adding to an instant rather than subtracting from `now`, so the
    /// test states an elapsed time exactly instead of racing the clock.
    fn observation_aged(elapsed: Duration) -> (Observation, Instant) {
        let taken = Instant::now();
        (Observation::taken_at(Vec::new(), taken), taken + elapsed)
    }

    #[test]
    fn an_observation_inside_the_bound_is_reusable() {
        let (observation, now) = observation_aged(Duration::from_millis(499));
        assert!(observation.is_fresh_at(now, Duration::from_millis(500)));
    }

    #[test]
    fn an_observation_at_the_bound_is_not_reusable() {
        let (observation, now) = observation_aged(Duration::from_millis(500));
        assert!(!observation.is_fresh_at(now, Duration::from_millis(500)));
    }

    #[test]
    fn a_zero_bound_reuses_nothing_not_even_an_observation_just_taken() {
        let (observation, now) = observation_aged(Duration::ZERO);
        assert!(!observation.is_fresh_at(now, Duration::ZERO));
    }
}
