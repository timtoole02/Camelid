//! Identity and observed state of one Camelid node in a fabric.
//!
//! A node is a whole, independent `camelid serve` process that owns a complete
//! model and a complete session. The fabric never reaches inside one: the only
//! thing it learns is what `/v1/health` reports, and the only thing it decides
//! is which node a whole request goes to. That boundary is what keeps the token
//! loop free of cross-node traffic.

use std::fmt;
use std::time::Duration;

use serde::{Serialize, Serializer};

use super::engine::NodeEngine;

/// Operator-supplied identity of one node.
///
/// `host` stays a string rather than a resolved `SocketAddr` because fabric
/// members are typically named (`workstation.local`), and a name that resolves
/// late survives a DHCP lease change that a cached address would not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NodeSpec {
    pub label: String,
    pub host: String,
    pub port: u16,
    /// Declared, never inferred from what answers. See [`NodeEngine`].
    pub engine: NodeEngine,
}

/// Default port for a `camelid serve` process.
pub const DEFAULT_NODE_PORT: u16 = 8181;

impl NodeSpec {
    /// A node running the default engine, which is every specification written
    /// before the fabric could read more than one.
    pub fn camelid(label: impl Into<String>, host: impl Into<String>, port: u16) -> Self {
        Self {
            label: label.into(),
            host: host.into(),
            port,
            engine: NodeEngine::Camelid,
        }
    }

    pub fn authority(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

impl fmt::Display for NodeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.label, self.authority())
    }
}

/// Why parsing an operator-supplied node string failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeSpecParseError {
    Empty,
    MissingLabel,
    MissingHost,
    DuplicateLabel(String),
    BadPort(String),
    /// A bare IPv6 literal cannot be told apart from `host:port`, so RFC 3986
    /// brackets are required rather than guessed at.
    UnbracketedIpv6(String),
    MalformedEndpoint(String),
    UnknownEngine(String),
}

impl fmt::Display for NodeSpecParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "node specification is empty"),
            Self::MissingLabel => write!(
                f,
                "node specification needs a label, as in `windows=127.0.0.1:8181`"
            ),
            Self::MissingHost => write!(f, "node specification needs a host"),
            Self::DuplicateLabel(label) => {
                write!(f, "node label `{label}` is used more than once")
            }
            Self::BadPort(port) => write!(f, "`{port}` is not a valid port"),
            Self::UnbracketedIpv6(endpoint) => write!(
                f,
                "`{endpoint}` looks like an IPv6 address; wrap it in brackets, as in `[::1]:8181`"
            ),
            Self::MalformedEndpoint(endpoint) => {
                write!(f, "`{endpoint}` is not a valid host[:port]")
            }
            Self::UnknownEngine(scheme) => write!(
                f,
                "`{scheme}` is not an engine this fabric can read; known engines are {}",
                NodeEngine::known().join(", ")
            ),
        }
    }
}

fn parse_port(raw: &str) -> Result<u16, NodeSpecParseError> {
    match raw.parse::<u16>() {
        Ok(port) if port != 0 => Ok(port),
        _ => Err(NodeSpecParseError::BadPort(raw.to_string())),
    }
}

impl std::error::Error for NodeSpecParseError {}

/// Parse one `label=[engine://]host[:port]` string.
///
/// The label is mandatory and not derived from the host: routing decisions,
/// affinity keys and receipts all quote it, and a host that changes address
/// must not silently become a different node.
///
/// The engine scheme is optional and defaults to Camelid, so every
/// specification written before other engines existed still means what it did.
/// It is declared rather than detected — see [`NodeEngine`].
pub fn parse_node_spec(raw: &str) -> Result<NodeSpec, NodeSpecParseError> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(NodeSpecParseError::Empty);
    }
    let (label, endpoint) = raw
        .split_once('=')
        .ok_or(NodeSpecParseError::MissingLabel)?;
    let label = label.trim();
    let endpoint = endpoint.trim();
    if label.is_empty() {
        return Err(NodeSpecParseError::MissingLabel);
    }
    if endpoint.is_empty() {
        return Err(NodeSpecParseError::MissingHost);
    }

    // Split the scheme before anything else looks at colons: `ollama://[::1]`
    // has to lose its scheme before the IPv6 bracket rules below apply.
    let (engine, endpoint) = match endpoint.split_once("://") {
        Some((scheme, rest)) => {
            let engine = NodeEngine::from_scheme(scheme)
                .ok_or_else(|| NodeSpecParseError::UnknownEngine(scheme.to_string()))?;
            (engine, rest.trim())
        }
        None => (NodeEngine::default(), endpoint),
    };
    if endpoint.is_empty() {
        return Err(NodeSpecParseError::MissingHost);
    }

    // A bracketed IPv6 literal owns every colon up to `]`; only a colon after the
    // bracket introduces a port. Splitting on the last colon instead would read
    // `[::1]` as host `[:` and port `1]`.
    let (host, port) = if endpoint.starts_with('[') {
        let close = endpoint
            .find(']')
            .ok_or_else(|| NodeSpecParseError::MalformedEndpoint(endpoint.to_string()))?;
        let host = &endpoint[..=close];
        if host.len() <= 2 {
            return Err(NodeSpecParseError::MissingHost);
        }
        match &endpoint[close + 1..] {
            "" => (host, engine.default_port()),
            rest => match rest.strip_prefix(':') {
                Some(port) => (host, parse_port(port)?),
                None => return Err(NodeSpecParseError::MalformedEndpoint(endpoint.to_string())),
            },
        }
    } else {
        match endpoint.rsplit_once(':') {
            Some((host, port)) => {
                if host.contains(':') {
                    return Err(NodeSpecParseError::UnbracketedIpv6(endpoint.to_string()));
                }
                (host, parse_port(port)?)
            }
            None => (endpoint, engine.default_port()),
        }
    };
    if host.is_empty() {
        return Err(NodeSpecParseError::MissingHost);
    }

    Ok(NodeSpec {
        label: label.to_string(),
        host: host.to_string(),
        port,
        engine,
    })
}

/// Parse a whole fabric, rejecting duplicate labels.
///
/// Duplicates are refused rather than de-duplicated: two nodes sharing a label
/// would make an affinity decision ambiguous, and silently dropping one would
/// hide a typo that costs the operator a machine.
pub fn parse_fabric(raws: &[String]) -> Result<Vec<NodeSpec>, NodeSpecParseError> {
    let mut specs: Vec<NodeSpec> = Vec::with_capacity(raws.len());
    for raw in raws {
        let spec = parse_node_spec(raw)?;
        if specs.iter().any(|existing| existing.label == spec.label) {
            return Err(NodeSpecParseError::DuplicateLabel(spec.label));
        }
        specs.push(spec);
    }
    Ok(specs)
}

/// What a node reported about its own load.
///
/// Separate from [`NodeReady`] so it can be absent as a whole: an engine either
/// publishes both figures or neither, and a partial pair would invite reading
/// the missing half as zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NodeLoad {
    /// Jobs accepted and not yet finished, queued plus running
    /// (`engine_queue_depth`). This is the load signal placement ranks on.
    pub in_flight: usize,
    /// Jobs queued but not yet running (`engine_queued_tasks`).
    pub waiting: usize,
}

/// The generation-relevant subset of what a node says about itself.
///
/// `/v1/health` reports current load but **not** the node's queue bound. The bound
/// is `CAMELID_QUEUE_DEPTH`, read on the node itself and never serialised — the
/// field named `engine_queue_depth` is a gauge of jobs in flight, not a capacity.
/// So the fabric ranks by observed load and never declares a node full: a node
/// genuinely at its bound answers a typed 503, which is a retry concern rather
/// than a placement one.
///
/// Every field an engine may not publish is an `Option`, and `None` means we
/// were not told. Nothing here defaults to a plausible value: a node reporting
/// no load must not be ranked as idle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NodeReady {
    /// Which engine answered. Echoed from the spec, because a probe reads the
    /// engine it was told to read.
    pub engine: NodeEngine,
    /// The single model this node is serving, when it serves exactly one.
    ///
    /// `None` for an engine that holds several at once, or none resident. Use
    /// [`NodeReady::models`] to see everything it can serve.
    pub active_model_id: Option<String>,
    /// Every model this node can serve right now, sorted.
    pub models: Vec<String>,
    /// The execution lane the engine chose, when it names one.
    pub backend: Option<String>,
    /// The engine's own version string, when it publishes one.
    pub version: Option<String>,
    /// Load, or `None` when the engine publishes none.
    ///
    /// Flattened so a node that reports load keeps the wire shape it always
    /// had, and a node that does not simply **omits the fields**. An absent key
    /// reads as unknown to every consumer; a zero would not.
    #[serde(flatten)]
    pub load: Option<NodeLoad>,
}

impl NodeReady {
    /// Jobs in flight, when the engine reported any. Never substitutes a zero.
    pub fn in_flight(&self) -> Option<usize> {
        self.load.map(|load| load.in_flight)
    }
}

/// What a probe learned about a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum NodeStatus {
    /// Answered `/v1/health` and reports it can generate.
    Ready(NodeReady),
    /// Answered, but cannot generate — no model loaded, or still warming.
    NotReady { reason: String },
    /// Did not answer within the probe budget.
    Unreachable { reason: String },
}

impl NodeStatus {
    pub fn ready(&self) -> Option<&NodeReady> {
        match self {
            Self::Ready(ready) => Some(ready),
            _ => None,
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }
}

/// Render a probe duration as whole milliseconds; `Duration`'s own
/// representation is not a useful thing to hand an operator or a script.
fn serialize_latency_ms<S: Serializer>(
    latency: &Option<Duration>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match latency {
        Some(latency) => serializer.serialize_some(&(latency.as_millis() as u64)),
        None => serializer.serialize_none(),
    }
}

/// One node's spec plus the most recent observation of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NodeSnapshot {
    pub spec: NodeSpec,
    pub status: NodeStatus,
    /// Round-trip time of the probe itself. Recorded for reporting only; routing
    /// deliberately does not rank on it, because one sample of a Wi-Fi RTT is
    /// noise (measured 3-13 ms across six pings on this fabric).
    #[serde(rename = "latency_ms", serialize_with = "serialize_latency_ms")]
    pub latency: Option<Duration>,
}

impl NodeSnapshot {
    pub fn label(&self) -> &str {
        &self.spec.label
    }

    /// The engine this node was declared to run.
    pub fn engine(&self) -> NodeEngine {
        self.spec.engine
    }

    /// Whether the fabric may place a request here. A node can be perfectly
    /// healthy and still not be somewhere work goes — see
    /// [`NodeEngine::is_placeable`].
    pub fn is_placeable(&self) -> bool {
        self.spec.engine.is_placeable() && self.status.is_ready()
    }

    /// What this node's engine can be asked, and how we know.
    ///
    /// Keyed on the version the node reported, so a measurement taken against a
    /// different build of the same engine is not credited to this one.
    pub fn capabilities(&self) -> super::capability::Capabilities {
        self.spec.engine.capabilities(
            self.status
                .ready()
                .and_then(|ready| ready.version.as_deref()),
        )
    }

    /// The model this node is currently serving, when it can serve at all.
    pub fn active_model_id(&self) -> Option<&str> {
        self.status
            .ready()
            .and_then(|ready| ready.active_model_id.as_deref())
    }

    /// Every model this node can serve right now.
    pub fn models(&self) -> &[String] {
        self.status
            .ready()
            .map_or(&[] as &[String], |ready| ready.models.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_host_takes_the_default_port() {
        let spec = parse_node_spec("mac=workstation.local").expect("parses");
        assert_eq!(spec.label, "mac");
        assert_eq!(spec.host, "workstation.local");
        assert_eq!(spec.port, DEFAULT_NODE_PORT);
    }

    #[test]
    fn a_specification_without_a_scheme_is_still_a_camelid_node() {
        // Every specification written before the fabric could read another
        // engine has to keep meaning exactly what it meant.
        let spec = parse_node_spec("win=127.0.0.1:8181").expect("parses");
        assert_eq!(spec.engine, NodeEngine::Camelid);
        assert_eq!(
            parse_node_spec("win=camelid://127.0.0.1:8181").expect("parses"),
            spec,
            "naming the default engine explicitly must change nothing"
        );
    }

    #[test]
    fn a_declared_engine_brings_its_own_default_port() {
        let spec = parse_node_spec("studio=ollama://workstation.local").expect("parses");
        assert_eq!(spec.engine, NodeEngine::Ollama);
        assert_eq!(spec.host, "workstation.local");
        assert_eq!(spec.port, 11434, "an Ollama node defaults to Ollama's port");

        let explicit = parse_node_spec("studio=ollama://workstation.local:9000").expect("parses");
        assert_eq!(explicit.port, 9000, "an explicit port still wins");
    }

    #[test]
    fn an_engine_is_declared_rather_than_detected() {
        // A typo, not a product. Refused rather than silently treated as
        // Camelid: sending this fabric's bearer to whatever answered would be
        // the alternative.
        assert_eq!(
            parse_node_spec("x=olama://127.0.0.1:11434"),
            Err(NodeSpecParseError::UnknownEngine("olama".to_string()))
        );
        let message = NodeSpecParseError::UnknownEngine("olama".to_string()).to_string();
        for engine in NodeEngine::known() {
            assert!(
                message.contains(engine),
                "the refusal must name every engine this build knows, and is missing {engine}: {message}"
            );
        }
    }

    #[test]
    fn a_scheme_is_stripped_before_ipv6_brackets_are_read() {
        let spec = parse_node_spec("v6=ollama://[::1]:11434").expect("parses");
        assert_eq!(spec.engine, NodeEngine::Ollama);
        assert_eq!(spec.host, "[::1]");
        assert_eq!(spec.port, 11434);
    }

    #[test]
    fn a_scheme_with_no_host_is_refused() {
        assert_eq!(
            parse_node_spec("x=ollama://"),
            Err(NodeSpecParseError::MissingHost)
        );
    }

    #[test]
    fn an_explicit_port_overrides_the_default() {
        let spec = parse_node_spec("win=127.0.0.1:8200").expect("parses");
        assert_eq!(spec.host, "127.0.0.1");
        assert_eq!(spec.port, 8200);
    }

    #[test]
    fn ipv6_literals_keep_their_colons() {
        let spec = parse_node_spec("v6=[::1]:8181").expect("parses");
        assert_eq!(spec.host, "[::1]");
        assert_eq!(spec.port, 8181);

        let bare = parse_node_spec("v6=[::1]").expect("parses");
        assert_eq!(bare.host, "[::1]");
        assert_eq!(bare.port, DEFAULT_NODE_PORT);

        let full = parse_node_spec("v6=[2001:db8::1]:9000").expect("parses");
        assert_eq!(full.host, "[2001:db8::1]");
        assert_eq!(full.port, 9000);
    }

    #[test]
    fn an_unbracketed_ipv6_is_refused_with_advice_instead_of_misparsed() {
        // `::1` would otherwise read as host `:` port `1`.
        assert_eq!(
            parse_node_spec("v6=::1"),
            Err(NodeSpecParseError::UnbracketedIpv6("::1".to_string()))
        );
        assert_eq!(
            parse_node_spec("v6=2001:db8::1"),
            Err(NodeSpecParseError::UnbracketedIpv6(
                "2001:db8::1".to_string()
            ))
        );
    }

    #[test]
    fn a_malformed_bracketed_endpoint_is_refused() {
        assert_eq!(
            parse_node_spec("v6=[::1"),
            Err(NodeSpecParseError::MalformedEndpoint("[::1".to_string()))
        );
        assert_eq!(
            parse_node_spec("v6=[::1]junk"),
            Err(NodeSpecParseError::MalformedEndpoint(
                "[::1]junk".to_string()
            ))
        );
        assert_eq!(
            parse_node_spec("v6=[]"),
            Err(NodeSpecParseError::MissingHost)
        );
    }

    #[test]
    fn surrounding_whitespace_is_tolerated() {
        let spec = parse_node_spec("  mac = 192.0.2.10:8181  ").expect("parses");
        assert_eq!(spec.label, "mac");
        assert_eq!(spec.host, "192.0.2.10");
    }

    #[test]
    fn malformed_specs_are_refused_with_a_reason() {
        assert_eq!(parse_node_spec(""), Err(NodeSpecParseError::Empty));
        assert_eq!(
            parse_node_spec("127.0.0.1:8181"),
            Err(NodeSpecParseError::MissingLabel)
        );
        assert_eq!(
            parse_node_spec("=127.0.0.1"),
            Err(NodeSpecParseError::MissingLabel)
        );
        assert_eq!(
            parse_node_spec("win="),
            Err(NodeSpecParseError::MissingHost)
        );
        assert_eq!(
            parse_node_spec("win=host:0"),
            Err(NodeSpecParseError::BadPort("0".to_string()))
        );
        assert_eq!(
            parse_node_spec("win=host:99999"),
            Err(NodeSpecParseError::BadPort("99999".to_string()))
        );
    }

    #[test]
    fn duplicate_labels_are_refused_rather_than_deduplicated() {
        let raws = vec!["a=h1:1".to_string(), "a=h2:2".to_string()];
        assert_eq!(
            parse_fabric(&raws),
            Err(NodeSpecParseError::DuplicateLabel("a".to_string()))
        );
    }

    #[test]
    fn distinct_labels_on_the_same_host_are_allowed() {
        // Two engines on one box, different ports, is a legitimate fabric.
        let raws = vec!["a=h:1".to_string(), "b=h:2".to_string()];
        assert_eq!(parse_fabric(&raws).expect("parses").len(), 2);
    }

    #[test]
    fn an_idle_node_reports_no_load() {
        // Regression: an idle engine reports 0 in flight. An earlier version read
        // `engine_queue_depth` as a capacity bound and concluded a healthy idle
        // fabric was full, refusing every request.
        let idle = NodeReady {
            engine: NodeEngine::Camelid,
            active_model_id: Some("llama-3b".to_string()),
            models: vec!["llama-3b".to_string()],
            backend: Some("llama".to_string()),
            version: Some("0.5.4".to_string()),
            load: Some(NodeLoad {
                in_flight: 0,
                waiting: 0,
            }),
        };
        assert_eq!(idle.in_flight(), Some(0));
    }

    #[test]
    fn a_node_that_reports_no_load_is_not_reported_as_idle() {
        // The distinction the whole engine seam exists to keep: an engine that
        // publishes no queue depth must not be indistinguishable from one that
        // published a zero.
        let unknown = NodeReady {
            engine: NodeEngine::Ollama,
            active_model_id: None,
            models: vec!["llama3.2:latest".to_string()],
            backend: None,
            version: Some("0.33.3".to_string()),
            load: None,
        };
        assert_eq!(unknown.in_flight(), None);

        let value = serde_json::to_value(NodeStatus::Ready(unknown)).expect("serializes");
        assert!(
            value.get("in_flight").is_none(),
            "an unreported load must be absent from the wire, not zero: {value}"
        );
        assert!(value.get("waiting").is_none(), "{value}");
        assert_eq!(value["engine"], "ollama");
        assert_eq!(value["backend"], serde_json::Value::Null);
    }

    #[test]
    fn a_snapshot_serializes_with_a_flat_state_tag_and_millisecond_latency() {
        let snapshot = NodeSnapshot {
            spec: NodeSpec::camelid("windows", "127.0.0.1", 8181),
            status: NodeStatus::Ready(NodeReady {
                engine: NodeEngine::Camelid,
                active_model_id: Some("llama-3b".to_string()),
                models: vec!["llama-3b".to_string()],
                backend: Some("llama".to_string()),
                version: Some("0.5.4".to_string()),
                load: Some(NodeLoad {
                    in_flight: 1,
                    waiting: 0,
                }),
            }),
            latency: Some(Duration::from_millis(7)),
        };
        let value = serde_json::to_value(&snapshot).expect("serializes");
        assert_eq!(value["status"]["state"], "ready");
        assert_eq!(value["status"]["active_model_id"], "llama-3b");
        // Load stays where every existing reader looks for it.
        assert_eq!(value["status"]["in_flight"], 1);
        assert_eq!(value["status"]["engine"], "camelid");
        assert_eq!(value["latency_ms"], 7);
        assert_eq!(value["spec"]["label"], "windows");
        assert_eq!(value["spec"]["engine"], "camelid");
    }

    #[test]
    fn an_offline_snapshot_serializes_its_reason_and_a_null_latency() {
        let snapshot = NodeSnapshot {
            spec: NodeSpec::camelid("mac", "192.0.2.10", 8181),
            status: NodeStatus::Unreachable {
                reason: "cannot connect".to_string(),
            },
            latency: None,
        };
        let value = serde_json::to_value(&snapshot).expect("serializes");
        assert_eq!(value["status"]["state"], "unreachable");
        assert_eq!(value["status"]["reason"], "cannot connect");
        assert!(value["latency_ms"].is_null());
    }
}
