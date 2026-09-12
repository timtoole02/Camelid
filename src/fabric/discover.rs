//! Finding machines to add, and adding exactly the one a person confirmed.
//!
//! This is the highest-risk surface in the lane: it is the only part that sends
//! traffic to machines the operator did not name. Four rules shape all of it.
//!
//! **Discovery proposes; a person disposes (I8).** Nothing here creates a node.
//! A scan produces *proposed declarations*, and one becomes a node only when
//! somebody confirms it and a line is appended. From then on it is probed
//! exactly like a hand-written line, and no discovery state lives anywhere.
//!
//! **There is no route from here to a credential.** [`scan`] and [`join`] take
//! a [`NodeTransport`], never a `Fabric`, and every request goes through
//! `http::request_anonymous_any`, whose signature has no bearer parameter. The
//! only bearer-shaped fact in this module is a `bool` saying whether one is
//! configured, which is what lets the warning be true without anything here
//! holding the secret.
//!
//! **Loopback is the whole default.** With no arguments the scan covers
//! 127.0.0.1 and [::1] on each engine's own port and sends nothing off the
//! machine. A LAN is reached only through an explicit range or host, and then
//! only after the same cleartext acknowledgement a node needs (I11).
//!
//! **A stranger's strings never reach the operator's file or terminal raw.**
//! Versions must match a grammar, hosts and labels must match another, terminal
//! output is escaped, and [`super::nodes::append_node`] re-parses the bytes it
//! composed and refuses unless they mean the old file plus exactly one node.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::cancel::Cancel;
use super::engine::NodeEngine;
use super::http::{self, AnonymousAnswer, ConnectOutcome, HttpError};
use super::identify::{
    self, Answer, Classification, EngineReport, Evidence, Outcome, VERSION_NOT_RECORDED,
};
use super::netscope::{self, RangeRefusal, MAX_ADDRESSES, MAX_PORTS};
use super::node::{self, NodeSpec};
use super::nodes::{self, AppendRefusal, Base};
use super::transport::NodeTransport;

/// Bounds on one scan.
///
/// Gentle on a consumer router's ARP and connection tables, and fast on a home
/// network. The effective rate against addresses that black-hole is
/// `min(connects_per_second, concurrency / connect_timeout)` — with these
/// values 128/s, not 200/s — so a /24 across three ports is about six seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Limits {
    pub max_addresses: usize,
    pub max_ports: usize,
    pub concurrency: usize,
    pub connects_per_second: usize,
    pub connect_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub max_body_bytes: usize,
    pub wall_clock_ms: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_addresses: MAX_ADDRESSES,
            max_ports: MAX_PORTS,
            concurrency: 32,
            connects_per_second: 200,
            connect_timeout_ms: 250,
            request_timeout_ms: 1500,
            max_body_bytes: 1024 * 1024,
            wall_clock_ms: 60_000,
        }
    }
}

impl Limits {
    fn connect_timeout(&self) -> Duration {
        Duration::from_millis(self.connect_timeout_ms)
    }

    fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }

    fn wall_clock(&self) -> Duration {
        Duration::from_millis(self.wall_clock_ms)
    }
}

/// What a scan was asked to cover.
///
/// The defaults are the whole safety story: `loopback` and `default_ports` on,
/// everything else empty, so a scan with no arguments sends nothing off this
/// machine. The two booleans are opt-*outs* rather than opt-ins so that a scan
/// can also be made to cover exactly what was named and nothing more.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Scope {
    #[serde(default)]
    pub ranges: Vec<String>,
    #[serde(default)]
    pub hosts: Vec<String>,
    #[serde(default)]
    pub ports: Vec<u16>,
    #[serde(default = "yes")]
    pub loopback: bool,
    #[serde(default = "yes")]
    pub default_ports: bool,
}

fn yes() -> bool {
    true
}

impl Default for Scope {
    fn default() -> Self {
        Self {
            ranges: Vec::new(),
            hosts: Vec::new(),
            ports: Vec::new(),
            loopback: true,
            default_ports: true,
        }
    }
}

/// Why a scan never started. Every one of these happens before a socket opens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeRefusal {
    Range(RangeRefusal),
    TooManyPorts { ports: usize, limit: usize },
    TooLarge { addresses: usize, limit: usize },
    /// The fabric's own fail-closed transport rule, unchanged, applied to a
    /// target this scan would otherwise have opened a socket to.
    TransportRefused(String),
    Unresolvable { host: String, detail: String },
    Nothing,
}

impl std::fmt::Display for ScopeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Range(refusal) => write!(f, "{refusal}"),
            Self::TooManyPorts { ports, limit } => write!(
                f,
                "that names {ports} ports and this build scans at most {limit} in one run"
            ),
            Self::TooLarge { addresses, limit } => write!(
                f,
                "that covers {addresses} addresses and this build scans at most {limit} in one \
                 run; it is refused rather than cut short, so nobody believes a range was \
                 covered that was not"
            ),
            Self::TransportRefused(detail) => write!(f, "{detail}"),
            Self::Unresolvable { host, detail } => {
                write!(f, "`{}` could not be resolved: {detail}", display_safe(host))
            }
            Self::Nothing => write!(
                f,
                "that scan would cover nothing; name a range with --cidr, a machine with --host, \
                 or leave the loopback defaults on"
            ),
        }
    }
}

impl ScopeRefusal {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Range(refusal) => refusal.code(),
            Self::TooLarge { .. } | Self::TooManyPorts { .. } => "scope_too_large",
            Self::TransportRefused(_) => "transport_refused",
            Self::Unresolvable { .. } | Self::Nothing => "scope_refused",
        }
    }
}

/// One address and port a scan will look at, and how it was named.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Target {
    addr: SocketAddr,
    /// The name an operator wrote that produced this address, if any. It is
    /// what goes in the `Host` header and what a TLS certificate is checked
    /// against, because it is the spelling the fabric itself would connect by.
    via_name: Option<String>,
}

/// A scan that has passed every bound and every transport check.
#[derive(Debug, Clone)]
pub struct Plan {
    targets: Vec<Target>,
    ports: Vec<u16>,
    ranges: Vec<String>,
    hosts: Vec<String>,
    addresses: usize,
    loopback: bool,
    default_ports: bool,
    limits: Limits,
}

impl Plan {
    pub fn addresses(&self) -> usize {
        self.addresses
    }

    pub fn probes(&self) -> usize {
        self.targets.len()
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Bound each connect attempt, within what this build will accept.
    pub fn with_connect_timeout_ms(mut self, milliseconds: u64) -> Self {
        self.limits.connect_timeout_ms = milliseconds.clamp(1, 2_000);
        self
    }

    /// Bound the whole scan more tightly than the default, so a test does not
    /// have to wait out a minute of wall clock.
    #[cfg(test)]
    fn within(mut self, wall_clock: Duration) -> Self {
        self.limits.wall_clock_ms = wall_clock.as_millis() as u64;
        self
    }
}

/// Discovery with its transport policy resolved once.
///
/// The public face of this module, and the reason it is the *only* one: a
/// caller outside this crate cannot name a [`NodeTransport`], so it cannot
/// assemble a scan that skips the fail-closed rule (I11). It states the two
/// flags it was given and gets back something that can only scan within them.
pub struct Session {
    transport: NodeTransport,
}

impl Session {
    /// Resolve the same transport policy a node hop is held to.
    ///
    /// The same call `fabric status` and `fabric serve` make, so a machine
    /// discovery will show you is one this fabric would agree to talk to.
    pub fn open(
        ca_file: Option<&Path>,
        allow_cleartext_remote: bool,
    ) -> std::io::Result<Self> {
        Ok(Self {
            transport: NodeTransport::resolve(ca_file, allow_cleartext_remote)?,
        })
    }

    pub fn transport_description(&self) -> &'static str {
        self.transport.description()
    }

    /// Whether this transport would let a scan reach off this machine at all.
    pub fn lan_permitted(&self) -> bool {
        self.transport
            .permitted_addresses(&[SocketAddr::from(([100, 64, 0, 1], 8181))])
            .is_ok()
    }

    pub fn plan(&self, scope: &Scope) -> Result<Plan, ScopeRefusal> {
        plan(scope, &self.transport)
    }

    pub fn scan(&self, plan: &Plan, cancel: &Cancel, context: &ScanContext) -> Discovery {
        let connector = Sockets::over(&self.transport);
        scan(plan, &connector, cancel, context, &self.transport)
    }

    pub fn join(
        &self,
        path: &Path,
        request: &JoinRequest,
        context: &ScanContext,
        cancel: &Cancel,
    ) -> Result<Joined, JoinRefusal> {
        let connector = Sockets::over(&self.transport);
        join(path, request, &self.transport, &connector, context, cancel)
    }
}

/// The ports a scan covers when it is not told otherwise.
///
/// Read off the engine rows rather than written out here, so a fourth engine
/// is a row in `engine.rs` and nothing else (S6).
fn default_ports() -> Vec<(NodeEngine, u16)> {
    NodeEngine::ALL
        .iter()
        .map(|engine| (*engine, engine.default_port()))
        .collect()
}

/// Work out everything a scan will touch, and refuse before touching any of it.
///
/// Crate-internal: [`Session`] is the public way in, and it is the only way in
/// on purpose — a caller who cannot name a [`NodeTransport`] cannot assemble a
/// scan that skips the transport check.
pub(crate) fn plan(scope: &Scope, transport: &NodeTransport) -> Result<Plan, ScopeRefusal> {
    plan_over(scope, transport, &default_ports())
}

/// [`plan`], with the engine default ports supplied.
///
/// The seam exists so a test can put a stub's ephemeral port where an engine's
/// default would be, and show that nothing here infers an engine from a port.
#[cfg(test)]
pub(crate) fn plan_with_default_ports(
    scope: &Scope,
    transport: &NodeTransport,
    defaults: &[(NodeEngine, u16)],
) -> Result<Plan, ScopeRefusal> {
    plan_over(scope, transport, defaults)
}

fn plan_over(
    scope: &Scope,
    transport: &NodeTransport,
    defaults: &[(NodeEngine, u16)],
) -> Result<Plan, ScopeRefusal> {
    let limits = Limits::default();

    let mut ports: Vec<u16> = Vec::new();
    if scope.default_ports {
        ports.extend(defaults.iter().map(|(_, port)| *port));
    }
    ports.extend(scope.ports.iter().copied());
    ports.sort_unstable();
    ports.dedup();
    if ports.len() > limits.max_ports {
        return Err(ScopeRefusal::TooManyPorts {
            ports: ports.len(),
            limit: limits.max_ports,
        });
    }

    // Address, then the name that produced it. A name is kept because a
    // finding's evidence has to say how the machine was reached.
    let mut addresses: Vec<(IpAddr, Option<String>)> = Vec::new();
    if scope.loopback {
        addresses.push((IpAddr::V4(Ipv4Addr::LOCALHOST), None));
        addresses.push((IpAddr::V6(Ipv6Addr::LOCALHOST), None));
    }
    for raw in &scope.ranges {
        let range = netscope::checked_range(raw).map_err(ScopeRefusal::Range)?;
        addresses.extend(range.hosts().into_iter().map(|ip| (IpAddr::V4(ip), None)));
    }
    for host in &scope.hosts {
        // Resolved on discovery's own pool, before anything is sent: every
        // address a name reaches is its own target, so a finding can never
        // describe one machine's address beside another's answer.
        let resolved = netscope::forward_lookup(host, 0, &Cancel::never()).map_err(|error| {
            ScopeRefusal::Unresolvable {
                host: host.clone(),
                detail: error.to_string(),
            }
        })?;
        if resolved.is_empty() {
            return Err(ScopeRefusal::Unresolvable {
                host: host.clone(),
                detail: "it resolved to no addresses".to_string(),
            });
        }
        addresses.extend(
            resolved
                .into_iter()
                .map(|addr| (addr.ip(), Some(host.clone()))),
        );
    }

    let mut seen = BTreeSet::new();
    addresses.retain(|(ip, _)| seen.insert(*ip));
    if addresses.is_empty() || ports.is_empty() {
        return Err(ScopeRefusal::Nothing);
    }
    if addresses.len() > limits.max_addresses {
        return Err(ScopeRefusal::TooLarge {
            addresses: addresses.len(),
            limit: limits.max_addresses,
        });
    }

    // The fabric's own fail-closed rule, applied per target before any socket
    // opens. Discovery does not get a looser policy than the nodes it finds
    // would be reached under (I11).
    let mut targets = Vec::with_capacity(addresses.len() * ports.len());
    for (ip, via_name) in &addresses {
        for port in &ports {
            let addr = SocketAddr::new(*ip, *port);
            transport
                .permitted_addresses(&[addr])
                .map_err(|error| ScopeRefusal::TransportRefused(error.to_string()))?;
            targets.push(Target {
                addr,
                via_name: via_name.clone(),
            });
        }
    }

    Ok(Plan {
        targets,
        ports,
        ranges: scope.ranges.clone(),
        hosts: scope.hosts.clone(),
        addresses: addresses.len(),
        loopback: scope.loopback,
        default_ports: scope.default_ports,
        limits,
    })
}

/// What a scan is allowed to do to a socket. The test seam for everything below.
pub(crate) trait Connector: Sync {
    /// Connect and close, writing nothing at all.
    fn connect_only(&self, addr: SocketAddr, timeout: Duration) -> ConnectOutcome;

    /// Send one credential-free GET to exactly this address.
    #[allow(clippy::too_many_arguments)]
    fn ask(
        &self,
        addr: SocketAddr,
        host_header: &str,
        tls_name: Option<&str>,
        path: &str,
        timeout: Duration,
        max_body: usize,
        cancel: &Cancel,
    ) -> Result<AnonymousAnswer, HttpError>;
}

/// The real connector. Every method is a thin call into `http`, and neither of
/// them can present a credential, because neither signature has one.
pub(crate) struct Sockets<'a> {
    transport: &'a NodeTransport,
}

impl<'a> Sockets<'a> {
    pub(crate) fn over(transport: &'a NodeTransport) -> Self {
        Self { transport }
    }
}

impl Connector for Sockets<'_> {
    fn connect_only(&self, addr: SocketAddr, timeout: Duration) -> ConnectOutcome {
        http::connect_only(addr, timeout)
    }

    fn ask(
        &self,
        addr: SocketAddr,
        host_header: &str,
        tls_name: Option<&str>,
        path: &str,
        timeout: Duration,
        max_body: usize,
        cancel: &Cancel,
    ) -> Result<AnonymousAnswer, HttpError> {
        http::request_anonymous_any(
            &[addr],
            host_header,
            tls_name,
            path,
            timeout,
            max_body,
            self.transport,
            cancel,
        )
        .map(|(answer, _)| answer)
    }
}

/// What the caller knows that the scan itself cannot see.
#[derive(Debug, Clone, Default)]
pub struct ScanContext {
    /// The nodes already declared, so a finding can say it is one of them.
    pub existing: Vec<NodeSpec>,
    pub nodes_file: Option<PathBuf>,
    /// Whether the process that would probe a joined node holds a bearer.
    /// `None` where the caller never reads one and so cannot know.
    pub bearer_configured: Option<bool>,
    pub reverse_dns: bool,
}

/// Addresses that answered nothing, split by what happened.
///
/// Split rather than summed because "nothing answered" and "this machine was
/// not allowed to look" are different facts, and on macOS the second is
/// reported as the first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct NotListed {
    pub refused: usize,
    pub timed_out: usize,
    pub unreachable: usize,
    pub other: usize,
}

impl NotListed {
    fn total(&self) -> usize {
        self.refused + self.timed_out + self.unreachable + self.other
    }

    fn record(&mut self, outcome: &ConnectOutcome) {
        match outcome {
            ConnectOutcome::Open => {}
            ConnectOutcome::Refused => self.refused += 1,
            ConnectOutcome::TimedOut => self.timed_out += 1,
            ConnectOutcome::Unreachable => self.unreachable += 1,
            ConnectOutcome::Other(_) => self.other += 1,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ScopeReport {
    pub ranges: Vec<String>,
    pub hosts: Vec<String>,
    pub loopback: bool,
    pub default_ports: bool,
    pub addresses: usize,
    pub ports: Vec<u16>,
    pub limits: Limits,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodesFileReport {
    pub path: String,
    pub sha256: Option<String>,
}

/// A name a machine might be listed under, and what was actually established
/// about it.
///
/// An absent name is a name plus a reason, never an empty string: "no reverse
/// DNS record" and "the name resolves somewhere else" lead to different
/// actions, and both differ from "we did not look".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct NameProof {
    pub name: Option<String>,
    pub source: Option<&'static str>,
    pub source_trust: Option<&'static str>,
    pub proof: Option<&'static str>,
    pub resolved: Option<Vec<String>>,
    pub why: Option<String>,
    /// A name that exists and was not good enough to list, escaped.
    pub rejected: Option<RejectedName>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RejectedName {
    pub proof: &'static str,
    pub escaped: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HostAlternative {
    pub host: String,
    pub label: String,
    pub source_trust: &'static str,
    pub warning: &'static str,
}

/// A node line somebody may choose to add, and everything they should know
/// before they do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Proposal {
    pub label: String,
    pub engine: NodeEngine,
    pub host: String,
    pub port: u16,
    pub line: String,
    /// The comment as it will be written, except for its timestamp, which is
    /// filled in at the moment of writing.
    pub comment_preview: String,
    /// Engines a person must choose between. Empty unless the address matched
    /// more than one, and never resolved by taking the first.
    pub engine_choices: Vec<NodeEngine>,
    pub host_alternatives: Vec<HostAlternative>,
    /// Decided on the server from facts it holds. A page renders these
    /// verbatim and never derives one from the engine name.
    pub warnings: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InFabric {
    pub label: Option<String>,
    pub declared_engine: Option<NodeEngine>,
    pub agrees: Option<bool>,
    /// Set when an existing spec could not be resolved, so whether this address
    /// is already in the fabric is unknown rather than false.
    pub unknown: Option<String>,
}

/// One address that answered, and everything established about it.
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub id: String,
    pub address: String,
    pub port: u16,
    /// Every address this one machine answered on, where they were merged.
    pub addresses: Vec<String>,
    pub via_name: Option<String>,
    pub this_machine: Option<bool>,
    pub identity_basis: &'static str,
    pub tls_name_used: Option<String>,
    pub engines: Vec<EngineReport>,
    pub classification: Classification,
    pub evidence: Vec<Evidence>,
    pub name: NameProof,
    pub in_fabric: Option<InFabric>,
    pub possibly_same_as: Vec<String>,
    pub proposal: Option<Proposal>,
    pub not_proposed: Option<String>,
}

/// Everything one scan established. The single wire shape: the CLI's `--json`
/// and the proxy route serialize this same value, so the two front doors
/// cannot drift apart.
#[derive(Debug, Clone, Serialize)]
pub struct Discovery {
    pub scope: ScopeReport,
    pub transport: &'static str,
    pub credentials_presented: &'static str,
    pub user_agent: &'static str,
    pub elapsed_ms: u64,
    pub planned: usize,
    pub probes: usize,
    pub not_listed: NotListed,
    pub not_scanned: usize,
    pub nodes_file: Option<NodesFileReport>,
    pub findings: Vec<Finding>,
    /// Said only when the numbers support it.
    pub hint: Option<&'static str>,
}

/// What one target's probe established.
enum Probe {
    Answered {
        target: Target,
        answers: Vec<Answer>,
        tls_name_used: Option<String>,
    },
    Silent(ConnectOutcome),
}

/// Every path any engine wants asked, each asked once.
fn union_paths() -> Vec<&'static str> {
    let mut paths = Vec::new();
    for engine in NodeEngine::ALL {
        for path in engine.identification_paths() {
            if !paths.contains(path) {
                paths.push(*path);
            }
        }
    }
    paths
}

/// Paces new connections so a scan is gentle on a consumer router's ARP and
/// connection tables.
///
/// Each caller reserves *the next slot* rather than being told how long to
/// wait. With 32 workers the two are not the same thing: per-caller waits are
/// computed at the same instant and therefore elapse at the same instant, so
/// the connects go out in a burst anyway — which is exactly what the limit
/// exists to prevent. Handing out increasing moments makes the rate hold
/// across the whole scan rather than on average.
struct Pacer {
    interval: Duration,
    next: Instant,
}

impl Pacer {
    fn new(per_second: usize) -> Self {
        Self {
            interval: Duration::from_secs_f64(1.0 / per_second.max(1) as f64),
            next: Instant::now(),
        }
    }

    /// The moment this caller may open its connection.
    fn reserve(&mut self) -> Instant {
        let at = self.next.max(Instant::now());
        self.next = at + self.interval;
        at
    }
}

/// Look at everything the plan covers, and report what answered.
///
/// Two stages, both bound to exact addresses. Stage one is a connect that
/// writes nothing, so an address that is not an engine is never sent a byte.
/// Stage two asks only the addresses stage one found open, at the socket stage
/// one found them on.
pub(crate) fn scan(
    plan: &Plan,
    connector: &dyn Connector,
    cancel: &Cancel,
    context: &ScanContext,
    transport: &NodeTransport,
) -> Discovery {
    let started = Instant::now();
    let deadline = started + plan.limits.wall_clock();
    let next = AtomicUsize::new(0);
    let pacer = Mutex::new(Pacer::new(plan.limits.connects_per_second));
    let probed: Mutex<Vec<Probe>> = Mutex::new(Vec::new());

    let workers = plan.limits.concurrency.min(plan.targets.len()).max(1);
    std::thread::scope(|threads| {
        for _ in 0..workers {
            threads.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::SeqCst);
                let Some(target) = plan.targets.get(index) else {
                    return;
                };
                // Anything past here is counted as not scanned rather than
                // dropped: a cap must never read as coverage.
                if cancel.is_cancelled() || Instant::now() >= deadline {
                    return;
                }
                let at = pacer
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .reserve();
                let wait = at.saturating_duration_since(Instant::now());
                if !wait.is_zero() {
                    std::thread::sleep(wait.min(deadline.saturating_duration_since(Instant::now())));
                }
                let probe = probe_target(target, plan, connector, cancel, transport);
                probed
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(probe);
            });
        }
    });

    let probes = probed.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut not_listed = NotListed::default();
    let mut answered = Vec::new();
    for probe in probes {
        match probe {
            Probe::Answered {
                target,
                answers,
                tls_name_used,
            } => answered.push((target, answers, tls_name_used)),
            Probe::Silent(outcome) => not_listed.record(&outcome),
        }
    }

    let scanned = answered.len() + not_listed.total();
    let findings = assemble(answered, context, transport, cancel);

    Discovery {
        scope: ScopeReport {
            ranges: plan.ranges.clone(),
            hosts: plan.hosts.iter().map(|host| display_safe(host).into_owned()).collect(),
            loopback: plan.loopback,
            default_ports: plan.default_ports,
            addresses: plan.addresses,
            ports: plan.ports.clone(),
            limits: plan.limits,
        },
        transport: transport.description(),
        credentials_presented: "none",
        user_agent: http::DISCOVERY_USER_AGENT,
        elapsed_ms: started.elapsed().as_millis() as u64,
        planned: plan.targets.len(),
        probes: scanned,
        not_listed,
        not_scanned: plan.targets.len().saturating_sub(scanned),
        nodes_file: context.nodes_file.as_ref().map(|path| NodesFileReport {
            path: path.display().to_string(),
            sha256: nodes::file_sha256(path),
        }),
        findings,
        hint: hint_for(&not_listed, plan),
    }
}

/// The one hint worth giving, and only when the numbers support it.
fn hint_for(not_listed: &NotListed, plan: &Plan) -> Option<&'static str> {
    let off_box = plan
        .targets
        .iter()
        .filter(|target| !target.addr.ip().is_loopback())
        .count();
    (off_box > 0 && not_listed.unreachable >= off_box).then_some(
        "every address off this machine answered `no route to host`. On macOS that is also what a \
         denied Local Network permission looks like: check System Settings > Privacy & Security > \
         Local Network for the program running this scan",
    )
}

fn probe_target(
    target: &Target,
    plan: &Plan,
    connector: &dyn Connector,
    cancel: &Cancel,
    transport: &NodeTransport,
) -> Probe {
    let outcome = connector.connect_only(target.addr, plan.limits.connect_timeout());
    if outcome != ConnectOutcome::Open {
        return Probe::Silent(outcome);
    }

    // The `Host` header is the spelling the fabric itself would connect by, and
    // under a pinned CA it is also the name the certificate must be signed for.
    let authority = match &target.via_name {
        Some(name) => format!("{name}:{}", target.addr.port()),
        None => target.addr.to_string(),
    };
    let tls_name = transport
        .tls_config()
        .is_some()
        .then(|| target.via_name.clone())
        .flatten();

    let answers = union_paths()
        .into_iter()
        .map(|path| {
            let outcome = match connector.ask(
                target.addr,
                &authority,
                tls_name.as_deref(),
                path,
                plan.limits.request_timeout(),
                plan.limits.max_body_bytes,
                cancel,
            ) {
                Ok(answer) => Outcome::Http {
                    status: answer.status,
                    content_type: answer.content_type.clone(),
                    json: serde_json::from_slice(&answer.body).ok(),
                    bytes_len: answer.body.len(),
                },
                Err(HttpError::Tls(detail)) => Outcome::TlsRefused(detail),
                Err(HttpError::Malformed(detail)) => {
                    if detail.contains("nothing at all") {
                        Outcome::Silent
                    } else {
                        Outcome::NotHttp(detail)
                    }
                }
                Err(error) => Outcome::Unanswered(error.to_string()),
            };
            Answer::new(path, outcome)
        })
        .collect();

    Probe::Answered {
        target: target.clone(),
        answers,
        tls_name_used: tls_name,
    }
}

/// Turn what answered into rows a person can act on.
fn assemble(
    answered: Vec<(Target, Vec<Answer>, Option<String>)>,
    context: &ScanContext,
    transport: &NodeTransport,
    cancel: &Cancel,
) -> Vec<Finding> {
    let existing = resolved_existing(&context.existing, cancel);
    let authenticated = transport.tls_config().is_some();

    let mut findings: Vec<Finding> = Vec::new();
    for (index, (target, answers, tls_name_used)) in answered.into_iter().enumerate() {
        let identification = identify::classify(&answers);
        let this_machine = netscope::is_this_machine(target.addr.ip());
        let name = prove_name(&target, context, cancel);
        let in_fabric = in_fabric(&existing, &target, &identification.classification);

        findings.push(Finding {
            id: format!("f{index}"),
            address: target.addr.ip().to_string(),
            port: target.addr.port(),
            addresses: vec![target.addr.ip().to_string()],
            via_name: target.via_name.clone(),
            this_machine,
            identity_basis: if authenticated && tls_name_used.is_some() {
                "certificate_verified"
            } else {
                "unauthenticated_answer"
            },
            tls_name_used,
            engines: identification.engines,
            classification: identification.classification,
            evidence: identification.evidence,
            name,
            in_fabric,
            possibly_same_as: Vec::new(),
            proposal: None,
            not_proposed: None,
        });
    }

    merge_local_findings(&mut findings);
    note_lookalikes(&mut findings);

    // Proposals last, so a label can be checked against every other one this
    // scan is about to suggest as well as against the file.
    let mut taken: Vec<String> = context
        .existing
        .iter()
        .map(|spec| spec.label.clone())
        .collect();
    let loopback_answers = loopback_answers(&findings);
    for finding in &mut findings {
        let (proposal, not_proposed) =
            propose(finding, context, transport, &mut taken, &loopback_answers);
        finding.proposal = proposal;
        finding.not_proposed = not_proposed;
    }
    findings
}

/// Every existing spec, with the addresses its host resolves to now.
///
/// A spec whose host does not resolve gets `None`, which makes "is this already
/// in the fabric" unknown for it rather than false.
fn resolved_existing(
    existing: &[NodeSpec],
    cancel: &Cancel,
) -> Vec<(NodeSpec, Option<Vec<IpAddr>>)> {
    existing
        .iter()
        .map(|spec| {
            let resolved = netscope::forward_lookup(&spec.host, spec.port, cancel)
                .ok()
                .map(|addrs| addrs.into_iter().map(|addr| addr.ip()).collect());
            (spec.clone(), resolved)
        })
        .collect()
}

fn in_fabric(
    existing: &[(NodeSpec, Option<Vec<IpAddr>>)],
    target: &Target,
    classification: &Classification,
) -> Option<InFabric> {
    let mut unknown = None;
    for (spec, resolved) in existing {
        let Some(resolved) = resolved else {
            unknown.get_or_insert_with(|| {
                format!("`{}` did not resolve", display_safe(&spec.label))
            });
            continue;
        };
        if spec.port != target.addr.port() {
            continue;
        }
        // Every address of this machine is the same machine, so an engine
        // declared on loopback and found on the LAN address is one node.
        let same = resolved.contains(&target.addr.ip())
            || (resolved.iter().any(|address| address.is_loopback())
                && netscope::is_this_machine(target.addr.ip()) == Some(true));
        if same {
            return Some(InFabric {
                label: Some(spec.label.clone()),
                declared_engine: Some(spec.engine),
                agrees: classification
                    .matched_engine()
                    .map(|engine| engine == spec.engine),
                unknown: None,
            });
        }
    }
    unknown.map(|detail| InFabric {
        label: None,
        declared_engine: None,
        agrees: None,
        unknown: Some(detail),
    })
}

/// Fold findings that are this machine, on one port, into one row.
///
/// An engine bound to every interface answers on loopback and on the LAN
/// address. Left alone that is two proposals for one server, and a fabric that
/// took both would believe it had two machines and re-place a failed request
/// onto the one that just failed.
fn merge_local_findings(findings: &mut Vec<Finding>) {
    let mut merged: Vec<Finding> = Vec::new();
    for finding in findings.drain(..) {
        let local = finding.this_machine == Some(true);
        let into = merged.iter_mut().find(|kept| {
            kept.this_machine == Some(true)
                && local
                && kept.port == finding.port
                && kept.classification == finding.classification
        });
        match into {
            Some(kept) => {
                for address in finding.addresses {
                    if !kept.addresses.contains(&address) {
                        kept.addresses.push(address);
                    }
                }
                // A name proven for either address describes the machine.
                if kept.name.name.is_none() && finding.name.name.is_some() {
                    kept.name = finding.name;
                }
            }
            None => merged.push(finding),
        }
    }
    *findings = merged;
}

/// Note rows that answer identically, without claiming they are one machine.
fn note_lookalikes(findings: &mut [Finding]) {
    let signature = |finding: &Finding| -> Option<(String, Vec<String>)> {
        let engine = finding.classification.matched_engine()?;
        let facts: Vec<String> = finding
            .evidence
            .iter()
            .map(|evidence| format!("{} {}", evidence.request, evidence.fact))
            .collect();
        Some((engine.as_str().to_string(), facts))
    };
    let signatures: Vec<Option<(String, Vec<String>)>> =
        findings.iter().map(&signature).collect();
    let addresses: Vec<String> = findings.iter().map(|finding| finding.address.clone()).collect();
    let ids: Vec<String> = findings.iter().map(|finding| finding.id.clone()).collect();

    for index in 0..findings.len() {
        let Some(mine) = &signatures[index] else {
            continue;
        };
        let like: Vec<String> = (0..findings.len())
            .filter(|other| *other != index)
            .filter(|other| signatures[*other].as_ref() == Some(mine))
            .filter(|other| addresses[*other] != addresses[index])
            .map(|other| ids[other].clone())
            .collect();
        findings[index].possibly_same_as = like;
    }
}

/// Which ports answered on loopback in this scan, and as what.
fn loopback_answers(findings: &[Finding]) -> BTreeMap<u16, Option<NodeEngine>> {
    findings
        .iter()
        .filter(|finding| {
            finding
                .addresses
                .iter()
                .filter_map(|address| address.parse::<IpAddr>().ok())
                .any(|address| address.is_loopback())
        })
        .map(|finding| (finding.port, finding.classification.matched_engine()))
        .collect()
}

/// A name is listed only when a forward lookup on this host returns the address
/// that actually answered.
fn prove_name(target: &Target, context: &ScanContext, cancel: &Cancel) -> NameProof {
    // A name the operator typed is used as given: they named the machine, so
    // the name is theirs rather than the device's.
    if let Some(name) = &target.via_name {
        return proven(name, "operator", "operator", target, cancel);
    }
    if !context.reverse_dns {
        return NameProof {
            why: Some("reverse DNS disabled".to_string()),
            ..NameProof::default()
        };
    }
    let Some(candidate) = netscope::reverse_lookup(target.addr.ip(), cancel) else {
        return NameProof {
            why: Some("no reverse DNS record".to_string()),
            ..NameProof::default()
        };
    };
    // A name that is not writable never appears unescaped anywhere, and is
    // never a candidate for the file.
    if !node::is_writable_host(&candidate) {
        return NameProof {
            why: Some("the name is not one this build will write".to_string()),
            rejected: Some(RejectedName {
                proof: "not_a_writable_hostname",
                escaped: display_safe(&candidate).into_owned(),
            }),
            ..NameProof::default()
        };
    }
    proven(&candidate, "reverse_dns", "device_claimed", target, cancel)
}

fn proven(
    name: &str,
    source: &'static str,
    source_trust: &'static str,
    target: &Target,
    cancel: &Cancel,
) -> NameProof {
    match netscope::forward_lookup(name, target.addr.port(), cancel) {
        Ok(resolved) => {
            let addresses: Vec<IpAddr> = resolved.iter().map(|addr| addr.ip()).collect();
            if addresses.contains(&target.addr.ip()) {
                NameProof {
                    name: Some(name.to_string()),
                    source: Some(source),
                    source_trust: Some(source_trust),
                    proof: Some("resolves_to_this_address"),
                    resolved: Some(addresses.iter().map(ToString::to_string).collect()),
                    why: None,
                    rejected: None,
                }
            } else {
                NameProof {
                    why: Some("the name resolves somewhere else".to_string()),
                    rejected: Some(RejectedName {
                        proof: "resolves_elsewhere",
                        escaped: display_safe(name).into_owned(),
                    }),
                    ..NameProof::default()
                }
            }
        }
        Err(error) => {
            let timed_out = error.to_string().contains("deadline");
            NameProof {
                why: Some(if timed_out {
                    "lookup timed out".to_string()
                } else {
                    "the name does not resolve".to_string()
                }),
                rejected: Some(RejectedName {
                    proof: if timed_out {
                        "lookup_timed_out"
                    } else {
                        "does_not_resolve"
                    },
                    escaped: display_safe(name).into_owned(),
                }),
                ..NameProof::default()
            }
        }
    }
}

/// What, if anything, this finding may be added as.
fn propose(
    finding: &Finding,
    context: &ScanContext,
    transport: &NodeTransport,
    taken: &mut Vec<String>,
    loopback_answers: &BTreeMap<u16, Option<NodeEngine>>,
) -> (Option<Proposal>, Option<String>) {
    if let Some(in_fabric) = &finding.in_fabric {
        if let Some(label) = &in_fabric.label {
            return (
                None,
                Some(format!(
                    "already in this fabric as `{}`",
                    display_safe(label)
                )),
            );
        }
    }

    let (engine, choices) = match &finding.classification {
        Classification::AnswersLike { engine, .. } => (*engine, Vec::new()),
        Classification::Ambiguous { engines } => match engines.first() {
            Some(first) => (*first, engines.clone()),
            None => return (None, None),
        },
        Classification::FabricProxy { .. } => {
            return (
                None,
                Some(
                    "a fabric proxy is not a node; point the Cluster view at it instead"
                        .to_string(),
                ),
            )
        }
        Classification::Incomplete { unanswered, .. } => {
            return (
                None,
                Some(format!(
                    "a check did not finish ({}), so what this is has not been established; \
                     scan again",
                    unanswered.join(", ")
                )),
            )
        }
        Classification::RequiresCredentials { paths } => {
            return (
                None,
                Some(format!(
                    "it asked for a credential on {}; this fabric holds none for a machine it \
                     has not been told about",
                    paths.join(", ")
                )),
            )
        }
        Classification::OtherHttp { .. } => {
            return (
                None,
                Some(
                    "it speaks HTTP and matched no engine this build knows; add it by hand if \
                     you know what it is"
                        .to_string(),
                ),
            )
        }
        Classification::NotHttp { .. } | Classification::SilentAfterConnect => {
            return (None, Some("it did not answer as HTTP".to_string()))
        }
        Classification::TlsNotAuthenticated { .. } => {
            return (
                None,
                Some("its certificate did not authenticate against the pinned CA".to_string()),
            )
        }
    };

    // Loopback is proposed for this machine only when loopback was actually
    // seen to answer the same way on this port. An engine bound only to the LAN
    // address is not reachable on 127.0.0.1, and a line saying it is would
    // simply not work.
    let host = if finding.this_machine == Some(true)
        && loopback_answers.get(&finding.port) == Some(&Some(engine))
    {
        Ipv4Addr::LOCALHOST.to_string()
    } else {
        finding.address.clone()
    };
    let host = writable_host(&host);

    // A device-claimed name is offered, never defaulted: on a consumer router
    // the device chooses its own name, so anything on the network can claim it
    // later and receive this node's prompts.
    let mut alternatives = Vec::new();
    if let (Some(name), Some(trust)) = (&finding.name.name, finding.name.source_trust) {
        if trust == "device_claimed" {
            alternatives.push(HostAlternative {
                host: name.clone(),
                label: label_for(name, engine, &[]),
                source_trust: trust,
                warning: "this name comes from your router; the device chooses it, and anything \
                          on this network can claim it later",
            });
        }
    }
    // An operator's own name, or one a pinned CA authenticated, is the default.
    let host = match (&finding.name.name, finding.name.source_trust) {
        (Some(name), Some("operator" | "certificate_verified")) => name.clone(),
        _ => host,
    };

    let label = label_for(&host, engine, taken);
    taken.push(label.clone());

    let mut warnings: Vec<&'static str> = Vec::new();
    if transport.tls_config().is_none() {
        warnings.push("cleartext");
    }
    if let Some(warning) = engine.bearer_warning(context.bearer_configured) {
        warnings.push(warning);
    }
    if finding
        .name
        .resolved
        .as_ref()
        .is_some_and(|resolved| resolved.len() > 1)
    {
        warnings.push("name_resolves_to_several_addresses");
    }

    let spec = NodeSpec {
        label: label.clone(),
        host: host.clone(),
        port: finding.port,
        engine,
    };
    let version = finding
        .engines
        .iter()
        .find(|report| report.engine == engine)
        .and_then(|report| report.verdict.recorded_version());

    (
        Some(Proposal {
            line: spec.to_line(),
            comment_preview: provenance_comment(
                "<time of writing>",
                &format!("{}:{}", finding.address, finding.port),
                engine,
                version,
            ),
            label,
            engine,
            host,
            port: finding.port,
            engine_choices: choices,
            host_alternatives: alternatives,
            warnings,
        }),
        None,
    )
}

/// A host string that is safe to write, falling back to the address shape.
fn writable_host(host: &str) -> String {
    if node::is_writable_host(host) {
        return host.to_string();
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V6(address)) => format!("[{address}]"),
        Ok(IpAddr::V4(address)) => address.to_string(),
        Err(_) => host.to_string(),
    }
}

/// A label derived from the host somebody is actually going to use, never from
/// a name they did not choose.
fn label_for(host: &str, engine: NodeEngine, taken: &[String]) -> String {
    let stem = if let Ok(address) = host.parse::<IpAddr>() {
        if address.is_loopback() {
            "local".to_string()
        } else {
            format!(
                "host-{}",
                address.to_string().replace(['.', ':'], "-").trim_matches('-')
            )
        }
    } else if host.starts_with('[') {
        "host".to_string()
    } else {
        host.split('.').next().unwrap_or("host").to_string()
    };

    let cleaned: String = stem
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-'))
        .collect();
    let base = format!(
        "{}-{}",
        if cleaned.is_empty() { "host" } else { &cleaned },
        engine.as_str()
    );
    let base = if node::is_writable_label(&base) {
        base
    } else {
        format!("host-{}", engine.as_str())
    };

    if !taken.contains(&base) {
        return base;
    }
    (2..)
        .map(|suffix| format!("{base}-{suffix}"))
        .find(|candidate| !taken.contains(candidate))
        .unwrap_or(base)
}

/// The one comment a join writes, built only from facts this build wrote.
///
/// The version is the single third-party string that reaches it, and only
/// through the grammar. A version of `0.1\nx=camelid://…` would otherwise split
/// this comment into a second, meaningful node line — one the file's own parser
/// accepts, and one the fabric would then send its bearer to on every poll.
pub(crate) fn provenance_comment(
    at: &str,
    endpoint: &str,
    engine: NodeEngine,
    version: Option<&str>,
) -> String {
    let comment = format!(
        "# joined by fabric discover {at}: {endpoint} answered like {} {}",
        engine.as_str(),
        version
            .and_then(identify::recorded_version)
            .unwrap_or(VERSION_NOT_RECORDED)
    );
    debug_assert!(
        !comment.chars().any(char::is_control),
        "a provenance comment must never carry a control character"
    );
    comment
}

/// Escape everything outside printable ASCII, so a stranger's string cannot
/// rewrite what an operator is reading.
pub fn display_safe(raw: &str) -> Cow<'_, str> {
    if raw
        .chars()
        .all(|character| character == ' ' || character.is_ascii_graphic())
    {
        return Cow::Borrowed(raw);
    }
    Cow::Owned(
        raw.chars()
            .map(|character| {
                if character == ' ' || character.is_ascii_graphic() {
                    character.to_string()
                } else {
                    character.escape_default().to_string()
                }
            })
            .collect(),
    )
}

/// One node somebody has confirmed they want added.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct JoinRequest {
    pub label: String,
    pub engine: String,
    pub host: String,
    pub port: u16,
    /// What the file was when they were shown it. Absent means "there is no
    /// file yet", which only the CLI may say.
    #[serde(default)]
    pub base_sha256: Option<String>,
    /// The socket the scan recorded. The join refuses if the name now reaches
    /// a different one.
    #[serde(default)]
    pub scanned_address: Option<String>,
}

/// A node file that gained exactly one node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Joined {
    pub written: bool,
    pub path: String,
    pub appended: String,
    pub line: String,
    pub answered_from: String,
    pub sha256_before: Option<String>,
    pub sha256_after: String,
    pub note: &'static str,
}

/// Why nothing was added. Every one leaves the file byte-identical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinRefusal {
    InvalidLabel(String),
    InvalidHost(String),
    NotAnEngine(String),
    NameNotProven { host: String, detail: String },
    DuplicateEndpoint { existing_label: String },
    NoLongerAnswers { expected: NodeEngine, found: String },
    NameReachesAnotherAddress { scanned: String, answered_from: String },
    TransportRefused(String),
    Append(AppendRefusal),
}

impl std::fmt::Display for JoinRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLabel(label) => write!(
                f,
                "`{}` is not a label this build will write; use letters, digits, dot, underscore \
                 and dash, starting with a letter or digit",
                display_safe(label)
            ),
            Self::InvalidHost(host) => write!(
                f,
                "`{}` is not a host this build will write; use a hostname, a dotted IPv4 address, \
                 or a bracketed IPv6 address",
                display_safe(host)
            ),
            Self::NotAnEngine(engine) => write!(
                f,
                "`{}` is not an engine this fabric can read; known engines are {}",
                display_safe(engine),
                NodeEngine::known().join(", ")
            ),
            Self::NameNotProven { host, detail } => write!(
                f,
                "`{}` could not be proven to reach that machine from here: {detail}",
                display_safe(host)
            ),
            Self::DuplicateEndpoint { existing_label } => write!(
                f,
                "that endpoint is already in this fabric as `{}`. Two labels for one server is a \
                 deliberate arrangement rather than an accident, so add the second by hand if \
                 that is what you want",
                display_safe(existing_label)
            ),
            Self::NoLongerAnswers { expected, found } => write!(
                f,
                "it no longer answers like {expected}; right now it answers as {found}. Nothing \
                 was added"
            ),
            Self::NameReachesAnotherAddress {
                scanned,
                answered_from,
            } => write!(
                f,
                "that name now reaches {answered_from}, and the machine that was scanned was \
                 {scanned}. Nothing was added; add the address itself if that is what you meant"
            ),
            Self::TransportRefused(detail) => write!(f, "{detail}"),
            Self::Append(refusal) => write!(f, "{refusal}"),
        }
    }
}

impl JoinRefusal {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidLabel(_) => "invalid_label",
            Self::InvalidHost(_) => "invalid_host",
            Self::NotAnEngine(_) => "not_an_engine",
            Self::NameNotProven { .. } => "name_not_proven",
            Self::DuplicateEndpoint { .. } => "duplicate_endpoint",
            Self::NoLongerAnswers { .. } => "no_longer_answers",
            Self::NameReachesAnotherAddress { .. } => "name_reaches_another_address",
            Self::TransportRefused(_) => "transport_refused",
            Self::Append(refusal) => refusal.code(),
        }
    }
}

/// Add exactly one confirmed node to the nodes file.
///
/// Everything is re-checked here, after the click: the grammars, that the
/// endpoint is not already in the file, that the name still reaches the machine
/// that was scanned, and that the machine still answers like the engine being
/// written. The re-identification is anonymous, like the scan.
pub(crate) fn join(
    path: &Path,
    request: &JoinRequest,
    transport: &NodeTransport,
    connector: &dyn Connector,
    context: &ScanContext,
    cancel: &Cancel,
) -> Result<Joined, JoinRefusal> {
    if !node::is_writable_label(&request.label) {
        return Err(JoinRefusal::InvalidLabel(request.label.clone()));
    }
    if !node::is_writable_host(&request.host) {
        return Err(JoinRefusal::InvalidHost(request.host.clone()));
    }
    let engine = NodeEngine::from_scheme(&request.engine)
        .ok_or_else(|| JoinRefusal::NotAnEngine(request.engine.clone()))?;

    // Resolved in the fabric's own order, so the address that answers here is
    // the one the fabric's own connect would reach first.
    let resolved = netscope::forward_lookup(&request.host, request.port, cancel).map_err(
        |error| JoinRefusal::NameNotProven {
            host: request.host.clone(),
            detail: error.to_string(),
        },
    )?;
    if resolved.is_empty() {
        return Err(JoinRefusal::NameNotProven {
            host: request.host.clone(),
            detail: "it resolves to no addresses".to_string(),
        });
    }
    let permitted = transport
        .permitted_addresses(&resolved)
        .map_err(|error| JoinRefusal::TransportRefused(error.to_string()))?;

    let addresses: Vec<IpAddr> = permitted.iter().map(|addr| addr.ip()).collect();
    let existing = resolved_existing(&context.existing, cancel);
    let comparable: Vec<(NodeSpec, Vec<IpAddr>)> = existing
        .into_iter()
        .filter_map(|(spec, resolved)| resolved.map(|resolved| (spec, resolved)))
        .collect();
    if let Some(existing_label) = nodes::endpoint_conflict(&comparable, request.port, &addresses) {
        return Err(JoinRefusal::DuplicateEndpoint { existing_label });
    }

    // Re-identified anonymously, through the name, before anything is written.
    let authority = format!("{}:{}", request.host, request.port);
    let tls_name = transport
        .tls_config()
        .is_some()
        .then(|| request.host.clone());
    let limits = Limits::default();
    let mut answered_from = None;
    let answers: Vec<Answer> = union_paths()
        .into_iter()
        .map(|path| {
            let outcome = match first_answer(
                connector,
                &permitted,
                &authority,
                tls_name.as_deref(),
                path,
                &limits,
                cancel,
            ) {
                Ok((answer, from)) => {
                    answered_from.get_or_insert(from);
                    Outcome::Http {
                        status: answer.status,
                        content_type: answer.content_type.clone(),
                        json: serde_json::from_slice(&answer.body).ok(),
                        bytes_len: answer.body.len(),
                    }
                }
                Err(HttpError::Tls(detail)) => Outcome::TlsRefused(detail),
                Err(error) => Outcome::Unanswered(error.to_string()),
            };
            Answer::new(path, outcome)
        })
        .collect();

    let identification = identify::classify(&answers);
    let agrees = match &identification.classification {
        Classification::AnswersLike { engine: found, .. } => *found == engine,
        Classification::Ambiguous { engines } => engines.contains(&engine),
        _ => false,
    };
    if !agrees {
        return Err(JoinRefusal::NoLongerAnswers {
            expected: engine,
            found: identification.classification.kind().to_string(),
        });
    }

    let answered_from = answered_from.ok_or_else(|| JoinRefusal::NoLongerAnswers {
        expected: engine,
        found: "nothing that answered".to_string(),
    })?;
    if let Some(scanned) = &request.scanned_address {
        if scanned != &answered_from.to_string() {
            return Err(JoinRefusal::NameReachesAnotherAddress {
                scanned: scanned.clone(),
                answered_from: answered_from.to_string(),
            });
        }
    }

    let version = identification
        .engines
        .iter()
        .find(|report| report.engine == engine)
        // Reported, not recorded: the grammar is applied inside the comment
        // builder, which is the one place it has to hold.
        .and_then(|report| report.verdict.reported_version());
    let comment = provenance_comment(
        &crate::receipt::rfc3339_utc_now(),
        &answered_from.to_string(),
        engine,
        version,
    );

    let spec = NodeSpec {
        label: request.label.clone(),
        host: request.host.clone(),
        port: request.port,
        engine,
    };
    let base = match &request.base_sha256 {
        Some(hash) => Base::Sha256(hash.clone()),
        None => Base::Absent,
    };
    let appended = nodes::append_node(path, &spec, &base, &comment).map_err(JoinRefusal::Append)?;

    Ok(Joined {
        written: true,
        path: appended.path,
        appended: appended.appended,
        line: appended.line,
        answered_from: answered_from.to_string(),
        sha256_before: appended.sha256_before,
        sha256_after: appended.sha256_after,
        note: "the proxy re-reads this file within 1 s; the node appears in /v1/health once it has",
    })
}

/// Ask each address in turn and return the first that answers, with its socket.
fn first_answer(
    connector: &dyn Connector,
    addrs: &[SocketAddr],
    authority: &str,
    tls_name: Option<&str>,
    path: &str,
    limits: &Limits,
    cancel: &Cancel,
) -> Result<(AnonymousAnswer, SocketAddr), HttpError> {
    let mut last = None;
    for addr in addrs {
        match connector.ask(
            *addr,
            authority,
            tls_name,
            path,
            limits.request_timeout(),
            limits.max_body_bytes,
            cancel,
        ) {
            Ok(answer) => return Ok((answer, *addr)),
            Err(error) => last = Some(error),
        }
    }
    Err(last.unwrap_or_else(|| HttpError::Connect("no address answered".to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;

    /// A connector that answers from a script and records every call, so a
    /// test can assert what a scan did and did not touch.
    #[derive(Default)]
    struct Fake {
        open: BTreeMap<SocketAddr, BTreeMap<&'static str, (u16, &'static str, String)>>,
        connects: Mutex<Vec<SocketAddr>>,
        asked: Mutex<Vec<(SocketAddr, String)>>,
        overlapping: AtomicUsize,
        peak: AtomicUsize,
        starts: Mutex<Vec<Instant>>,
        stall: Option<Duration>,
    }

    impl Fake {
        fn serving(addr: SocketAddr, answers: &[(&'static str, u16, &'static str, String)]) -> Self {
            let mut fake = Fake::default();
            fake.add(addr, answers);
            fake
        }

        fn add(&mut self, addr: SocketAddr, answers: &[(&'static str, u16, &'static str, String)]) {
            let scripted = answers
                .iter()
                .map(|(path, status, kind, body)| (*path, (*status, *kind, body.clone())))
                .collect();
            self.open.insert(addr, scripted);
        }

        fn connected_to(&self) -> Vec<SocketAddr> {
            self.connects.lock().expect("connects").clone()
        }
    }

    impl Connector for Fake {
        fn connect_only(&self, addr: SocketAddr, _timeout: Duration) -> ConnectOutcome {
            self.connects.lock().expect("connects").push(addr);
            self.starts.lock().expect("starts").push(Instant::now());
            let now = self.overlapping.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            if let Some(stall) = self.stall {
                std::thread::sleep(stall);
            }
            self.overlapping.fetch_sub(1, Ordering::SeqCst);
            if self.open.contains_key(&addr) {
                ConnectOutcome::Open
            } else {
                ConnectOutcome::Refused
            }
        }

        fn ask(
            &self,
            addr: SocketAddr,
            _host_header: &str,
            _tls_name: Option<&str>,
            path: &str,
            _timeout: Duration,
            _max_body: usize,
            _cancel: &Cancel,
        ) -> Result<AnonymousAnswer, HttpError> {
            self.asked
                .lock()
                .expect("asked")
                .push((addr, path.to_string()));
            let scripted = self
                .open
                .get(&addr)
                .and_then(|paths| paths.get(path))
                .ok_or_else(|| HttpError::Io("nothing scripted".to_string()))?;
            Ok(AnonymousAnswer {
                status: scripted.0,
                content_type: Some(scripted.1.to_string()),
                body: scripted.2.clone().into_bytes(),
            })
        }
    }

    fn json_body(value: serde_json::Value) -> String {
        value.to_string()
    }

    fn ollama_answers(version: &str) -> Vec<(&'static str, u16, &'static str, String)> {
        vec![
            ("/v1/health", 404, "application/json", "{}".to_string()),
            (
                "/api/version",
                200,
                "application/json",
                json_body(json!({ "version": version })),
            ),
            (
                "/api/tags",
                200,
                "application/json",
                json_body(json!({"models": [{"name": "llama3.2:latest"}]})),
            ),
            ("/api/v0/models", 404, "application/json", "{}".to_string()),
        ]
    }

    fn scan_with(plan: &Plan, fake: &Fake, context: &ScanContext) -> Discovery {
        scan(
            plan,
            fake,
            &Cancel::never(),
            context,
            &NodeTransport::default(),
        )
    }

    fn loopback_context() -> ScanContext {
        ScanContext {
            reverse_dns: false,
            ..ScanContext::default()
        }
    }

    /// The whole default. With no arguments this sends nothing off the machine.
    #[test]
    fn the_default_scope_is_loopback_only() {
        let plan = plan(&Scope::default(), &NodeTransport::default()).expect("loopback plans");
        let expected_ports: Vec<u16> = NodeEngine::ALL
            .iter()
            .map(|engine| engine.default_port())
            .collect();
        let mut sorted = expected_ports.clone();
        sorted.sort_unstable();

        assert_eq!(plan.addresses, 2, "127.0.0.1 and [::1], and nothing else");
        assert_eq!(plan.ports, sorted);
        assert_eq!(plan.probes(), 2 * sorted.len());
        for target in &plan.targets {
            assert!(
                target.addr.ip().is_loopback(),
                "{} is not this machine",
                target.addr
            );
        }

        let fake = Fake::default();
        scan_with(&plan, &fake, &loopback_context());
        assert!(
            fake.connected_to().iter().all(|addr| addr.ip().is_loopback()),
            "a default scan connected off-box: {:?}",
            fake.connected_to()
        );
        assert_eq!(fake.connected_to().len(), 2 * sorted.len());
    }

    /// The ports come off the engine rows, so a fourth engine is a row and
    /// nothing else.
    #[test]
    fn every_default_port_comes_from_an_engine_row() {
        let plan = plan(&Scope::default(), &NodeTransport::default()).expect("plans");
        let mut from_rows: Vec<u16> = NodeEngine::ALL
            .iter()
            .map(|engine| engine.default_port())
            .collect();
        from_rows.sort_unstable();
        assert_eq!(plan.ports, from_rows);
    }

    #[test]
    fn no_loopback_and_no_default_ports_scan_only_what_is_named() {
        let scope = Scope {
            hosts: vec!["127.0.0.1".to_string()],
            ports: vec![9931],
            loopback: false,
            default_ports: false,
            ..Scope::default()
        };
        let plan = plan(&scope, &NodeTransport::default()).expect("plans");
        assert_eq!(plan.probes(), 1);
        assert_eq!(
            plan.targets[0].addr,
            SocketAddr::from(([127, 0, 0, 1], 9931))
        );
    }

    /// Asking for the acknowledgement at scan time means an operator never
    /// discovers a machine their own fabric would then refuse to reach.
    #[test]
    fn cleartext_to_the_lan_is_refused_without_the_acknowledgement() {
        let scope = Scope {
            ranges: vec!["100.64.0.0/24".to_string()],
            loopback: false,
            ..Scope::default()
        };
        let refusal = plan(&scope, &NodeTransport::default()).expect_err("refused");
        assert_eq!(refusal.code(), "transport_refused");
        let message = refusal.to_string();
        assert!(message.contains("--allow-cleartext-node-transport"), "{message}");
        assert!(message.contains("--node-tls-ca"), "{message}");

        // Paired: the acknowledgement is the same one a node needs, and it works.
        let acknowledged = NodeTransport::resolve(None, true).expect("acknowledged");
        assert!(plan(&scope, &acknowledged).is_ok());
    }

    /// The paired half: the zero-flag default must not be caught by the rule
    /// that protects the LAN.
    #[test]
    fn loopback_needs_no_acknowledgement() {
        plan(&Scope::default(), &NodeTransport::default())
            .expect("the default scan needs no flag at all");
    }

    /// A cap must never read as coverage.
    #[test]
    fn a_scan_accounts_for_every_planned_probe() {
        let scope = Scope {
            ranges: vec!["100.64.0.0/24".to_string()],
            loopback: false,
            default_ports: false,
            ports: vec![8181],
            ..Scope::default()
        };
        let acknowledged = NodeTransport::resolve(None, true).expect("acknowledged");
        let plan = plan(&scope, &acknowledged)
            .expect("plans")
            .within(Duration::from_millis(300));
        let fake = Fake {
            stall: Some(Duration::from_millis(40)),
            ..Fake::default()
        };
        let discovery = scan(
            &plan,
            &fake,
            &Cancel::never(),
            &loopback_context(),
            &acknowledged,
        );

        assert_eq!(discovery.planned, 254);
        assert_eq!(
            discovery.findings.len() + discovery.not_listed.total() + discovery.not_scanned,
            discovery.planned,
            "every planned probe has to be accounted for somewhere"
        );
        assert!(
            discovery.not_scanned > 0,
            "the wall clock should have stopped this scan short"
        );
    }

    #[test]
    fn the_scheduler_never_exceeds_its_concurrency_or_rate() {
        let scope = Scope {
            ranges: vec!["100.64.4.0/24".to_string()],
            loopback: false,
            default_ports: false,
            ports: vec![8181],
            ..Scope::default()
        };
        let acknowledged = NodeTransport::resolve(None, true).expect("acknowledged");
        let plan = plan(&scope, &acknowledged).expect("plans");
        let fake = Fake {
            stall: Some(Duration::from_millis(5)),
            ..Fake::default()
        };
        scan(&plan, &fake, &Cancel::never(), &loopback_context(), &acknowledged);

        assert!(
            fake.peak.load(Ordering::SeqCst) <= plan.limits.concurrency,
            "{} connects were open at once, over the limit of {}",
            fake.peak.load(Ordering::SeqCst),
            plan.limits.concurrency
        );
        // Sorted first: 32 threads append to this as they go, so the order it
        // is recorded in is not the order things happened.
        //
        // Measured as a span rather than as a count inside a sliding window.
        // The window count is hostage to a millisecond of scheduler jitter at
        // either end — it reads 201 or 202 for the same correct behaviour —
        // while the thing the rate limit exists for is not putting a burst
        // through somebody's router. Unpaced, all 254 of these start within a
        // few milliseconds, so the span is what actually separates the two.
        let mut starts = fake.starts.lock().expect("starts").clone();
        starts.sort_unstable();
        let span = starts
            .last()
            .expect("targets were probed")
            .duration_since(starts[0]);
        let paced = Duration::from_secs_f64(
            (starts.len() - 1) as f64 / plan.limits.connects_per_second as f64,
        );
        assert!(
            span >= paced.mul_f64(0.8),
            "{} connects took {span:?}, which is faster than the {paced:?} that pacing them at \
             {}/s would take",
            starts.len(),
            plan.limits.connects_per_second
        );
    }

    /// A finding's address and its evidence have to describe the same socket.
    #[test]
    fn stage_two_asks_the_address_stage_one_found() {
        let open = SocketAddr::from(([127, 0, 0, 1], 9941));
        let closed = SocketAddr::from(([127, 0, 0, 2], 9941));
        let mut fake = Fake::serving(open, &ollama_answers("0.33.2"));
        fake.add(open, &ollama_answers("0.33.2"));

        let scope = Scope {
            hosts: vec!["127.0.0.1".to_string(), "127.0.0.2".to_string()],
            ports: vec![9941],
            loopback: false,
            default_ports: false,
            ..Scope::default()
        };
        let plan = plan(&scope, &NodeTransport::default()).expect("plans");
        let discovery = scan_with(&plan, &fake, &loopback_context());

        assert_eq!(discovery.findings.len(), 1);
        assert_eq!(discovery.findings[0].address, "127.0.0.1");
        assert!(
            fake.asked
                .lock()
                .expect("asked")
                .iter()
                .all(|(addr, _)| *addr == open),
            "stage two asked an address stage one did not find open"
        );
        assert!(
            !fake.connected_to().contains(&closed) || discovery.not_listed.refused > 0,
            "a closed address must be counted, not listed"
        );
    }

    /// Nothing here may map "nothing matched on engine X's default port" to X.
    #[test]
    fn the_engine_is_never_inferred_from_the_port() {
        let addr = SocketAddr::from(([127, 0, 0, 1], 9951));
        let page = "<!doctype html><title>hello</title>".to_string();
        let fake = Fake::serving(
            addr,
            &[
                ("/v1/health", 200, "text/html", page.clone()),
                ("/api/version", 200, "text/html", page.clone()),
                ("/api/tags", 200, "text/html", page.clone()),
                ("/api/v0/models", 200, "text/html", page),
            ],
        );
        // The stub sits on what this build is told is an engine's default port.
        let scope = Scope {
            hosts: vec!["127.0.0.1".to_string()],
            loopback: false,
            ..Scope::default()
        };
        let plan = plan_with_default_ports(
            &scope,
            &NodeTransport::default(),
            &[(NodeEngine::LmStudio, 9951)],
        )
        .expect("plans");
        let discovery = scan_with(&plan, &fake, &loopback_context());

        assert_eq!(discovery.findings.len(), 1);
        assert_eq!(discovery.findings[0].classification.kind(), "other_http");
        assert!(discovery.findings[0].proposal.is_none());
        assert!(discovery.findings[0]
            .not_proposed
            .as_ref()
            .is_some_and(|why| why.contains("matched no engine")));
    }

    /// A row that is not established is a row with no way to add it.
    #[test]
    fn only_an_identified_machine_is_ever_proposed() {
        let addr = SocketAddr::from(([127, 0, 0, 1], 9961));
        let fake = Fake::serving(addr, &ollama_answers("0.33.2"));
        let scope = Scope {
            hosts: vec!["127.0.0.1".to_string()],
            ports: vec![9961],
            loopback: false,
            default_ports: false,
            ..Scope::default()
        };
        let plan = plan(&scope, &NodeTransport::default()).expect("plans");
        let discovery = scan_with(&plan, &fake, &loopback_context());

        let finding = &discovery.findings[0];
        let proposal = finding.proposal.as_ref().expect("an ollama node is proposable");
        assert_eq!(proposal.engine, NodeEngine::Ollama);
        assert_eq!(proposal.port, 9961);
        assert!(proposal.line.contains("ollama://"), "{}", proposal.line);
        assert!(
            proposal.comment_preview.contains("0.33.2"),
            "{}",
            proposal.comment_preview
        );
        assert_eq!(finding.not_proposed, None);
    }

    /// A proxy answers `/v1/health` too. Joined as a node it would place work
    /// on every machine behind it a second time, through itself.
    #[test]
    fn a_fabric_proxy_is_never_proposed() {
        let addr = SocketAddr::from(([127, 0, 0, 1], 9971));
        let fake = Fake::serving(
            addr,
            &[(
                "/v1/health",
                200,
                "application/json",
                json_body(json!({"ok": true, "service": "camelid-fabric", "ready": true})),
            )],
        );
        let scope = Scope {
            hosts: vec!["127.0.0.1".to_string()],
            ports: vec![9971],
            loopback: false,
            default_ports: false,
            ..Scope::default()
        };
        let plan = plan(&scope, &NodeTransport::default()).expect("plans");
        let discovery = scan_with(&plan, &fake, &loopback_context());

        assert_eq!(discovery.findings[0].classification.kind(), "fabric_proxy");
        assert!(discovery.findings[0].proposal.is_none());
        assert!(discovery.findings[0]
            .not_proposed
            .as_ref()
            .is_some_and(|why| why.contains("not a node")));
    }

    /// The label follows the host somebody will actually use, and never
    /// collides with one already in the file.
    #[test]
    fn a_proposed_label_never_collides_with_the_file() {
        let taken = vec!["local-ollama".to_string(), "local-ollama-2".to_string()];
        assert_eq!(
            label_for("127.0.0.1", NodeEngine::Ollama, &taken),
            "local-ollama-3"
        );
        assert_eq!(
            label_for("100.64.0.37", NodeEngine::Ollama, &[]),
            "host-100-64-0-37-ollama"
        );
        assert_eq!(label_for("mini2.lan", NodeEngine::Camelid, &[]), "mini2-camelid");
        assert!(node::is_writable_label(&label_for(
            "[fd00::1]",
            NodeEngine::Camelid,
            &[]
        )));
    }

    /// The grammar is the one thing between a stranger's version string and a
    /// file that is parsed line by line.
    #[test]
    fn a_hostile_version_string_is_never_written() {
        let comment = provenance_comment(
            "2026-09-12T10:04:11Z",
            "100.64.0.37:11434",
            NodeEngine::Ollama,
            Some("0.1\nx=camelid://169.254.0.9:8181"),
        );
        assert!(comment.contains(VERSION_NOT_RECORDED), "{comment}");
        assert!(!comment.contains("169.254.0.9"), "{comment}");
        assert!(!comment.contains('\n'), "{comment:?}");

        let carriage = provenance_comment(
            "2026-09-12T10:04:11Z",
            "100.64.0.37:11434",
            NodeEngine::Ollama,
            Some("0.1\r\nx=camelid://169.254.0.9:8181"),
        );
        assert!(carriage.contains(VERSION_NOT_RECORDED), "{carriage}");

        // A version that is what a version actually looks like is recorded.
        let ordinary = provenance_comment(
            "2026-09-12T10:04:11Z",
            "100.64.0.37:11434",
            NodeEngine::Ollama,
            Some("0.33.2"),
        );
        assert!(ordinary.ends_with("answered like ollama 0.33.2"), "{ordinary}");
    }

    #[test]
    fn third_party_strings_are_sanitized_before_printing() {
        let hostile = "mini2\u{1b}[2J\u{7}.lan";
        let escaped = display_safe(hostile);
        assert!(!escaped.contains('\u{1b}'), "{escaped}");
        assert!(!escaped.contains('\u{7}'), "{escaped}");
        assert!(escaped.contains("\\u{1b}"), "{escaped}");
        // Ordinary text is borrowed rather than rewritten.
        assert!(matches!(display_safe("mini2.lan"), Cow::Borrowed(_)));
    }

    #[test]
    fn a_range_that_is_too_large_or_public_never_reaches_a_socket() {
        let acknowledged = NodeTransport::resolve(None, true).expect("acknowledged");
        for refused in ["8.8.8.0/24", "100.64.0.0/16"] {
            let scope = Scope {
                ranges: vec![refused.to_string()],
                loopback: false,
                ..Scope::default()
            };
            assert!(
                plan(&scope, &acknowledged).is_err(),
                "{refused} was planned"
            );
        }
    }

    #[test]
    fn a_scan_that_would_cover_nothing_says_so() {
        let scope = Scope {
            loopback: false,
            default_ports: false,
            ..Scope::default()
        };
        assert_eq!(
            plan(&scope, &NodeTransport::default()).expect_err("nothing to scan"),
            ScopeRefusal::Nothing
        );
    }

    #[test]
    fn too_many_ports_are_refused_rather_than_trimmed() {
        let scope = Scope {
            ports: (9000..9010).collect(),
            loopback: false,
            default_ports: false,
            hosts: vec!["127.0.0.1".to_string()],
            ..Scope::default()
        };
        let refusal = plan(&scope, &NodeTransport::default()).expect_err("refused");
        assert_eq!(refusal.code(), "scope_too_large");
    }
}
