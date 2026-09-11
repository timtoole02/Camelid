//! A fabric of independent Camelid nodes.
//!
//! # Why this exists
//!
//! Camelid already has a distributed lane (`crate::distributed`) that shards one
//! model's layers across two machines. It is a *capacity* mechanism: it lets a
//! model run that fits on neither machine, and it is measurably slower than a
//! single node, because every generated token crosses the network twice and each
//! machine idles while the other computes.
//!
//! The fabric is the opposite arrangement. Each node owns a whole model and a
//! whole session; the fabric places *whole requests*. Nothing crosses the network
//! inside the token loop, so throughput scales with nodes instead of being taxed
//! by them. It composes with the layer-sharded lane rather than replacing it —
//! a fabric node may itself be a distributed coordinator.
//!
//! # Shape
//!
//! * [`node`] — identity and observed state.
//! * [`probe`] — turning `/v1/health` into a routing fact.
//! * [`policy`] — pure placement decisions; the correctness of the fabric.
//! * [`forward`] — sending a placed request to the node that will serve it.
//! * [`cancel`] — telling that send it is no longer wanted.
//! * `transport` — authenticating or explicitly constraining the node hop.

pub mod aliases;
pub(crate) mod camelid;
pub mod cancel;
pub mod capability;
pub(crate) mod client_keys;
pub mod divergence;
pub mod engine;
pub mod forward;
pub(crate) mod http;
mod lmstudio;
pub mod node;
pub(crate) mod nodes;
mod ollama;
pub mod policy;
pub mod probe;
pub mod sample;
pub mod server;
pub mod textdiff;
mod transport;
pub(crate) mod watch;

use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

pub use aliases::{
    parse_model_alias, parse_model_aliases, AliasParseError, ModelAlias, ModelAliases,
};
pub use cancel::Cancel;
pub use capability::{BlockerDetail, Capabilities, Capability, Provenance, RequirementLimit};
pub use divergence::{
    check_temperature, conclude, sha256_hex, shared_weights_digest, AppliedSampling, Comparison,
    Honoured, ModelIdentity, RenderedPrompt, Sample, SamplingPlan, Side, Stability,
    TemplateEvidence, TextMatch, Uncontrolled, Verdict, WeightsCheck, WeightsDigest,
    HISTORY_PERTURBATION, HISTORY_PERTURBATION_MAX_TOKENS, MAX_COMPARE_REPETITIONS,
    MAX_COMPARE_TEMPERATURE, REQUEST_HISTORY,
};
pub use engine::NodeEngine;
pub use forward::{
    wants_streaming, ForwardError, Forwarded, NodeAnswer, StreamOutcome, Streaming,
    DEFAULT_FORWARD_TIMEOUT, ENGINE_QUEUE_FULL_CODE,
};
pub use node::{
    parse_fabric, parse_node_spec, NodeLoad, NodeReady, NodeSnapshot, NodeSpec, NodeSpecParseError,
    NodeStatus, DEFAULT_NODE_PORT,
};
pub use nodes::ForeignAddition;
use nodes::NodeSet;
use policy::{load_of, route_reserved_with_estimates, ServiceTimeEstimates};
pub use policy::{
    route, route_reserved, MixedEngines, Requirements, Reservations, Residency, RouteDecision,
    RouteError, RouteMode, RouteReason, RouteRequest, COLD_LOAD_COST, MIXED_ENGINES_FLAG,
    UNREPORTED_LOAD_COST,
};
pub use probe::{probe_fabric, probe_node, Observation, ProbeError, DEFAULT_PROBE_TIMEOUT};
pub use sample::SampleError;
pub use textdiff::{diff_lines, Diff, DiffLine, Eol, Op};
use transport::NodeTransport;

/// Why a comparison could not be set up or completed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompareError {
    NoSuchNode {
        label: String,
        known: Vec<String>,
    },
    /// Comparing a node with itself measures nothing about the engines.
    SameNode {
        label: String,
    },
    Side(SampleError),
}

impl std::fmt::Display for CompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchNode { label, known } => write!(
                f,
                "no node named {label} in this fabric; it has {}",
                if known.is_empty() {
                    "none".to_string()
                } else {
                    known.join(", ")
                }
            ),
            Self::SameNode { label } => write!(
                f,
                "{label} cannot be compared with itself; name two different nodes"
            ),
            Self::Side(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CompareError {}

/// How many nodes one request may be sent to before it fails.
///
/// Two means one retry. A request that finds its node gone is still served, and
/// a fabric where several nodes have gone still fails after a bounded number of
/// dials rather than walking every node it has. Raise it for a large fabric
/// where more of that walk is worth paying for; one turns failover off.
pub const DEFAULT_MAX_FORWARD_ATTEMPTS: usize = 2;

/// A configured set of nodes.
#[derive(Clone)]
pub struct Fabric {
    /// The machines this fabric places on.
    ///
    /// Shared across clones for the same reason as `reserved`, and one reason
    /// more: two clones that disagreed about which machines exist would place
    /// against different fabrics.
    nodes: NodeSet,
    timeout: Duration,
    /// Budget for each generation a comparison runs; see
    /// [`Fabric::with_generation_timeout`].
    generation_timeout: Duration,
    bearer: Option<String>,
    transport: NodeTransport,
    /// What each node calls a model, where an operator has said so.
    ///
    /// Empty for every fabric that never declared one, and the resolution
    /// falls through to the id as given, so this changes nothing until it is
    /// used.
    aliases: Arc<ModelAliases>,
    /// Requests this fabric has placed and not yet finished.
    ///
    /// Shared across clones on purpose: the resident proxy hands a `Fabric` to
    /// every request, and they have to be counting into the same place or they
    /// cannot see each other.
    reserved: Arc<Mutex<Reservations>>,
    /// Successful service times learned by this resident fabric.
    ///
    /// Shared across clones because each proxy request receives a clone and
    /// all of them must learn one policy. The estimates are consulted only by
    /// the opt-in completion-time mode.
    service_times: Arc<Mutex<ServiceTimeEstimates>>,
    /// The most recent observation, with the node-set generation it was taken
    /// over, reused while it is fresh enough *and* still describes the set.
    ///
    /// Shared across clones for the same reason as `reserved`: an observation
    /// only one clone can see would be re-taken by every other one.
    observed: Arc<Mutex<Option<(u64, Observation)>>>,
    /// How stale a reused observation may be. Zero means never reuse one.
    max_observation_age: Duration,
    /// How many nodes one request may be sent to. One never fails over.
    max_forward_attempts: usize,
}

/// Hand-written so a token can never reach a log through a derived `Debug`,
/// the way `ServerPolicyOptions` redacts the key it holds.
impl std::fmt::Debug for Fabric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fabric")
            .field("nodes", &self.nodes)
            .field("timeout", &self.timeout)
            .field("generation_timeout", &self.generation_timeout)
            .field("bearer", &self.bearer.as_ref().map(|_| "[REDACTED]"))
            .field("transport", &self.transport)
            .field("max_observation_age", &self.max_observation_age)
            .field("max_forward_attempts", &self.max_forward_attempts)
            .finish()
    }
}

impl Fabric {
    pub fn new(specs: Vec<NodeSpec>) -> Self {
        Self::over(NodeSet::fixed(specs))
    }

    /// Place on the machines named by a file, re-read as it changes.
    ///
    /// Fails rather than starting on an empty fabric: a proxy that silently
    /// ignored an unreadable node file would refuse every request while
    /// looking as though it had started correctly.
    pub fn from_node_file(path: std::path::PathBuf) -> std::io::Result<Self> {
        let aliases = nodes::load_model_aliases(&path)?;
        Ok(Self::over(NodeSet::from_file(path)?).with_model_aliases(aliases))
    }

    /// [`Self::from_node_file`] with the staleness bound supplied, so a test
    /// does not have to wait one out.
    #[cfg(test)]
    fn from_node_file_every(path: std::path::PathBuf, interval: Duration) -> std::io::Result<Self> {
        Ok(Self::over(NodeSet::from_file_every(path, interval)?))
    }

    fn over(nodes: NodeSet) -> Self {
        Self {
            nodes,
            timeout: DEFAULT_PROBE_TIMEOUT,
            generation_timeout: DEFAULT_FORWARD_TIMEOUT,
            bearer: None,
            transport: NodeTransport::default(),
            aliases: Arc::new(ModelAliases::default()),
            reserved: Arc::new(Mutex::new(Reservations::none())),
            service_times: Arc::new(Mutex::new(ServiceTimeEstimates::default())),
            observed: Arc::new(Mutex::new(None)),
            max_observation_age: Duration::ZERO,
            max_forward_attempts: DEFAULT_MAX_FORWARD_ATTEMPTS,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Bound each generation [`Fabric::compare`] runs.
    ///
    /// Apart from [`Fabric::with_timeout`] because the two budget different
    /// things: a probe is a health read and should fail fast, while a
    /// generation can legitimately take minutes. Defaults to
    /// [`DEFAULT_FORWARD_TIMEOUT`].
    pub fn with_generation_timeout(mut self, timeout: Duration) -> Self {
        self.generation_timeout = timeout;
        self
    }

    /// Adopt an operator's declarations about what each node calls a model.
    pub fn with_model_aliases(mut self, aliases: ModelAliases) -> Self {
        self.aliases = Arc::new(aliases);
        self
    }

    /// What `label` calls `model`, which is `model` itself unless somebody said
    /// otherwise.
    pub fn local_model_id<'a>(&'a self, label: &str, model: &'a str) -> &'a str {
        self.aliases.resolve(label, model)
    }

    /// The alias lines in force, for the startup line. Read once when the
    /// fabric was built and never again, which that line has to say.
    pub fn aliases_in_force(&self) -> Vec<ModelAlias> {
        self.aliases.declared()
    }

    /// Where a request would be placed, given what the nodes say now.
    ///
    /// The one placement rule, with this fabric's aliases applied — what
    /// `fabric route`, `GET /v1/models/{id}` and dispatch all go through, so
    /// the dry run, the listing and the send cannot drift apart.
    pub fn decide(
        &self,
        snapshots: &[NodeSnapshot],
        request: &RouteRequest<'_>,
    ) -> Result<RouteDecision, RouteError> {
        route(snapshots, &request.with_aliases(&self.aliases))
    }

    /// Every model id a request could name and be placed, in `mixed` mode.
    ///
    /// Exactly what [`Self::decide`] would accept: a canonical id an alias
    /// resolves to something a placeable node holds, and every id a placeable
    /// node holds directly. An entry that exists only because of an alias says
    /// so, because what it rests on is somebody's word.
    pub fn servable_models(
        &self,
        snapshots: &[NodeSnapshot],
        mixed: MixedEngines,
    ) -> Vec<ServableModel> {
        let placeable: Vec<&NodeSnapshot> = snapshots
            .iter()
            .filter(|snapshot| snapshot.is_placeable_under(mixed))
            .collect();
        let mut listed: std::collections::BTreeMap<String, ModelIdentity> =
            std::collections::BTreeMap::new();
        for snapshot in &placeable {
            let ready = snapshot.status.ready().expect("placeable nodes are ready");
            let held = ready
                .active_model_id
                .iter()
                .chain(ready.resident_models.iter().flatten())
                .chain(ready.models.iter());
            for id in held {
                if policy::serves(snapshot, id).is_some() {
                    listed.entry(id.clone()).or_insert(ModelIdentity::SameId);
                }
            }
        }
        for alias in self.aliases.declared() {
            let held = placeable.iter().any(|snapshot| {
                snapshot.label() == alias.label && policy::serves(snapshot, &alias.local).is_some()
            });
            if held && alias.local != alias.canonical {
                // A claim that travels: an id also held directly can still be
                // answered by the aliased node, so the entry says it may rest
                // on the declaration.
                listed.insert(alias.canonical, ModelIdentity::AssertedByOperator);
            }
        }
        listed
            .into_iter()
            .map(|(id, identity)| ServableModel { id, identity })
            .collect()
    }

    /// How many nodes one request may be sent to; see
    /// [`Self::with_max_forward_attempts`].
    pub fn max_forward_attempts(&self) -> usize {
        self.max_forward_attempts
    }

    /// Announce each foreign node added to the node file from now on. The
    /// proxy turns this on when it places on other engines, because then such
    /// a node is placed on the moment it is added.
    pub fn announce_foreign_additions(&self, on: bool) {
        self.nodes.announce_foreign_additions(on);
    }

    /// The foreign nodes announced since this fabric was built.
    pub fn foreign_added_since_start(&self) -> Vec<ForeignAddition> {
        self.nodes.foreign_added_since_start()
    }

    /// Reuse an observation for up to `max_age` instead of probing again.
    ///
    /// A process that makes one request and exits wants the default of zero:
    /// it has nothing to reuse, and paying for the freshest possible view is
    /// free. A resident proxy is the opposite — without a bound it probes every
    /// node on every request. See [`Observation`].
    pub fn with_max_observation_age(mut self, max_age: Duration) -> Self {
        self.max_observation_age = max_age;
        self
    }

    /// How many nodes one request may be sent to before it fails.
    ///
    /// See [`DEFAULT_MAX_FORWARD_ATTEMPTS`]. One disables failover entirely and
    /// is the exact behaviour of a fabric that has never heard of it. Zero is
    /// meaningless — a request must be sent somewhere — and is read as one.
    pub fn with_max_forward_attempts(mut self, attempts: usize) -> Self {
        self.max_forward_attempts = attempts.max(1);
        self
    }

    /// Authenticate to every node with this bearer token.
    ///
    /// Required against a node started with an API key: `/v1/health` is exempt
    /// from the server's auth, so without a token such a node observes as ready
    /// and places fine, then answers the request itself with 401.
    pub fn with_bearer(mut self, bearer: Option<&str>) -> Self {
        self.bearer = bearer.map(str::to_string);
        self
    }

    /// Configure how every probe and forwarded request reaches a node.
    ///
    /// A CA bundle enables server-authenticated TLS for every node. Without
    /// one, cleartext is restricted to loopback unless the operator explicitly
    /// acknowledges direct cleartext node transport. The two modes cannot be
    /// combined.
    pub fn with_node_transport(
        mut self,
        ca_file: Option<&std::path::Path>,
        allow_cleartext_remote: bool,
    ) -> std::io::Result<Self> {
        self.transport = NodeTransport::resolve(ca_file, allow_cleartext_remote)?;
        Ok(self)
    }

    /// The machines this fabric places on, as they stand right now.
    pub fn specs(&self) -> Vec<NodeSpec> {
        self.nodes.current().0.to_vec()
    }

    /// Whether the node set changes without a restart.
    pub fn is_reloadable(&self) -> bool {
        self.nodes.is_reloadable()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.current().0.is_empty()
    }

    /// A secret-free description suitable for startup logs.
    pub fn node_transport_description(&self) -> &'static str {
        self.transport.description()
    }

    /// Run one prompt on two of this fabric's nodes and report what differed.
    ///
    /// This is measurement, not placement: the two nodes are named, not chosen,
    /// and nothing here fails over. A comparison that quietly moved to a third
    /// node would be reporting about a machine the operator never asked about.
    /// Foreign engines are therefore *comparable* even though they are not
    /// placeable — reading a node was never the thing we refused.
    ///
    /// `left_model` and `right_model` may differ, which is an operator saying
    /// two differently-named ids are the same weights. Engines do not agree on
    /// naming, so that claim is recorded as unverified rather than inferred —
    /// unless both sides publish the same file digest, which verifies it, or
    /// one engine kind publishes two that differ, or a Camelid node refuses
    /// the other's, which shows different weights. Two engines publishing
    /// different file digests settle nothing, and the claim stands as made.
    ///
    /// Each node is probed within [`Fabric::with_timeout`] and each generation
    /// within [`Fabric::with_generation_timeout`]. `plan` is bounded to what
    /// will actually run before anything is sent, so the plan the comparison
    /// records is the one that ran.
    pub fn compare(
        &self,
        left: &str,
        right: &str,
        left_model: &str,
        right_model: &str,
        prompt: &str,
        plan: SamplingPlan,
    ) -> Result<Comparison, CompareError> {
        if left == right {
            return Err(CompareError::SameNode {
                label: left.to_string(),
            });
        }
        let specs = self.specs();
        let find = |label: &str| {
            specs
                .iter()
                .find(|spec| spec.label == label)
                .cloned()
                .ok_or_else(|| CompareError::NoSuchNode {
                    label: label.to_string(),
                    known: specs.iter().map(|spec| spec.label.clone()).collect(),
                })
        };
        let left_spec = find(left)?;
        let right_spec = find(right)?;

        let plan = plan.bounded();
        let bearer = self.bearer.as_deref();
        let left_request = sample::SideRequest {
            spec: &left_spec,
            model: left_model,
            prompt,
            plan: &plan,
            bearer,
            probe_timeout: self.timeout,
            generation_timeout: self.generation_timeout,
        };
        let right_request = sample::SideRequest {
            spec: &right_spec,
            model: right_model,
            ..left_request
        };

        // Both sides are read before either generates: a Camelid side's runs
        // are bound to the other side's loaded-file digest, where the other
        // side is a Camelid node too, so the engine checks its own loaded bytes
        // instead of this fabric trusting a listing.
        let left_prepared =
            sample::prepare(&left_request, &self.transport).map_err(CompareError::Side)?;
        let right_prepared =
            sample::prepare(&right_request, &self.transport).map_err(CompareError::Side)?;
        let left_bind = right_prepared.loaded_file_digest().map(str::to_string);
        let right_bind = left_prepared.loaded_file_digest().map(str::to_string);
        let left_side = sample::measure(
            &left_request,
            left_prepared,
            left_bind.as_deref(),
            &self.transport,
        )
        .map_err(CompareError::Side)?;
        let right_side = sample::measure(
            &right_request,
            right_prepared,
            right_bind.as_deref(),
            &self.transport,
        )
        .map_err(CompareError::Side)?;
        let identity = if left_model == right_model {
            divergence::ModelIdentity::SameId
        } else {
            divergence::ModelIdentity::AssertedByOperator
        };
        Ok(divergence::conclude(
            prompt, plan, left_side, right_side, identity,
        ))
    }

    /// [`Self::compare`] with each side's model id resolved through the alias
    /// table, so an operator who declared the names once does not retype them.
    pub fn compare_model(
        &self,
        left: &str,
        right: &str,
        model: &str,
        prompt: &str,
        plan: SamplingPlan,
    ) -> Result<Comparison, CompareError> {
        let left_model = self.aliases.resolve(left, model).to_string();
        let right_model = self.aliases.resolve(right, model).to_string();
        self.compare(left, right, &left_model, &right_model, prompt, plan)
    }

    /// Observe every node.
    ///
    /// Probes unless a previous observation is still inside
    /// [`Fabric::with_max_observation_age`], which defaults to zero — so this
    /// probes every time until a caller asks for something else.
    pub fn observe(&self) -> Vec<NodeSnapshot> {
        self.observe_reporting_reuse().0
    }

    /// Observe, and say whether the answer came from a previous observation.
    ///
    /// Placement needs that second fact: a refusal decided from a reused
    /// observation is a statement about the fabric as it was, and some
    /// refusals are reported to clients as permanent.
    fn observe_reporting_reuse(&self) -> (Vec<NodeSnapshot>, bool) {
        // Resolved first, and its lock released before the observation lock is
        // taken: the order is nodes -> observed -> reserved, and no two are
        // ever held together.
        let (specs, generation) = self.nodes.current();

        if self.max_observation_age.is_zero() {
            return (self.probe(&specs), false);
        }

        // Refreshing under the lock makes concurrent callers wait for one probe
        // rather than each starting their own. Without that, a burst arriving
        // on an expired observation would reproduce exactly the per-request
        // probing this bound exists to remove.
        let mut observed = lock(&self.observed);
        if let Some((taken_over, observation)) = observed.as_ref() {
            // The generation is checked before the clock. An observation of a
            // set that has since gained or lost a machine is not stale, it is
            // about something else: reusing it would place on a node that has
            // gone, or refuse to place on one that has just arrived, for the
            // whole of the freshness window.
            if *taken_over == generation
                && observation.is_fresh_at(Instant::now(), self.max_observation_age)
            {
                return (observation.snapshots().to_vec(), true);
            }
        }
        let snapshots = self.probe(&specs);
        *observed = Some((
            generation,
            Observation::taken_at(snapshots.clone(), Instant::now()),
        ));
        (snapshots, false)
    }

    /// Probe every node in `specs`, whatever was observed before.
    fn probe(&self, specs: &[NodeSpec]) -> Vec<NodeSnapshot> {
        probe::probe_fabric_with_transport(
            specs,
            self.bearer.as_deref(),
            self.timeout,
            &self.transport,
        )
    }

    /// Drop the current observation, so the next one is taken fresh.
    fn forget_observation(&self) {
        *lock(&self.observed) = None;
    }

    /// A node that did not answer may be gone, and an observation that has
    /// already proved wrong must not be reused for the rest of its window.
    ///
    /// This is what keeps the freshness bound honest: a node dying inside the
    /// window costs the one request that discovers it, not every request until
    /// the observation expires.
    ///
    /// A cancelled request is deliberately not one of those: it says nothing
    /// about the node it was placed on, and dropping the observation for it
    /// would make every client that hangs up cost the *next* request a full
    /// probe round.
    fn forget_observation_if_node_vanished(&self, error: &ForwardError) {
        if matches!(
            error,
            ForwardError::Transport { .. } | ForwardError::Unreachable { .. }
        ) {
            self.forget_observation();
        }
    }

    /// Observe the fabric and choose a node for a request.
    ///
    /// The returned [`Placement`] counts the request against the chosen node
    /// until it is dropped, so a placement running concurrently can see it.
    pub fn place(&self, request: &RouteRequest<'_>) -> Result<Placement, RouteError> {
        self.place_excluding(request, &[])
    }

    /// Choose a node for a request, ignoring nodes already found gone.
    ///
    /// `excluded` holds labels this request has already been sent to and could
    /// not reach. They are dropped from the observation rather than from the
    /// policy, so placement stays the one pure decision it was.
    fn place_excluding(
        &self,
        request: &RouteRequest<'_>,
        excluded: &[String],
    ) -> Result<Placement, RouteError> {
        // Observing is socket I/O against every node. It stays outside the
        // reservation lock: holding that across it would serialise placement on
        // the slowest node, and would deadlock a `Placement` being dropped.
        let (snapshots, reused) = self.observe_reporting_reuse();

        match self.place_observed(snapshots, request, excluded) {
            // Refusing on a reused observation would settle from memory a
            // question the caller asked about now — and the proxy reports
            // `ModelUnavailable` as a 404 a client is told never to retry. A
            // node that has loaded a model since must not be refused that way,
            // so look again before saying no. This costs a probe only on a
            // request that was about to fail.
            //
            // Not once anything is excluded: by then the caller already holds a
            // forwarding failure, and that failure — not this refusal — is what
            // it will report, so a fresh look would buy nothing and would pay a
            // whole probe round against a node just found to be gone.
            Err(_) if reused && excluded.is_empty() => {
                self.forget_observation();
                let (fresh, _) = self.observe_reporting_reuse();
                self.place_observed(fresh, request, excluded)
            }
            settled => settled,
        }
    }

    /// Choose a node from an observation already taken, and reserve it.
    fn place_observed(
        &self,
        snapshots: Vec<NodeSnapshot>,
        request: &RouteRequest<'_>,
        excluded: &[String],
    ) -> Result<Placement, RouteError> {
        let snapshots: Vec<NodeSnapshot> = snapshots
            .into_iter()
            .filter(|snapshot| !excluded.iter().any(|label| label == snapshot.label()))
            .collect();
        let request = &request.with_aliases(&self.aliases);

        let (decision, service_model, service_ahead) = if request.mode == RouteMode::CompletionTime
        {
            // Deciding, reserving and recording cold-selection recency are one
            // step. Split them and concurrent placements can all decide from
            // the same policy state. These are the only nested fabric locks,
            // always in service-times -> reservations order; completion
            // recording takes only service-times and Placement::drop takes
            // only reservations.
            let mut service_times = lock(&self.service_times);
            let mut reserved = lock(&self.reserved);
            let decision =
                route_reserved_with_estimates(&snapshots, request, &reserved, &service_times)?;
            let selected = snapshots
                .iter()
                .find(|snapshot| snapshot.label() == decision.label)
                .expect("placement returns a label it was given");
            let service_ahead =
                selected
                    .status
                    .ready()
                    .map_or(0, |ready| match ready.in_flight() {
                        Some(observed) => load_of(observed, reserved.get(&decision.label)),
                        None => reserved.get(&decision.label),
                    });
            reserved.take(&decision.label);
            let service_model = request
                .model
                .or_else(|| selected.active_model_id())
                .map(str::to_string);
            if let (Some(model), Some(route)) = (&service_model, request.service_class) {
                service_times.selected(selected, model, route);
            }
            drop(reserved);
            drop(service_times);
            (decision, service_model, service_ahead)
        } else {
            // Exactly the pre-existing throughput/affinity path: those modes
            // neither allocate nor lock service-time state during placement.
            let mut reserved = lock(&self.reserved);
            let decision = route_reserved(&snapshots, request, &reserved)?;
            reserved.take(&decision.label);
            drop(reserved);
            let selected = snapshots
                .iter()
                .find(|snapshot| snapshot.label() == decision.label)
                .expect("placement returns a label it was given");
            let service_model = request
                .model
                .or_else(|| selected.active_model_id())
                .map(str::to_string);
            (decision, service_model, 0)
        };

        let node = snapshots
            .into_iter()
            .find(|snapshot| snapshot.label() == decision.label)
            .expect("placement returns a label it was given");
        Ok(Placement {
            decision,
            node,
            reserved: Arc::clone(&self.reserved),
            service_times: Arc::clone(&self.service_times),
            service_model,
            service_class: request.service_class.map(str::to_string),
            service_mode: request.mode,
            service_ahead,
            service_recorded: false,
        })
    }

    /// What this fabric currently believes it has outstanding, for tests and
    /// for reporting. A copy, so reading it holds the lock no longer than that.
    pub fn reserved(&self) -> Reservations {
        lock(&self.reserved).clone()
    }

    /// Send a request to a node already chosen by this fabric.
    ///
    /// This exists for the one-shot `fabric run` path, whose request body uses
    /// the chosen node's active model. Keeping the send here ensures it cannot
    /// bypass the fabric's bearer or transport policy — or what the body
    /// requires: a placement made for one request is not a pass for another,
    /// so a body carrying tools is refused for a node not measured for them
    /// however it was placed.
    pub fn forward_to(
        &self,
        placement: &Placement,
        path: &str,
        body: &Value,
        timeout: Duration,
        cancel: &Cancel,
    ) -> Result<Forwarded, ForwardError> {
        let node = placement.node();
        if let Err(missing) =
            policy::meets(&node.capabilities(), placement_requirements(path, body))
        {
            return Err(ForwardError::Unsupported(format!(
                "node `{}` ({}) is not sent requests needing `{missing}`: nothing here shows it \
                 can handle them",
                node.label(),
                node.engine_and_version()
            )));
        }
        forward::forward_with_transport(
            &node.spec,
            path,
            &body_for(placement, body),
            self.bearer.as_deref(),
            timeout,
            cancel,
            &self.transport,
        )
    }

    /// Observe, place, and send — the whole path a caller actually wants.
    ///
    /// Returns the placement alongside the answer so a caller can record which
    /// node served the request and whether affinity held. A node that turns out
    /// to be gone does not fail the request while another can serve it; see
    /// [`DEFAULT_MAX_FORWARD_ATTEMPTS`].
    pub fn dispatch(
        &self,
        path: &str,
        body: &Value,
        request: &RouteRequest<'_>,
        forward_timeout: Duration,
        cancel: &Cancel,
    ) -> Result<Dispatched, DispatchError> {
        // Refuse an unsupported request before spending any probes on it.
        forward::reject_streaming(body)?;

        // What the body requires is read here, at the one place every placed
        // request passes, rather than trusted from whoever built `request`: a
        // caller may add a requirement, never take one the body carries away.
        let request = request.requiring(request.requires.union(placement_requirements(path, body)));
        let service_class =
            (request.mode != RouteMode::Throughput).then(|| service_class(path, body));
        let request = request.with_service_class(service_class.as_deref());
        let mut sent = self.send_until_a_node_takes_it(&request, |placement| {
            forward::forward_with_transport(
                &placement.node.spec,
                path,
                &body_for(placement, body),
                self.bearer.as_deref(),
                forward_timeout,
                cancel,
                &self.transport,
            )
        })?;
        if sent.value.is_success() {
            sent.placement.record_success(sent.value.elapsed);
        } else if sent.value.status >= 500 && !trusted_backpressure(&sent.placement, &sent.value) {
            sent.placement.invalidate_service_time();
        }
        Ok(Dispatched {
            decision: sent.placement.decision.clone(),
            answer: sent.value,
            attempts: sent.attempts,
        })
    }

    /// Observe, place, and start a streaming request.
    ///
    /// The same placement as [`Fabric::dispatch`] — both go through
    /// [`Fabric::place`] — but the answer is read as it arrives instead of all
    /// at once. Returns once the node's response head is in, so a caller knows
    /// the status and which node it came from before relaying a single byte.
    ///
    /// The [`Placement`] comes back rather than being dropped here because the
    /// request outlives this call: the node is busy until the last event is
    /// read, so whoever pumps the stream must hold it for that long.
    pub fn dispatch_streaming(
        &self,
        path: &str,
        body: &Value,
        request: &RouteRequest<'_>,
        head_timeout: Duration,
        idle_timeout: Duration,
        cancel: &Cancel,
    ) -> Result<DispatchedStream, DispatchError> {
        // The same chokepoint as `dispatch`: see there.
        let request = request.requiring(request.requires.union(placement_requirements(path, body)));
        let service_class =
            (request.mode != RouteMode::Throughput).then(|| service_class(path, body));
        let request = request.with_service_class(service_class.as_deref());
        let mut sent = self.send_until_a_node_takes_it(&request, |placement| {
            forward::forward_streaming_with_transport(
                &placement.node.spec,
                path,
                &body_for(placement, body),
                self.bearer.as_deref(),
                head_timeout,
                idle_timeout,
                cancel,
                &self.transport,
            )
        })?;
        if let StreamOutcome::Buffered(answer) = &sent.value {
            if answer.is_success() {
                sent.placement.record_success(answer.elapsed);
            } else if answer.status >= 500 && !trusted_backpressure(&sent.placement, answer) {
                sent.placement.invalidate_service_time();
            }
        }
        Ok(DispatchedStream {
            outcome: sent.value,
            placement: sent.placement,
            attempts: sent.attempts,
        })
    }

    /// Place the request and send it, moving on while a node turns out to be
    /// gone or too busy to take it.
    ///
    /// `send` must not have told anyone anything on the strength of an attempt
    /// it reports as failed: this calls it again against another node, which is
    /// only safe while nothing has been said and — per
    /// [`ForwardError::node_never_received_it`] — the failed node cannot have
    /// started the work.
    ///
    /// A node that answers with a queue-full refusal gets the same treatment
    /// for the same reason ([`NodeAnswer::refused_for_backpressure`]): it
    /// rejected the request at its queue boundary rather than running it, so
    /// another node can be asked. Only from an engine whose capabilities
    /// declare that refusal, though: a label says which engine to expect, not
    /// which one answered, and a gateway or impostor reusing the code would
    /// otherwise get a request that may already be running sent a second time.
    /// Any other answer — an untyped 5xx, a foreign node's anything — is
    /// relayed once, because nothing distinguishes it from a real failure. The two are tracked apart because only one of
    /// them says the observation was wrong — a saturated node is exactly where
    /// the observation said it was, and re-probing on every refusal would spend
    /// a probe per request under load, which is the cost
    /// [`Fabric::with_max_observation_age`] exists to remove.
    ///
    /// If nowhere else can take it, the node's own refusal is returned rather
    /// than an error this proxy invented: the client gets the status and body
    /// the node sent, its code and message included. Not its headers, though —
    /// [`Forwarded`] carries none, so the node's `Retry-After` does not survive
    /// the hop.
    ///
    /// A cancelled attempt ends the whole dispatch on the spot. It is not a
    /// node failure, so it is neither re-placed nor recorded against the node:
    /// there is no longer anybody to hand a second node's answer to.
    fn send_until_a_node_takes_it<T: NodeAnswer>(
        &self,
        request: &RouteRequest<'_>,
        mut send: impl FnMut(&Placement) -> Result<T, ForwardError>,
    ) -> Result<Sent<T>, DispatchError> {
        let mut gone: Vec<String> = Vec::new();
        let mut saturated: Vec<String> = Vec::new();
        // With the engine of the node that failed, taken from the placement
        // rather than looked up later: the node file can be re-declared while
        // a request is out, and the answer must name what it was sent to.
        let mut first_failure: Option<(ForwardError, NodeEngine)> = None;
        let mut refused: Option<(T, Placement)> = None;
        // Nodes actually asked, which is not the attempt number: placement can
        // run out before an attempt reaches one.
        let mut asked = 0;

        for attempt in 1..=self.max_forward_attempts.max(1) {
            let excluded: Vec<String> = gone.iter().chain(saturated.iter()).cloned().collect();
            let mut placement = match self.place_excluding(request, &excluded) {
                Ok(placement) => placement,
                Err(refusal) => {
                    if let Some((value, placement)) = refused {
                        return Ok(self.settle_for_the_refusal(value, placement, asked, &gone));
                    }
                    return Err(self.ran_out_of_nodes(first_failure, refusal));
                }
            };
            asked = attempt;

            match send(&placement) {
                Ok(value) if trusted_backpressure(&placement, &value) => {
                    saturated.push(placement.decision.label.clone());
                    // The first refusal is the one a client would have received
                    // before this existed, so it is the one to fall back to.
                    if refused.is_none() {
                        refused = Some((value, placement));
                    }
                }
                Ok(value) => {
                    // A node this request could not reach was named by the
                    // current observation, so that observation must not outlive
                    // the request even though the request itself succeeded.
                    if !gone.is_empty() {
                        self.forget_observation();
                    }
                    return Ok(Sent {
                        value,
                        placement,
                        attempts: attempt,
                    });
                }
                Err(error) if error.node_never_received_it() => {
                    placement.invalidate_service_time();
                    gone.push(placement.decision.label.clone());
                    first_failure.get_or_insert((error, placement.decision.engine));
                }
                // The node that took the request is the one that ended it, so
                // that is what gets reported. An earlier node found gone was
                // survived — naming it would send an operator to the node the
                // fabric successfully routed around, and say nothing about the
                // one actually failing requests.
                Err(error) => {
                    if !matches!(error, ForwardError::Cancelled { .. }) {
                        placement.invalidate_service_time();
                    }
                    if gone.is_empty() {
                        self.forget_observation_if_node_vanished(&error);
                    } else {
                        // An observation that named a node now gone must not
                        // outlive this request, however the request ended.
                        self.forget_observation();
                    }
                    return Err(DispatchError::Forward {
                        error,
                        engine: Some(placement.decision.engine),
                    });
                }
            }
        }

        if let Some((value, placement)) = refused {
            return Ok(self.settle_for_the_refusal(value, placement, asked, &gone));
        }

        // The budget ran out with nodes possibly still untried. Say so with the
        // failure that started it, not with a count.
        let (error, engine) = first_failure.expect("the budget can only run out after a failure");
        self.forget_observation();
        Err(DispatchError::Forward {
            error,
            engine: Some(engine),
        })
    }

    /// Settle on the failure to report when placement runs out of nodes.
    ///
    /// Placement can only run out on a later attempt, because this request
    /// excluded the nodes it just found gone. That finding is the story worth
    /// telling; a refusal derived from an exclusion this request invented is
    /// not. The observation is dropped with it: it named a node that is gone.
    /// Hand back a node's own refusal, the alternatives having been tried.
    ///
    /// The observation is dropped only for a node found *gone*, never for a
    /// saturated one: a full node is exactly where the observation said it was,
    /// so forgetting here would re-probe the whole fabric once per request for
    /// as long as it stays busy.
    fn settle_for_the_refusal<T>(
        &self,
        value: T,
        placement: Placement,
        attempts: usize,
        gone: &[String],
    ) -> Sent<T> {
        if !gone.is_empty() {
            self.forget_observation();
        }
        Sent {
            value,
            placement,
            attempts,
        }
    }

    fn ran_out_of_nodes(
        &self,
        first_failure: Option<(ForwardError, NodeEngine)>,
        refusal: RouteError,
    ) -> DispatchError {
        match first_failure {
            Some((error, engine)) => {
                self.forget_observation();
                DispatchError::Forward {
                    error,
                    engine: Some(engine),
                }
            }
            None => DispatchError::Route(refusal),
        }
    }
}

/// Group requests whose service cost is comparable enough to learn together.
///
/// Raw wall time without a workload class would teach the policy that the node
/// receiving longer prompts or generations is intrinsically slower. Powers of
/// two keep the number of classes bounded while separating order-of-magnitude
/// differences. Streaming is distinct because its measured lifetime ends at
/// body EOF rather than at one buffered response.
fn service_class(path: &str, body: &Value) -> String {
    fn bucket(value: usize) -> usize {
        value.checked_next_power_of_two().unwrap_or(usize::MAX)
    }

    let request_bytes = serde_json::to_vec(body).map_or(0, |encoded| encoded.len());
    let output_tokens = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0);
    format!(
        "{path}|stream={}|bytes={}|output={}",
        wants_streaming(body),
        bucket(request_bytes),
        bucket(output_tokens),
    )
}

/// What placement must hold a node to before this request may land on it,
/// read from the request. Pure.
///
/// This and [`service_class`] are the only readers of a request body in
/// placement, and between them they read: whether a top-level `tools` or
/// `functions` key is present, never its contents; which route it came in on;
/// and, for service time, `stream`, the token budget and the size bucket.
/// Message content reaches placement only through that bucket. `tools: []`
/// counts as carrying tools: failing closed costs our own engine nothing, and
/// a client that always sends the key has told us nothing by sending it empty.
/// A tool-loop follow-up carrying `role: "tool"` messages without `tools` is
/// not detected, because seeing it would mean reading the messages.
pub fn placement_requirements(path: &str, body: &Value) -> Requirements {
    let present = |key: &str| body.get(key).is_some_and(|value| !value.is_null());
    Requirements {
        tool_calls: present("tools") || present("functions"),
        rerank_route: matches!(
            path,
            "/rerank" | "/reranking" | "/v1/rerank" | "/v1/reranking"
        ),
    }
}

/// Whether `answer` is a queue-full refusal from a node whose engine declares
/// that refusal, which is the only kind that can safely be sent elsewhere.
fn trusted_backpressure(placement: &Placement, answer: &impl NodeAnswer) -> bool {
    answer.refused_for_backpressure()
        && placement.node().capabilities().typed_backpressure.supported == Some(true)
}

/// The body to send the node `placement` chose: the client's, with `model`
/// set to the id that node knows the model by when that differs. Nothing
/// else is ever written, and a body that needs no change is not copied.
fn body_for<'b>(placement: &Placement, body: &'b Value) -> Cow<'b, Value> {
    let Some(model) = placement.decision.model.as_deref() else {
        return Cow::Borrowed(body);
    };
    if body.get("model").and_then(Value::as_str) == Some(model) {
        return Cow::Borrowed(body);
    }
    let mut rewritten = body.clone();
    match rewritten.as_object_mut() {
        Some(object) => {
            object.insert("model".to_string(), Value::String(model.to_string()));
            Cow::Owned(rewritten)
        }
        None => Cow::Borrowed(body),
    }
}

/// One model id a request could name, and what the listing rests on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServableModel {
    pub id: String,
    pub identity: ModelIdentity,
}

/// A request that a node took, and what it cost to get there.
struct Sent<T> {
    value: T,
    placement: Placement,
    attempts: usize,
}

/// A complete answer from the node that served the request.
#[derive(Debug)]
pub struct Dispatched {
    pub decision: RouteDecision,
    pub answer: Forwarded,
    /// Nodes this request was sent to, the one that answered included. More
    /// than one means a node was found gone, or full, and the request was
    /// placed again.
    pub attempts: usize,
}

/// A streaming answer, still arriving from the node that took the request.
#[derive(Debug)]
pub struct DispatchedStream {
    pub outcome: StreamOutcome,
    /// Holds the node reserved; keep it until the last event is read.
    pub placement: Placement,
    /// As on [`Dispatched`]. Settled before the first byte is relayed, so it
    /// can be reported in the response head.
    pub attempts: usize,
}

/// A poisoned lock is not corrupted state: some other request panicked while
/// holding it. Recover the value rather than spreading the panic.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A node chosen for one request, and the fabric's record that it is busy.
///
/// The chosen node counts this request until this value is dropped, so hold it
/// for exactly as long as the request runs. Dropping it early tells the fabric
/// the node is free when it is not, and the next placement will pile onto it.
#[must_use = "dropping a Placement releases the node it reserved"]
pub struct Placement {
    decision: RouteDecision,
    node: NodeSnapshot,
    reserved: Arc<Mutex<Reservations>>,
    service_times: Arc<Mutex<ServiceTimeEstimates>>,
    service_model: Option<String>,
    service_class: Option<String>,
    service_mode: RouteMode,
    service_ahead: usize,
    service_recorded: bool,
}

impl Placement {
    pub fn decision(&self) -> &RouteDecision {
        &self.decision
    }

    /// The chosen node as it was observed, so a caller can read what it is
    /// already serving before building a request body for it.
    pub fn node(&self) -> &NodeSnapshot {
        &self.node
    }

    /// Record one successful, fully completed request.
    ///
    /// Failures and cancellations never call this. Idempotence is defensive:
    /// one request must never outweigh another because two completion paths
    /// happened to report it.
    pub(crate) fn record_success(&mut self, elapsed: Duration) {
        if self.service_recorded || self.service_mode != RouteMode::CompletionTime {
            return;
        }
        self.service_recorded = true;
        // A busy completion's wall time includes queueing from work whose
        // workload class is not exposed by node health. Dividing by the total
        // in-flight count would fabricate this class's service time, so only a
        // request placed with nobody ahead becomes a sample.
        if self.service_ahead != 0 {
            return;
        }
        let (Some(model), Some(service_class)) = (&self.service_model, &self.service_class) else {
            return;
        };
        lock(&self.service_times).observe(&self.node, model, service_class, elapsed);
    }

    /// Forget a learned speed after the node itself fails this workload.
    /// Selection recency is retained so cold fallback explores a sibling next.
    pub(crate) fn invalidate_service_time(&mut self) {
        let (Some(model), Some(service_class)) = (&self.service_model, &self.service_class) else {
            return;
        };
        lock(&self.service_times).invalidate(&self.node, model, service_class);
    }
}

impl std::fmt::Debug for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Placement")
            .field("decision", &self.decision)
            .field("node", &self.node.label())
            .finish_non_exhaustive()
    }
}

impl Drop for Placement {
    fn drop(&mut self) {
        lock(&self.reserved).release(&self.decision.label);
    }
}

/// Why a dispatch did not produce an answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// No node could take the request.
    Route(RouteError),
    /// A node was chosen, but the request did not complete against it.
    Forward {
        error: ForwardError,
        /// The engine of the node it was sent to, when it was sent anywhere,
        /// so a failure names its engine the way an answer does.
        engine: Option<NodeEngine>,
    },
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Route(error) => write!(f, "{error}"),
            Self::Forward { error, .. } => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<RouteError> for DispatchError {
    fn from(error: RouteError) -> Self {
        Self::Route(error)
    }
}

impl From<ForwardError> for DispatchError {
    fn from(error: ForwardError) -> Self {
        Self::Forward {
            error,
            engine: None,
        }
    }
}

/// Counts of each observed state, for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FabricSummary {
    pub ready: usize,
    pub not_ready: usize,
    pub unreachable: usize,
}

impl FabricSummary {
    pub fn of(snapshots: &[NodeSnapshot]) -> Self {
        let mut summary = Self::default();
        for snapshot in snapshots {
            match snapshot.status {
                NodeStatus::Ready(_) => summary.ready += 1,
                NodeStatus::NotReady { .. } => summary.not_ready += 1,
                NodeStatus::Unreachable { .. } => summary.unreachable += 1,
            }
        }
        summary
    }

    pub fn total(&self) -> usize {
        self.ready + self.not_ready + self.unreachable
    }
}

/// Every model the fabric can serve right now, deduplicated and ordered.
///
/// Only ready nodes count. A node that is unreachable, or that has no model
/// loaded, cannot serve one — listing its model would advertise something the
/// fabric would then refuse.
///
/// Kept pure so the listing is covered by a unit test rather than by starting a
/// server, the same way [`render_status`] is.
/// Every model the fabric would actually place a request on.
///
/// Deliberately not "every model any node holds": this list is what
/// `GET /v1/models` answers with, and advertising a model the fabric would then
/// refuse to route is worse than not advertising it. A node running an engine
/// this fabric does not place on is therefore absent here, however healthy it
/// is — its models are still visible per node in the status detail.
pub fn servable_models(snapshots: &[NodeSnapshot]) -> Vec<String> {
    let mut models: Vec<String> = snapshots
        .iter()
        .filter(|snapshot| snapshot.is_placeable())
        .flat_map(|snapshot| snapshot.models().iter().cloned())
        .collect();
    models.sort();
    models.dedup();
    models
}

/// Render a fabric observation as fixed-width text.
///
/// Kept pure so the CLI's output is covered by a unit test rather than by
/// eyeballing a terminal.
pub fn render_status(snapshots: &[NodeSnapshot]) -> String {
    if snapshots.is_empty() {
        return "no nodes configured\n".to_string();
    }

    let label_width = snapshots
        .iter()
        .map(|s| s.label().len())
        .max()
        .unwrap_or(5)
        .max(5);
    let authority_width = snapshots
        .iter()
        .map(|s| s.spec.authority().len())
        .max()
        .unwrap_or(8)
        .max(8);

    let mut out = String::new();
    out.push_str(&format!(
        "{:<label_width$}  {:<authority_width$}  {:<9}  {:<24}  {:<6}  {}\n",
        "NODE",
        "ADDRESS",
        "STATE",
        "MODEL",
        "LOAD",
        "DETAIL",
        label_width = label_width,
        authority_width = authority_width,
    ));

    for snapshot in snapshots {
        let (state, model, load, detail) = match &snapshot.status {
            NodeStatus::Ready(ready) => (
                "ready",
                ready.active_model_id.as_deref().unwrap_or("-").to_string(),
                // An engine that publishes no load reads as unknown. A zero
                // here would put an idle machine in the table.
                ready
                    .in_flight()
                    .map_or_else(|| "?".to_string(), |in_flight| in_flight.to_string()),
                {
                    let lane = ready.backend.as_deref().unwrap_or(ready.engine.as_str());
                    match snapshot.latency {
                        Some(latency) => format!("{} · {} ms", lane, latency.as_millis()),
                        None => lane.to_string(),
                    }
                },
            ),
            NodeStatus::NotReady { reason } => (
                "not-ready",
                "-".to_string(),
                "-".to_string(),
                reason.clone(),
            ),
            NodeStatus::Unreachable { reason } => {
                ("offline", "-".to_string(), "-".to_string(), reason.clone())
            }
        };
        out.push_str(&format!(
            "{:<label_width$}  {:<authority_width$}  {:<9}  {:<24}  {:<6}  {}\n",
            snapshot.label(),
            snapshot.spec.authority(),
            state,
            model,
            load,
            detail,
            label_width = label_width,
            authority_width = authority_width,
        ));
    }

    let summary = FabricSummary::of(snapshots);
    out.push_str(&format!(
        "\n{} node(s): {} ready, {} not ready, {} offline\n",
        summary.total(),
        summary.ready,
        summary.not_ready,
        summary.unreachable
    ));
    out
}

/// What a starting proxy tells its operator about the fabric behind it.
///
/// A proxy that cannot reach a node still starts: the node may be booting, and a
/// fabric that refuses to come up because one member is late is less available
/// than the member it is waiting for. But starting silently is worse — with only
/// a listening line to go on, a node whose name never resolves leaves the proxy
/// serving at reduced capacity forever with nothing said, and the first symptom
/// is a latency graph nobody can explain.
///
/// So: report, never refuse. Every node that cannot take work is named, with the
/// address that was tried and the reason it failed, because "which of my nodes,
/// and why" is the whole question an operator has at that moment.
///
/// Kept pure so the wording is covered by a unit test rather than by starting a
/// server, the same way [`render_status`] is.
pub fn startup_report(snapshots: &[NodeSnapshot]) -> String {
    startup_report_serving(snapshots, &servable_models(snapshots))
}

/// [`startup_report`], naming `models` as what is served — the listing the
/// proxy will actually answer with, in its mode and with its aliases.
pub fn startup_report_serving(snapshots: &[NodeSnapshot], models: &[String]) -> String {
    if snapshots.is_empty() {
        return "fabric: no nodes configured\n".to_string();
    }

    let summary = FabricSummary::of(snapshots);
    let mut out = format!(
        "fabric: {} of {} nodes ready",
        summary.ready,
        summary.total()
    );
    if models.is_empty() {
        out.push_str("; no model is being served\n");
    } else {
        out.push_str(&format!("; serving {}\n", models.join(", ")));
    }

    for snapshot in snapshots {
        let (state, reason) = match &snapshot.status {
            NodeStatus::Ready(_) => continue,
            NodeStatus::NotReady { reason } => ("not ready", reason),
            NodeStatus::Unreachable { reason } => ("unreachable", reason),
        };
        out.push_str(&format!(
            "  {} ({}) {state}: {reason}\n",
            snapshot.label(),
            snapshot.spec.authority(),
        ));
    }

    if summary.ready == 0 {
        out.push_str("fabric: every request will be refused until a node becomes ready\n");
    }

    out
}

/// What a starting proxy tells its operator about where it places work. Pure.
///
/// Under mixed placement each node of an engine not placed on by default is
/// named with everything accepted about it and every request it will still
/// never be sent, in the build's own words, because that is the grant the
/// operator just made. `reloadable` adds that the grant covers nodes added to
/// the file later, and `aliases` lists the declarations in force.
pub fn placement_report(
    snapshots: &[NodeSnapshot],
    mixed: MixedEngines,
    reloadable: bool,
    aliases: &[ModelAlias],
) -> String {
    let mut out = String::new();
    match mixed {
        MixedEngines::Refused => out.push_str("placement: Camelid engines only\n"),
        MixedEngines::Allowed => {
            out.push_str(&format!(
                "placement: ALSO placing on other engines ({MIXED_ENGINES_FLAG})\n"
            ));
            for snapshot in snapshots {
                let capabilities = snapshot.capabilities();
                let blockers = capabilities.placement_blockers();
                if blockers.is_empty() {
                    continue;
                }
                let engine = snapshot.engine_and_version();
                let never: Vec<String> = capabilities
                    .requirement_limits(&engine)
                    .into_iter()
                    .map(|limit| limit.consequence)
                    .collect();
                out.push_str(&format!(
                    "  {} ({engine}): accepted without asking: {}.",
                    snapshot.label(),
                    blockers.join("; ")
                ));
                if !never.is_empty() {
                    out.push_str(&format!(" {}.", never.join("; ")));
                }
                out.push('\n');
            }
            if reloadable {
                out.push_str(
                    "  note: nodes of other engines added to the nodes file later are accepted \
                     too, and are announced here.\n",
                );
            }
        }
    }
    if !aliases.is_empty() {
        let mut by_canonical: Vec<(String, Vec<String>)> = Vec::new();
        for alias in aliases {
            let target = format!("{}:{}", alias.label, alias.local);
            match by_canonical
                .iter_mut()
                .find(|(canonical, _)| *canonical == alias.canonical)
            {
                Some((_, targets)) => targets.push(target),
                None => by_canonical.push((alias.canonical.clone(), vec![target])),
            }
        }
        let listed: Vec<String> = by_canonical
            .into_iter()
            .map(|(canonical, targets)| format!("{canonical} -> {}", targets.join(", ")))
            .collect();
        out.push_str(&format!(
            "aliases in force (read at startup; edit and restart to change): {}\n",
            listed.join("; ")
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(label: &str, status: NodeStatus) -> NodeSnapshot {
        NodeSnapshot {
            spec: NodeSpec::camelid(label, "127.0.0.1", 8181),
            status,
            latency: Some(Duration::from_millis(4)),
        }
    }

    fn ready_status(model: &str) -> NodeStatus {
        ready_status_with_load(model, 1)
    }

    /// Load is a parameter because the service-time tests turn on it: a sample
    /// taken while a node had work queued is not a measurement of that node.
    fn ready_status_with_load(model: &str, in_flight: usize) -> NodeStatus {
        NodeStatus::Ready(NodeReady {
            engine: crate::fabric::engine::NodeEngine::Camelid,
            active_model_id: Some(model.to_string()),
            models: vec![model.to_string()],
            resident_models: Some(vec![model.to_string()]),
            backend: Some("llama".to_string()),
            version: Some("0.5.4".to_string()),
            load: Some(crate::fabric::node::NodeLoad {
                in_flight,
                waiting: 0,
            }),
        })
    }

    fn ready_snapshot(spec: NodeSpec, model: &str) -> NodeSnapshot {
        NodeSnapshot {
            spec,
            status: ready_status(model),
            latency: Some(Duration::from_millis(4)),
        }
    }

    #[test]
    fn a_tls_authentication_failure_fails_over_without_leaking_the_bearer() {
        use std::io::{Read, Write};

        let issued = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .expect("generate trusted node certificate");
        let key = rustls_pki_types::PrivatePkcs8KeyDer::from(issued.key_pair.serialize_der());
        let server = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![issued.cert.der().clone()], key.into())
            .expect("trusted certificate and key agree");
        let directory = tempfile::tempdir().expect("temp dir");
        let ca_path = directory.path().join("node-ca");
        std::fs::write(&ca_path, rcgen::Certificate::pem(&issued.cert)).expect("write CA");

        let impostor = std::net::TcpListener::bind("127.0.0.1:0").expect("bind impostor");
        let impostor_port = impostor.local_addr().expect("impostor address").port();
        let (captured_tx, captured_rx) = std::sync::mpsc::channel();
        let impostor_thread = std::thread::spawn(move || {
            let (mut socket, _) = impostor.accept().expect("accept first attempt");
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("bound impostor read");
            let mut bytes = vec![0_u8; 8192];
            let read = socket.read(&mut bytes).expect("read ClientHello");
            bytes.truncate(read);
            captured_tx.send(bytes).expect("send captured bytes");
        });

        let trusted = std::net::TcpListener::bind("127.0.0.1:0").expect("bind trusted node");
        let trusted_port = trusted.local_addr().expect("trusted address").port();
        let trusted_thread = std::thread::spawn(move || {
            let (socket, _) = trusted.accept().expect("accept failover attempt");
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("bound trusted read");
            socket
                .set_write_timeout(Some(Duration::from_secs(2)))
                .expect("bound trusted write");
            let connection =
                rustls::ServerConnection::new(Arc::new(server)).expect("create TLS server");
            let mut stream = rustls::StreamOwned::new(connection, socket);
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream
                    .read(&mut buffer)
                    .expect("read authenticated request");
                assert_ne!(read, 0, "request ended before its headers");
                request.extend_from_slice(&buffer[..read]);
            }
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .expect("request headers end")
                + 4;
            let content_length = std::str::from_utf8(&request[..header_end])
                .expect("ASCII request headers")
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length").then(|| {
                        value
                            .trim()
                            .parse::<usize>()
                            .expect("numeric content length")
                    })
                })
                .unwrap_or(0);
            while request.len() - header_end < content_length {
                let read = stream
                    .read(&mut buffer)
                    .expect("read authenticated request body");
                assert_ne!(read, 0, "request ended before its body");
                request.extend_from_slice(&buffer[..read]);
            }
            let body = r#"{"choices":[{"message":{"role":"assistant","content":"served"}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .expect("write authenticated answer");
            stream.flush().expect("flush authenticated answer");
        });

        let bad = NodeSpec::camelid("a-impostor", "localhost", impostor_port);
        let good = NodeSpec::camelid("b-trusted", "localhost", trusted_port);
        let bearer = "must-not-cross-before-authentication-74d1";
        let fabric = Fabric::new(vec![bad.clone(), good.clone()])
            .with_bearer(Some(bearer))
            .with_node_transport(Some(&ca_path), false)
            .expect("TLS transport resolves")
            .with_max_observation_age(Duration::from_secs(30))
            .with_max_forward_attempts(2);
        *lock(&fabric.observed) = Some((
            0,
            Observation::taken_at(
                vec![ready_snapshot(bad, "m"), ready_snapshot(good, "m")],
                Instant::now(),
            ),
        ));

        let dispatched = fabric
            .dispatch(
                "/v1/chat/completions",
                &forward::chat_request("m", "ping", 8),
                &RouteRequest::new(RouteMode::Throughput).with_model(Some("m")),
                Duration::from_secs(3),
                &Cancel::never(),
            )
            .expect("the authenticated sibling serves after TLS refusal");
        assert_eq!(dispatched.answer.label, "b-trusted");
        assert_eq!(dispatched.attempts, 2);

        let captured = captured_rx.recv().expect("captured ClientHello");
        assert!(
            !captured
                .windows(bearer.len())
                .any(|window| window == bearer.as_bytes()),
            "bearer crossed before server authentication"
        );
        impostor_thread.join().expect("impostor exits");
        trusted_thread.join().expect("trusted node exits");
    }

    #[test]
    fn service_classes_separate_workloads_with_different_cost_shapes() {
        let base = serde_json::json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "short" }],
            "max_tokens": 64,
            "stream": false,
        });
        let base_class = service_class("/v1/chat/completions", &base);

        let mut streaming = base.clone();
        streaming["stream"] = true.into();
        assert_ne!(
            base_class,
            service_class("/v1/chat/completions", &streaming)
        );

        let mut larger_output = base.clone();
        larger_output["max_tokens"] = 65.into();
        assert_ne!(
            base_class,
            service_class("/v1/chat/completions", &larger_output)
        );

        let mut larger_input = base.clone();
        larger_input["messages"][0]["content"] = "x".repeat(1_024).into();
        assert_ne!(
            base_class,
            service_class("/v1/chat/completions", &larger_input)
        );
        assert_ne!(base_class, service_class("/v1/embeddings", &base));
    }

    #[test]
    fn a_busy_completion_is_not_a_service_sample() {
        let node = snapshot("a", ready_status("m"));
        let measured = node.clone();
        let fabric = Fabric::new(Vec::new());
        {
            let mut reserved = lock(&fabric.reserved);
            reserved.take("a");
            reserved.take("a");
        }
        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("class"));

        let mut placement = fabric
            .place_observed(vec![node], &request, &[])
            .expect("routes");

        assert_eq!(
            placement.service_ahead, 2,
            "sampling must use the same max(observed, reserved) load as selection"
        );
        assert_eq!(fabric.reserved().get("a"), 3);
        placement.record_success(Duration::from_millis(300));
        let (_, samples) = lock(&fabric.service_times)
            .sample_for(&measured, "m", "class")
            .expect("selection recency keeps the class present");
        assert_eq!(samples, 0, "busy queueing became a speed sample");
        drop(placement);
        assert_eq!(fabric.reserved().get("a"), 2);
    }

    #[test]
    fn a_queue_free_completion_records_its_observed_wall_time() {
        let node = snapshot("a", ready_status_with_load("m", 0));
        let measured = node.clone();
        let fabric = Fabric::new(Vec::new());
        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("class"));
        let mut placement = fabric
            .place_observed(vec![node], &request, &[])
            .expect("routes");

        placement.record_success(Duration::from_millis(400));

        let (mean_nanos, samples) = lock(&fabric.service_times)
            .sample_for(&measured, "m", "class")
            .expect("the queue-free completion was sampled");
        assert_eq!(samples, 1);
        assert_eq!(mean_nanos, Duration::from_millis(400).as_nanos());
    }

    #[test]
    fn service_classes_do_not_record_request_content() {
        let body = serde_json::json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "private prompt" }],
            "max_tokens": 64,
        });
        let class = service_class("/v1/chat/completions", &body);
        assert!(!class.contains("private"), "{class}");
        assert!(!class.contains("prompt"), "{class}");
    }

    /// I12, pinned. Two bodies that differ in every piece of content but not
    /// in encoded size must be placed identically: what placement reads of
    /// content is the size bucket and the presence of a key, nothing else.
    #[test]
    fn contents_of_equal_encoded_size_cannot_change_requirements_or_class() {
        let one = serde_json::json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "aaaaaaaaaaaaaaaa" }],
            "tools": [{ "type": "function", "function": { "name": "get_weather" } }],
        });
        let two = serde_json::json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "bbbbbbbbbbbbbbbb" }],
            "tools": [{ "type": "function", "function": { "name": "put_weather" } }],
        });
        assert_eq!(
            serde_json::to_vec(&one).expect("encodes").len(),
            serde_json::to_vec(&two).expect("encodes").len(),
            "the two bodies must be the same size for this test to mean anything"
        );
        for path in ["/v1/chat/completions", "/v1/embeddings"] {
            assert_eq!(
                placement_requirements(path, &one),
                placement_requirements(path, &two)
            );
            assert_eq!(service_class(path, &one), service_class(path, &two));
        }

        let carries = |body: serde_json::Value| {
            placement_requirements("/v1/chat/completions", &body).tool_calls
        };
        assert!(carries(
            serde_json::json!({ "tools": [{ "type": "function" }] })
        ));
        assert!(
            carries(serde_json::json!({ "tools": [] })),
            "an empty list still fails closed"
        );
        assert!(carries(
            serde_json::json!({ "functions": [{ "name": "f" }] })
        ));
        assert!(!carries(serde_json::json!({ "tools": null })));
        assert!(!carries(serde_json::json!({})));
        assert!(
            !carries(serde_json::json!({
                "messages": [{ "role": "tool", "content": "{}", "tool_call_id": "c" }]
            })),
            "messages are not read, so a tool-result turn without `tools` is not detected"
        );

        for path in ["/v1/rerank", "/v1/reranking", "/rerank", "/reranking"] {
            assert!(placement_requirements(path, &one).rerank_route, "{path}");
        }
        assert!(!placement_requirements("/v1/chat/completions", &one).rerank_route);
        assert!(!placement_requirements("/v1/embeddings", &one).rerank_route);
    }

    #[test]
    fn only_the_size_bucket_of_content_reaches_the_service_class() {
        let with = |content: String| {
            serde_json::json!({
                "model": "m",
                "messages": [{ "role": "user", "content": content }],
                "max_tokens": 64,
            })
        };
        let small = with("x".repeat(40));
        let large = with("y".repeat(4_000));
        let small_class = service_class("/v1/chat/completions", &small);
        let large_class = service_class("/v1/chat/completions", &large);
        assert_ne!(
            small_class, large_class,
            "crossing a power of two moves the bucket"
        );

        let strip = |class: &str| -> Vec<String> {
            class
                .split('|')
                .filter(|part| !part.starts_with("bytes="))
                .map(str::to_string)
                .collect()
        };
        assert_eq!(
            strip(&small_class),
            strip(&large_class),
            "and nothing but the bucket"
        );
        assert_eq!(
            placement_requirements("/v1/chat/completions", &small),
            placement_requirements("/v1/chat/completions", &large)
        );
    }

    #[test]
    fn the_startup_placement_line_says_what_the_flag_accepts_and_that_it_stands() {
        let mut foreign_spec = NodeSpec::camelid("b-ollama", "127.0.0.1", 11434);
        foreign_spec.engine = NodeEngine::Ollama;
        let foreign = NodeSnapshot {
            spec: foreign_spec,
            status: NodeStatus::Ready(NodeReady {
                engine: NodeEngine::Ollama,
                active_model_id: None,
                models: vec!["m:latest".to_string()],
                resident_models: Some(Vec::new()),
                backend: None,
                version: Some("0.33.2".to_string()),
                load: None,
            }),
            latency: None,
        };
        let nodes = vec![snapshot("a-camelid", ready_status("m")), foreign];

        assert_eq!(
            placement_report(&nodes, MixedEngines::Refused, true, &[]),
            "placement: Camelid engines only\n"
        );
        let mixed = placement_report(&nodes, MixedEngines::Allowed, true, &[]);
        assert!(mixed.contains(MIXED_ENGINES_FLAG), "{mixed}");
        assert!(
            mixed.contains("b-ollama (ollama 0.33.2): accepted without asking"),
            "{mixed}"
        );
        assert!(mixed.contains("publishes no load to rank on"), "{mixed}");
        assert!(
            mixed.contains("requests carrying tools are never placed here"),
            "{mixed}"
        );
        assert!(mixed.contains("added to the nodes file later"), "{mixed}");
        assert!(!mixed.contains("a-camelid"), "{mixed}");
        assert!(
            !placement_report(&nodes, MixedEngines::Allowed, false, &[])
                .contains("added to the nodes file later"),
            "a fixed set covers nothing added later"
        );

        let aliases = parse_model_aliases(&[
            "llama-1b=a-camelid:m".to_string(),
            "llama-1b=b-ollama:m:latest".to_string(),
        ])
        .expect("parses")
        .declared();
        let listed = placement_report(&nodes, MixedEngines::Refused, true, &aliases);
        assert!(
            listed.contains("aliases in force (read at startup; edit and restart to change): llama-1b -> a-camelid:m, b-ollama:m:latest"),
            "{listed}"
        );
    }

    #[test]
    fn the_servable_list_is_the_union_of_ready_nodes() {
        let snapshots = vec![
            snapshot("b", ready_status("zeta")),
            snapshot("a", ready_status("alpha")),
            // The same model on a second node is one entry, not two.
            snapshot("c", ready_status("alpha")),
        ];
        assert_eq!(servable_models(&snapshots), vec!["alpha", "zeta"]);
    }

    #[test]
    fn a_node_that_cannot_serve_is_not_advertised() {
        let blank = NodeStatus::Ready(NodeReady {
            engine: crate::fabric::engine::NodeEngine::Camelid,
            active_model_id: None,
            models: Vec::new(),
            resident_models: Some(Vec::new()),
            backend: Some("llama".to_string()),
            version: Some("0.5.4".to_string()),
            load: Some(crate::fabric::node::NodeLoad {
                in_flight: 0,
                waiting: 0,
            }),
        });
        let snapshots = vec![
            snapshot("a", ready_status("alpha")),
            snapshot(
                "dead",
                NodeStatus::Unreachable {
                    reason: "connection refused".to_string(),
                },
            ),
            snapshot(
                "empty",
                NodeStatus::NotReady {
                    reason: "no model loaded".to_string(),
                },
            ),
            snapshot("blank", blank),
        ];
        assert_eq!(servable_models(&snapshots), vec!["alpha"]);
    }

    #[test]
    fn nothing_ready_lists_nothing_rather_than_failing() {
        assert!(servable_models(&[]).is_empty());
    }

    #[test]
    fn an_empty_fabric_renders_a_sentence_not_an_empty_table() {
        assert_eq!(render_status(&[]), "no nodes configured\n");
    }

    #[test]
    fn the_summary_counts_every_state() {
        let snapshots = vec![
            snapshot("a", ready_status("llama-3b")),
            snapshot(
                "b",
                NodeStatus::NotReady {
                    reason: "no model loaded".to_string(),
                },
            ),
            snapshot(
                "c",
                NodeStatus::Unreachable {
                    reason: "cannot connect".to_string(),
                },
            ),
        ];
        let summary = FabricSummary::of(&snapshots);
        assert_eq!(summary.ready, 1);
        assert_eq!(summary.not_ready, 1);
        assert_eq!(summary.unreachable, 1);
        assert_eq!(summary.total(), 3);
    }

    #[test]
    fn the_table_reports_model_load_and_failure_detail() {
        let snapshots = vec![
            snapshot("windows", ready_status("llama-3b")),
            snapshot(
                "mac",
                NodeStatus::Unreachable {
                    reason: "cannot connect: refused".to_string(),
                },
            ),
        ];
        let rendered = render_status(&snapshots);
        assert!(rendered.contains("windows"), "{rendered}");
        assert!(rendered.contains("llama-3b"), "{rendered}");
        assert!(rendered.contains("LOAD"), "{rendered}");
        assert!(rendered.contains("cannot connect: refused"), "{rendered}");
        assert!(rendered.contains("2 node(s): 1 ready"), "{rendered}");
    }

    #[test]
    fn the_startup_report_names_a_node_it_cannot_reach() {
        let snapshots = vec![
            snapshot("windows", ready_status("llama-3b")),
            snapshot(
                "mac",
                NodeStatus::Unreachable {
                    reason: "cannot resolve host: No such host is known. (os error 11001)"
                        .to_string(),
                },
            ),
        ];
        let report = startup_report(&snapshots);
        assert!(report.contains("1 of 2 nodes ready"), "{report}");
        assert!(report.contains("serving llama-3b"), "{report}");
        // The label, the address that was tried, and the reason: an operator
        // cannot act on any two of those three.
        assert!(
            report.contains("mac (127.0.0.1:8181) unreachable"),
            "{report}"
        );
        assert!(report.contains("os error 11001"), "{report}");
        // A node that is doing its job is not worth a line of its own.
        assert!(!report.contains("windows"), "{report}");
    }

    #[test]
    fn a_healthy_fabric_reports_one_line_and_names_no_node() {
        let snapshots = vec![
            snapshot("a", ready_status("llama-3b")),
            snapshot("b", ready_status("llama-3b")),
        ];
        assert_eq!(
            startup_report(&snapshots),
            "fabric: 2 of 2 nodes ready; serving llama-3b\n"
        );
    }

    #[test]
    fn a_startup_report_says_when_nothing_can_be_served_at_all() {
        let snapshots = vec![
            snapshot(
                "a",
                NodeStatus::Unreachable {
                    reason: "cannot connect".to_string(),
                },
            ),
            snapshot(
                "b",
                NodeStatus::Unreachable {
                    reason: "cannot connect".to_string(),
                },
            ),
        ];
        let report = startup_report(&snapshots);
        assert!(report.contains("0 of 2 nodes ready"), "{report}");
        assert!(report.contains("no model is being served"), "{report}");
        assert!(
            report.contains("every request will be refused until a node becomes ready"),
            "{report}"
        );
    }

    #[test]
    fn a_startup_report_separates_answering_late_from_not_answering() {
        // These are different operator problems: one node is booting, the other
        // is not there. Collapsing them into "down" sends you to the wrong box.
        let snapshots = vec![
            snapshot(
                "warming",
                NodeStatus::NotReady {
                    reason: "no model loaded".to_string(),
                },
            ),
            snapshot(
                "gone",
                NodeStatus::Unreachable {
                    reason: "cannot connect".to_string(),
                },
            ),
        ];
        let report = startup_report(&snapshots);
        assert!(
            report.contains("warming (127.0.0.1:8181) not ready: no model loaded"),
            "{report}"
        );
        assert!(
            report.contains("gone (127.0.0.1:8181) unreachable: cannot connect"),
            "{report}"
        );
    }

    #[test]
    fn a_fabric_with_no_specs_observes_nothing() {
        let fabric = Fabric::new(Vec::new());
        assert!(fabric.is_empty());
        assert!(fabric.observe().is_empty());
    }

    #[test]
    fn a_configured_token_is_never_exposed_by_debug() {
        // A `Fabric` is the kind of thing that ends up in an error context or a
        // trace line, so the derived `Debug` would have been a leak.
        let authenticated = format!("{:?}", Fabric::new(Vec::new()).with_bearer(Some("s3cret")));
        assert!(!authenticated.contains("s3cret"), "{authenticated}");
        assert!(authenticated.contains("REDACTED"), "{authenticated}");

        // Having no token reads as absent rather than as a redacted one.
        let open = format!("{:?}", Fabric::new(Vec::new()));
        assert!(open.contains("bearer: None"), "{open}");
    }

    #[test]
    fn dispatch_refuses_a_streaming_request_before_probing_anything() {
        // The node here is unreachable on purpose: if dispatch probed first, this
        // would surface as a routing failure instead of the streaming refusal.
        let fabric = Fabric::new(vec![NodeSpec::camelid("dead", "127.0.0.1", 1)]);
        let body = serde_json::json!({ "model": "m", "stream": true });
        let error = fabric
            .dispatch(
                "/v1/chat/completions",
                &body,
                &RouteRequest::new(RouteMode::Throughput),
                Duration::from_millis(200),
                &Cancel::never(),
            )
            .expect_err("streaming is unsupported");
        assert!(
            matches!(
                error,
                DispatchError::Forward {
                    error: ForwardError::Unsupported(_),
                    engine: None
                }
            ),
            "got {error:?}"
        );
    }

    #[test]
    fn dispatch_on_an_empty_fabric_fails_at_placement_not_transport() {
        let fabric = Fabric::new(Vec::new());
        let error = fabric
            .dispatch(
                "/v1/chat/completions",
                &serde_json::json!({ "model": "m" }),
                &RouteRequest::new(RouteMode::Throughput),
                Duration::from_millis(200),
                &Cancel::never(),
            )
            .expect_err("no nodes");
        assert_eq!(error, DispatchError::Route(RouteError::NoNodesConfigured));
    }

    #[test]
    fn dispatch_reports_a_dead_node_as_a_placement_failure() {
        // Every node being unreachable is a routing outcome, not a forward error:
        // there was never a node to send to.
        let fabric = Fabric::new(vec![NodeSpec::camelid("dead", "127.0.0.1", 1)])
            .with_timeout(Duration::from_millis(300));
        let error = fabric
            .dispatch(
                "/v1/chat/completions",
                &serde_json::json!({ "model": "m" }),
                &RouteRequest::new(RouteMode::Throughput),
                Duration::from_millis(300),
                &Cancel::never(),
            )
            .expect_err("node is dead");
        assert!(
            matches!(
                error,
                DispatchError::Route(RouteError::AllNodesUnavailable { .. })
            ),
            "got {error:?}"
        );
    }

    /// The whole point of a node file: a machine the operator adds is placed
    /// on, and one they take away stops being placed on, without a restart.
    ///
    /// The observation window here is an hour. That is deliberate — if a
    /// changed set did not invalidate the observation taken over the old one,
    /// this fabric would go on describing the old machines for that hour.
    #[test]
    fn a_changed_node_set_is_not_answered_from_the_observation_of_the_old_one() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("nodes");
        // Port 1 is never listening, so every probe settles as Unreachable.
        // Which labels come back is the whole question; whether they were
        // reachable is not.
        std::fs::write(&path, "a=127.0.0.1:1\n").expect("write");

        let fabric = Fabric::from_node_file_every(path.clone(), Duration::ZERO)
            .expect("load")
            .with_timeout(Duration::from_millis(50))
            .with_max_observation_age(Duration::from_secs(3600));

        let labels = |snapshots: &[NodeSnapshot]| -> Vec<String> {
            snapshots.iter().map(|s| s.label().to_string()).collect()
        };

        assert_eq!(labels(&fabric.observe()), vec!["a"]);

        std::fs::write(&path, "b=127.0.0.1:1\n").expect("rewrite");

        assert_eq!(
            labels(&fabric.observe()),
            vec!["b"],
            "an observation of the node set as it was must not survive the set changing"
        );
    }

    /// A set that has not changed must keep its observation, or every look at
    /// the file would cost a probe of every node.
    #[test]
    fn an_unchanged_node_file_keeps_the_observation_it_already_has() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("nodes");
        std::fs::write(&path, "a=127.0.0.1:1\n").expect("write");

        let fabric = Fabric::from_node_file_every(path.clone(), Duration::ZERO)
            .expect("load")
            .with_timeout(Duration::from_millis(50))
            .with_max_observation_age(Duration::from_secs(3600));

        let first = fabric.observe();
        // Rewritten with identical content: the stamp moves, the set does not.
        std::fs::write(&path, "a=127.0.0.1:1\n").expect("identical rewrite");
        let second = fabric.observe();

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_eq!(
            first[0].label(),
            second[0].label(),
            "an identical rewrite must not invalidate the observation"
        );
    }

    /// Taking a machine away must not disturb what is already running on it:
    /// reservations are counted by label and released when the placement is
    /// dropped, whether or not the node is still in the set.
    #[test]
    fn removing_a_node_leaves_the_requests_already_placed_on_it_alone() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("nodes");
        std::fs::write(&path, "a=127.0.0.1:1\nb=127.0.0.1:1\n").expect("write");
        let fabric = Fabric::from_node_file_every(path.clone(), Duration::ZERO).expect("load");

        // Reserved by hand: placement needs a ready node, and this test is
        // about the bookkeeping, not about probing.
        lock(&fabric.reserved).take("a");
        assert_eq!(fabric.reserved().get("a"), 1);

        std::fs::write(&path, "b=127.0.0.1:1\n").expect("remove a");
        assert_eq!(fabric.specs().len(), 1, "the set really did shrink");

        assert_eq!(
            fabric.reserved().get("a"),
            1,
            "a request in flight on a removed node must still be counted"
        );
        lock(&fabric.reserved).release("a");
        assert_eq!(
            fabric.reserved().get("a"),
            0,
            "and must still release when it finishes"
        );
    }

    #[test]
    fn a_fixed_fabric_does_not_claim_to_be_reloadable() {
        assert!(!Fabric::new(Vec::new()).is_reloadable());
    }

    #[test]
    fn a_request_is_never_placed_on_a_node_it_has_already_failed_against() {
        let fabric = Fabric::new(Vec::new());
        let snapshots = vec![
            snapshot("a", ready_status("m")),
            snapshot("b", ready_status("m")),
        ];
        let request = RouteRequest::new(RouteMode::Throughput);

        // Ties break on label, so an unconstrained placement picks `a`; the
        // exclusion is what moves it, not a difference between the nodes.
        let first = fabric
            .place_observed(snapshots.clone(), &request, &[])
            .expect("a is eligible");
        assert_eq!(first.decision().label, "a");
        drop(first);

        let second = fabric
            .place_observed(snapshots, &request, &["a".to_string()])
            .expect("b is still eligible");
        assert_eq!(second.decision().label, "b");
    }

    #[test]
    fn excluding_every_node_refuses_rather_than_placing_on_one_anyway() {
        let fabric = Fabric::new(Vec::new());
        let snapshots = vec![snapshot("a", ready_status("m"))];
        let error = fabric
            .place_observed(
                snapshots,
                &RouteRequest::new(RouteMode::Throughput),
                &["a".to_string()],
            )
            .expect_err("nothing is left to place on");
        assert_eq!(error, RouteError::NoNodesConfigured);
    }

    #[test]
    fn a_request_must_always_be_sendable_somewhere() {
        // Zero attempts would mean a request that is never sent at all, so the
        // budget floor is part of the contract rather than a caller's problem.
        let fabric = Fabric::new(Vec::new()).with_max_forward_attempts(0);
        assert_eq!(fabric.max_forward_attempts, 1);
        assert_eq!(
            Fabric::new(Vec::new()).max_forward_attempts,
            DEFAULT_MAX_FORWARD_ATTEMPTS
        );
    }
}
