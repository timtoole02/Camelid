//! Routing policy: which node receives a whole request.
//!
//! Selection has no I/O or external side effects: the fabric supplies snapshots
//! of observed load, reservations and learned service time, and this module
//! returns a decision. The service-time book performs no socket work either, so
//! both halves are tested against hand-built facts with no server or model.
//!
//! Two properties are load-bearing and each has a test that fails without it:
//!
//! * **Selection is deterministic.** Ties break on label, so an operator's dry
//!   run predicts what the fabric will actually do.
//! * **Affinity degrades, never fails.** A warm prefix is an optimisation; when
//!   the node holding it dies the request still gets served, and the decision
//!   records that affinity was lost rather than silently pretending it held.

use super::aliases::ModelAliases;
use super::capability::Capabilities;
use super::divergence::ModelIdentity;
use super::engine::NodeEngine;
use super::node::{NodeSnapshot, NodeStatus};

/// Successful completions required from every candidate before learned
/// service times may decide placement.
///
/// Until then the existing least-load rule explores the fabric without making
/// a routing claim from one unusually fast or slow request.
const MIN_SERVICE_TIME_SAMPLES: u32 = 5;

/// Weight of history in the steady-state EWMA. The new sample owns the fifth
/// part; integer arithmetic keeps the policy deterministic on every platform.
const EWMA_HISTORY_WEIGHT: u128 = 4;

/// A node not sampled inside this window is explored again before its old
/// speed may decide placement. This catches thermal, power and topology
/// changes without continuously diverting traffic from the estimated winner.
const MAX_SERVICE_TIME_AGE: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Dynamic node files and model switches must not grow resident policy state
/// without bound. This is far above a practical fabric's active class count.
const MAX_SERVICE_CLASSES: usize = 1_024;

/// How to choose among eligible nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteMode {
    /// Spread independent requests over the least-loaded nodes.
    Throughput,
    /// Minimise estimated time to finish from measured service time and load,
    /// falling back to [`RouteMode::Throughput`] while the estimate is cold.
    CompletionTime,
    /// Prefer the node that last served this session so its prompt prefix and KV
    /// cache stay warm, falling back to [`RouteMode::Throughput`] when it cannot.
    Affinity,
}

/// Whether this fabric will place work on an engine it cannot fully observe.
///
/// Refused by default. Turning it on does not make a foreign engine equal to a
/// Camelid one — it accepts three specific consequences, each handled
/// explicitly below: such a node publishes no load to rank on, cannot tell a
/// full queue from a failure, and cannot attest that a session's prefix is
/// still warm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MixedEngines {
    #[default]
    Refused,
    Allowed,
}

impl MixedEngines {
    /// The name health and the Routing screen report the mode under.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Refused => "refused",
            Self::Allowed => "allowed",
        }
    }
}

/// The command-line spelling that turns [`MixedEngines::Allowed`] on.
///
/// Published here, and in the proxy's health, so that anything telling an
/// operator which flag to add — a refusal message, the Routing screen — quotes
/// the build's own spelling rather than its own copy of it.
pub const MIXED_ENGINES_FLAG: &str = "--allow-mixed-engines";

/// What a node must be able to do before this request may land on it.
///
/// Expressed as capabilities rather than engine names on purpose: placement
/// has never known which engines exist, and a rule written against a name
/// would have to be revisited for every backend added after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Requirements {
    /// The request carries tools. A backend that has not been *measured* to
    /// handle them is not eligible, however confidently its vendor documents
    /// the feature.
    pub tool_calls: bool,
    /// The request is for a rerank route. Only a node whose API has one can
    /// take it; one without would answer 404 for a request another node
    /// could have served.
    pub rerank_route: bool,
}

impl Requirements {
    /// Everything either side requires. How a caller adds to what the body
    /// carries without being able to take any of it away.
    pub fn union(self, other: Self) -> Self {
        Self {
            tool_calls: self.tool_calls || other.tool_calls,
            rerank_route: self.rerank_route || other.rerank_route,
        }
    }
}

/// What the caller is asking the fabric to place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteRequest<'a> {
    pub mode: RouteMode,
    /// Required model id. `None` means any ready node will do.
    pub model: Option<&'a str>,
    /// Comparable workload class supplied by dispatch. Service times from
    /// unlike request sizes, routes or response shapes must not be mixed.
    pub service_class: Option<&'a str>,
    /// Label of the node that served this session previously, if any.
    pub sticky: Option<&'a str>,
    pub mixed: MixedEngines,
    pub requires: Requirements,
    /// What each node calls the requested model, where an operator said so.
    /// `None` resolves every id to itself.
    pub aliases: Option<&'a ModelAliases>,
}

impl<'a> RouteRequest<'a> {
    pub fn new(mode: RouteMode) -> Self {
        Self {
            mode,
            model: None,
            service_class: None,
            sticky: None,
            mixed: MixedEngines::Refused,
            requires: Requirements::default(),
            aliases: None,
        }
    }

    pub fn with_aliases(mut self, aliases: &'a ModelAliases) -> Self {
        self.aliases = Some(aliases);
        self
    }

    pub fn with_model(mut self, model: Option<&'a str>) -> Self {
        self.model = model;
        self
    }

    pub fn with_service_class(mut self, service_class: Option<&'a str>) -> Self {
        self.service_class = service_class;
        self
    }

    pub fn with_sticky(mut self, sticky: Option<&'a str>) -> Self {
        self.sticky = sticky;
        self
    }

    pub fn with_mixed_engines(mut self, mixed: MixedEngines) -> Self {
        self.mixed = mixed;
        self
    }

    pub fn requiring(mut self, requires: Requirements) -> Self {
        self.requires = requires;
        self
    }
}

/// Whether this fabric places on `snapshot` at all.
///
/// The first of three stages, each refused in its own words: is the node
/// admitted, does it hold the model, can it do what the request needs.
/// Folding them together sent operators to fix healthy machines.
fn admitted(snapshot: &NodeSnapshot, request: &RouteRequest<'_>) -> bool {
    // Mixed placement relaxes which engines are placed on, never whether the
    // node can take work now. With the two folded into one test, every down
    // or still-loading node became eligible the moment mixed placement was
    // asked for, and a request for a model against a dead fabric was refused
    // as permanently absent instead of as something that can clear.
    snapshot.is_placeable_under(request.mixed)
}

/// Whether a node whose engine can be asked `capabilities` can take a request
/// that requires `requires`, and if not, the capability it lacks.
///
/// Shared by placement and the one-shot send, so a request that names its node
/// directly is held to the same rule as one placed.
pub(crate) fn meets(
    capabilities: &Capabilities,
    requires: Requirements,
) -> Result<(), &'static str> {
    // `supported == Some(true)` is enough here, and deliberately so. The
    // capability table never credits a foreign engine with tool calls on
    // documentation alone -- it answers `not_probed` until somebody measures
    // that exact version -- so a `true` can only have come from a measurement
    // or from our own engine's test suite. Re-deriving that rule here would
    // mean placement knowing which engine is ours, and would have refused every
    // Camelid build not in the measurement table. The guarantee is pinned by
    // `capability::tests`.
    if requires.tool_calls && capabilities.tool_calls.supported != Some(true) {
        return Err("tool_calls");
    }
    if requires.rerank_route && capabilities.rerank_route.supported != Some(true) {
        return Err("rerank_route");
    }
    Ok(())
}

/// Whether the model a request lands on is held in memory, as far as the
/// node's own listing says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// The node's listing names it as loaded.
    Resident,
    /// The node's listing names what is loaded, and this is not among it: the
    /// request will load it first.
    NotResident,
    /// The node did not say what is loaded. Never read as resident.
    Unknown,
}

impl Residency {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::NotResident => "not_resident",
            Self::Unknown => "unknown",
        }
    }
}

/// Whether `snapshot` can serve `model`, and how warm it is. `None` when it
/// cannot. Pure.
///
/// A model held in memory is served. A model merely installed is served only
/// where the engine declares that naming one loads it: "installed" is not
/// "servable" on an engine that needs a person to load it first, and taking
/// it as such would place requests that come back refused.
pub(crate) fn serves(snapshot: &NodeSnapshot, model: &str) -> Option<Residency> {
    let ready = snapshot.status.ready()?;
    let resident = ready.active_model_id.as_deref() == Some(model)
        || ready
            .resident_models
            .as_ref()
            .is_some_and(|resident| resident.iter().any(|held| held == model));
    if resident {
        return Some(Residency::Resident);
    }
    let installed = ready.models.iter().any(|held| held == model);
    if installed && snapshot.capabilities().loads_on_demand.supported == Some(true) {
        return Some(match ready.resident_models {
            Some(_) => Residency::NotResident,
            None => Residency::Unknown,
        });
    }
    None
}

/// What `label` calls `model` under this request's aliases, and whether that
/// rests on a declaration.
fn local_id<'a>(
    request: &RouteRequest<'a>,
    label: &str,
    model: &'a str,
) -> (&'a str, ModelIdentity) {
    match request.aliases {
        Some(aliases) => aliases.resolve_with_identity(label, model),
        None => (model, ModelIdentity::SameId),
    }
}

/// What a node that publishes no load is charged when ranking.
///
/// Not a measurement, and not pretending to be one. Zero would make an
/// unobservable node beat every node that honestly reported work, so it would
/// win every placement and the fabric would concentrate load exactly where it
/// can least see it. A small positive cost makes it the choice only when the
/// observable nodes are actually busier than this.
pub const UNREPORTED_LOAD_COST: usize = 2;

/// What a holder that must load the model first is charged when ranking, on
/// top of its load.
///
/// A stated guess, like [`UNREPORTED_LOAD_COST`], not a measurement: nobody
/// here has timed a cold load, which on a small machine takes seconds and may
/// evict another model. It is a price rather than a rule that the warm holder
/// always wins, because a rule would send every request for a model to the
/// one node holding it warm — a node that, if foreign, publishes no load and
/// sends no typed refusal — and never to an idle sibling that could load it.
/// Once the warm holder carries more than this many requests beyond its
/// sibling, the sibling is chosen. Unknown residency is priced as cold.
pub const COLD_LOAD_COST: usize = 4;

/// How the winning node was chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteReason {
    /// The session's previous node is still eligible.
    Affinity,
    /// Exactly one node was eligible.
    OnlyCandidate,
    /// Fewest queued jobs among the eligible nodes.
    LeastLoaded,
    /// Lowest predicted time until this request finishes.
    EstimatedCompletion,
}

impl RouteReason {
    /// The name clients see, in `fabric … --json` and in the proxy's
    /// `x-camelid-fabric-reason` header.
    ///
    /// Pinned here, and asserted in a test, because these strings used to come
    /// from `#[derive(Debug)]` — which made renaming a variant a silent change
    /// to two public surfaces.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Affinity => "Affinity",
            Self::OnlyCandidate => "OnlyCandidate",
            Self::LeastLoaded => "LeastLoaded",
            Self::EstimatedCompletion => "EstimatedCompletion",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    pub label: String,
    /// The engine on the chosen node. Reported on every answer so a client never
    /// has to know which engines this fabric is willing to place on to find out
    /// which one served it.
    pub engine: NodeEngine,
    pub reason: RouteReason,
    /// Previous node's label when affinity was requested but could not be
    /// honoured. Reported so a caller can tell a warm hit from a cold re-prefill.
    pub affinity_lost: Option<String>,
    /// The model id to send the chosen node, which is the only thing about a
    /// request placement ever writes. The requested id unless an alias names
    /// the node's own; under mixed placement, the node's single active model
    /// when the request named none. `None` leaves the body as it came.
    pub model: Option<String>,
    /// What `model` rests on when the request named one: the same id, or an
    /// operator's word that a different id is the same weights.
    pub model_identity: Option<ModelIdentity>,
    /// How warm the chosen node's listing says the model is. `None` when no
    /// model was matched, which is placement making no claim either way.
    pub residency: Option<Residency>,
}

/// Why no node could take the request. Every variant carries enough detail to
/// act on without re-probing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    NoNodesConfigured,
    AllNodesUnavailable {
        unreachable: usize,
        not_ready: usize,
        /// Nodes that answered and are healthy, but run an engine this fabric
        /// does not place on. Counted apart from `not_ready` because nothing an
        /// operator does to those machines will make them eligible.
        not_placeable: usize,
    },
    ModelUnavailable {
        model: String,
        /// What the ready nodes are actually serving.
        serving: Vec<String>,
        /// Configured nodes that could not be consulted, because they were
        /// unreachable or reachable but not ready.
        ///
        /// This is what separates "the fabric does not serve that model" from
        /// "the fabric cannot say yet": while any node is unaccounted for, one
        /// of them may be the node that owns the model.
        unobserved: usize,
        /// Healthy nodes holding the model whose engine this fabric does not
        /// place on. They were consulted, so they are not `unobserved`: the
        /// refusal is settled, and what changes it is the operator's flag, not
        /// time.
        held_by_unplaced: Vec<String>,
    },
    /// Nodes hold the model, and none of them can do what the request needs.
    RequirementUnmet {
        /// The capability key: `tool_calls` or `rerank_route`.
        requirement: &'static str,
        model: Option<String>,
        /// Each holder that lacks it, with its engine and reported version.
        unmet: Vec<(String, String)>,
        /// Nodes that meet it but do not hold the model.
        capable_elsewhere: Vec<String>,
        unobserved: usize,
    },
    /// Under mixed placement a request that names no model is placed only
    /// where the model it will get is known, and no such node is ready.
    ModelUnspecified {
        unobserved: usize,
    },
    /// The session asked to go back to a node that cannot say whether its
    /// prefix is still warm. Refused rather than served there anyway, because
    /// honouring affinity that nothing attests is just a slower random choice
    /// wearing the word "affinity".
    AffinityUnsupported {
        label: String,
    },
}

fn unobserved_clause(unobserved: usize) -> String {
    match unobserved {
        0 => String::new(),
        1 => "; 1 node could not be consulted, so this may change".to_string(),
        many => format!("; {many} nodes could not be consulted, so this may change"),
    }
}

fn requirement_lead(requirement: &str) -> &'static str {
    match requirement {
        "tool_calls" => "a request carrying tools is placed only on a node measured to handle them",
        _ => "a rerank request is placed only on a node that exposes a rerank route",
    }
}

impl RouteError {
    /// The refusal as a listener reachable from off the machine may say it.
    ///
    /// Node labels and exact engine versions identify what is running where,
    /// which the proxy's health already withholds off-box; a refusal must not
    /// hand them out instead. Refusals that name no such thing read exactly as
    /// their [`std::fmt::Display`].
    pub fn public_message(&self) -> String {
        match self {
            Self::ModelUnavailable {
                model,
                serving,
                unobserved,
                ..
            } => Self::ModelUnavailable {
                model: model.clone(),
                serving: serving.clone(),
                unobserved: *unobserved,
                held_by_unplaced: Vec::new(),
            }
            .to_string(),
            Self::RequirementUnmet {
                requirement,
                unmet,
                unobserved,
                ..
            } => {
                let mut message = requirement_lead(requirement).to_string();
                match (*requirement, unmet.len()) {
                    ("tool_calls", 1) => message.push_str(
                        "; 1 node holding this model has not been measured handling tool calls",
                    ),
                    ("tool_calls", many) => message.push_str(&format!(
                        "; {many} nodes holding this model have not been measured handling tool calls"
                    )),
                    _ => message.push_str("; no node holding this model exposes one"),
                }
                message.push_str(&unobserved_clause(*unobserved));
                message
            }
            other => other.to_string(),
        }
    }
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoNodesConfigured => write!(f, "no nodes configured"),
            Self::AllNodesUnavailable {
                unreachable,
                not_ready,
                not_placeable,
            } => {
                write!(
                    f,
                    "no node can serve: {unreachable} unreachable, {not_ready} reachable but not ready"
                )?;
                match not_placeable {
                    0 => Ok(()),
                    1 => write!(
                        f,
                        ", 1 healthy node running an engine this fabric does not place on"
                    ),
                    many => write!(
                        f,
                        ", {many} healthy nodes running engines this fabric does not place on"
                    ),
                }
            }
            Self::ModelUnavailable {
                model,
                serving,
                unobserved,
                held_by_unplaced,
            } => {
                write!(f, "no ready node is serving model `{model}`")?;
                if !serving.is_empty() {
                    write!(f, "; ready nodes serve: {}", serving.join(", "))?;
                }
                match held_by_unplaced.as_slice() {
                    [] => {}
                    [one] => write!(
                        f,
                        "; {one} holds it but runs an engine this fabric does not place on \
                         (accept that with {MIXED_ENGINES_FLAG})"
                    )?,
                    many => write!(
                        f,
                        "; {} hold it but run engines this fabric does not place on \
                         (accept that with {MIXED_ENGINES_FLAG})",
                        many.join(", ")
                    )?,
                }
                f.write_str(&unobserved_clause(*unobserved))
            }
            Self::RequirementUnmet {
                requirement,
                model,
                unmet,
                capable_elsewhere,
                unobserved,
            } => {
                write!(f, "{}", requirement_lead(requirement))?;
                let holders = unmet
                    .iter()
                    .map(|(label, engine)| format!("{label} ({engine})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let many = unmet.len() > 1;
                if *requirement == "tool_calls" {
                    write!(
                        f,
                        "; {holders} {} not been measured handling tool calls here",
                        if many { "have" } else { "has" }
                    )?;
                } else {
                    match model {
                        Some(model) => {
                            write!(f, "; no node holding `{model}` exposes a rerank route")?
                        }
                        None => write!(f, "; no node that would take it exposes a rerank route")?,
                    }
                    write!(f, ": {holders}")?;
                }
                if let (Some(model), false) = (model, capable_elsewhere.is_empty()) {
                    let what = match (*requirement, capable_elsewhere.len() > 1) {
                        ("tool_calls", _) => "can take tools",
                        (_, true) => "expose one",
                        (_, false) => "exposes one",
                    };
                    write!(
                        f,
                        "; {} {what} but {} not hold `{model}`",
                        capable_elsewhere.join(", "),
                        if capable_elsewhere.len() > 1 {
                            "do"
                        } else {
                            "does"
                        }
                    )?;
                }
                f.write_str(&unobserved_clause(*unobserved))
            }
            Self::ModelUnspecified { unobserved } => {
                write!(
                    f,
                    "this request names no model, and no ready node this fabric places on serves \
                     exactly one, so which model it would get is not known; name the model"
                )?;
                f.write_str(&unobserved_clause(*unobserved))
            }
            Self::AffinityUnsupported { label } => write!(
                f,
                "{label} cannot attest that a session's prefix is still warm, so affinity to it \
                 is refused rather than pretended; retry without a sticky node to be placed \
                 normally"
            ),
        }
    }
}

impl std::error::Error for RouteError {}

/// Requests this fabric has placed on each node and not yet finished.
///
/// `/v1/health` reports what a node has *accepted*. A request already on its way
/// to a node is invisible there until it arrives, so without this a burst of
/// concurrent requests all observe the same idle fabric, and the label
/// tie-break sends every one of them to the same node.
///
/// Pure and self-contained: the arithmetic is tested here, without threads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reservations(std::collections::BTreeMap<String, usize>);

impl Reservations {
    /// What a dry run holds: nothing is in flight, so nothing is reserved.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn get(&self, label: &str) -> usize {
        self.0.get(label).copied().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Record one more request placed on `label`.
    pub fn take(&mut self, label: &str) {
        *self.0.entry(label.to_string()).or_insert(0) += 1;
    }

    /// Record that one request on `label` has finished.
    ///
    /// A label that reaches zero is removed rather than left at zero, so the map
    /// stays a statement of what is actually outstanding.
    pub fn release(&mut self, label: &str) {
        if let Some(count) = self.0.get_mut(label) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.0.remove(label);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ServiceClass {
    label: String,
    authority: String,
    /// Both stay optional so that a node which stops reporting a lane or a
    /// version invalidates its estimates rather than silently inheriting the
    /// timings of the lane it used to run.
    backend: Option<String>,
    version: Option<String>,
    model: String,
    workload: String,
}

impl ServiceClass {
    fn of(node: &NodeSnapshot, model: &str, workload: &str) -> Option<Self> {
        let ready = node.status.ready()?;
        Some(Self {
            label: node.label().to_string(),
            authority: node.spec.authority(),
            backend: ready.backend.clone(),
            version: ready.version.clone(),
            model: model.to_string(),
            workload: workload.to_string(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ServiceTimeEstimate {
    mean_nanos: u128,
    samples: u32,
    last_observed: Option<std::time::Instant>,
    last_selected: Option<std::time::Instant>,
}

/// Service time learned by this resident fabric, scoped to one node identity,
/// model and comparable workload class.
///
/// Only successful completed requests belong here. The dispatch layer owns
/// that distinction; this type only maintains deterministic estimates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ServiceTimeEstimates(
    std::collections::BTreeMap<ServiceClass, ServiceTimeEstimate>,
);

impl ServiceTimeEstimates {
    pub(crate) fn selected(&mut self, node: &NodeSnapshot, model: &str, workload: &str) {
        self.selected_at(node, model, workload, std::time::Instant::now());
    }

    fn selected_at(
        &mut self,
        node: &NodeSnapshot,
        model: &str,
        workload: &str,
        now: std::time::Instant,
    ) {
        let Some(class) = ServiceClass::of(node, model, workload) else {
            return;
        };
        self.0
            .entry(class)
            .and_modify(|estimate| estimate.last_selected = Some(now))
            .or_insert(ServiceTimeEstimate {
                mean_nanos: 0,
                samples: 0,
                last_observed: None,
                last_selected: Some(now),
            });
        self.prune();
    }

    pub(crate) fn observe(
        &mut self,
        node: &NodeSnapshot,
        model: &str,
        workload: &str,
        elapsed: std::time::Duration,
    ) {
        self.observe_at(node, model, workload, elapsed, std::time::Instant::now());
    }

    pub(crate) fn invalidate(&mut self, node: &NodeSnapshot, model: &str, workload: &str) {
        let Some(class) = ServiceClass::of(node, model, workload) else {
            return;
        };
        let now = std::time::Instant::now();
        self.0
            .entry(class)
            .and_modify(|estimate| {
                estimate.mean_nanos = 0;
                estimate.samples = 0;
                estimate.last_observed = None;
                estimate.last_selected = Some(now);
            })
            .or_insert(ServiceTimeEstimate {
                mean_nanos: 0,
                samples: 0,
                last_observed: None,
                last_selected: Some(now),
            });
        self.prune();
    }

    fn observe_at(
        &mut self,
        node: &NodeSnapshot,
        model: &str,
        workload: &str,
        elapsed: std::time::Duration,
        now: std::time::Instant,
    ) {
        let Some(class) = ServiceClass::of(node, model, workload) else {
            return;
        };
        let sample = elapsed.as_nanos().max(1);
        self.0
            .entry(class)
            .and_modify(|estimate| {
                let stale = estimate.last_observed.is_some_and(|observed| {
                    now.saturating_duration_since(observed) > MAX_SERVICE_TIME_AGE
                });
                if stale {
                    estimate.mean_nanos = sample;
                    estimate.samples = 0;
                } else if estimate.samples < MIN_SERVICE_TIME_SAMPLES {
                    let samples = u128::from(estimate.samples);
                    estimate.mean_nanos = estimate
                        .mean_nanos
                        .saturating_mul(samples)
                        .saturating_add(sample)
                        / (samples + 1);
                } else {
                    estimate.mean_nanos = estimate
                        .mean_nanos
                        .saturating_mul(EWMA_HISTORY_WEIGHT)
                        .saturating_add(sample)
                        / (EWMA_HISTORY_WEIGHT + 1);
                }
                estimate.samples = estimate.samples.saturating_add(1);
                estimate.last_observed = Some(now);
            })
            .or_insert(ServiceTimeEstimate {
                mean_nanos: sample,
                samples: 1,
                last_observed: Some(now),
                last_selected: None,
            });

        self.prune();
    }

    fn prune(&mut self) {
        if self.0.len() > MAX_SERVICE_CLASSES {
            let oldest = self
                .0
                .iter()
                .min_by_key(|(_, estimate)| estimate.last_observed.max(estimate.last_selected))
                .map(|(class, _)| class.clone())
                .expect("an oversized estimate map is non-empty");
            self.0.remove(&oldest);
        }
    }

    fn get(
        &self,
        node: &NodeSnapshot,
        model: Option<&str>,
        workload: Option<&str>,
    ) -> Option<ServiceTimeEstimate> {
        let model = model.or_else(|| node.active_model_id())?;
        let class = ServiceClass::of(node, model, workload?)?;
        self.0.get(&class).copied()
    }

    #[cfg(test)]
    pub(crate) fn sample_for(
        &self,
        node: &NodeSnapshot,
        model: &str,
        workload: &str,
    ) -> Option<(u128, u32)> {
        self.get(node, Some(model), Some(workload))
            .map(|estimate| (estimate.mean_nanos, estimate.samples))
    }
}

/// One eligible node reduced to the fields selection actually uses.
struct Candidate<'a> {
    label: &'a str,
    engine: NodeEngine,
    load: usize,
    service_nanos: Option<u128>,
    last_selected: Option<std::time::Instant>,
    model: Option<String>,
    identity: Option<ModelIdentity>,
    residency: Option<Residency>,
}

/// A ready node that can serve this request's model, with what to send it.
struct Holder<'a> {
    snapshot: &'a NodeSnapshot,
    model: Option<String>,
    identity: Option<ModelIdentity>,
    residency: Option<Residency>,
}

/// What a node is carrying, as far as this fabric can tell. Pure.
///
/// A request this fabric placed is either not yet accepted by the node — and so
/// missing from `observed` — or already accepted and counted in it. The truth is
/// therefore somewhere in `[max(observed, reserved), observed + reserved]`, and
/// `max` is the tighter end.
///
/// Adding them instead would double-count for most of a request's life, and
/// would do so asymmetrically: a node busy with this fabric's work would be
/// ranked worse than a node equally busy with somebody else's.
pub(crate) fn load_of(observed: usize, reserved: usize) -> usize {
    observed.max(reserved)
}

/// Choose a node for one request, ignoring anything already in flight.
///
/// This is what a dry run does, and what `fabric route` reports: given only what
/// the nodes say about themselves, which one would take the request.
pub fn route(
    snapshots: &[NodeSnapshot],
    request: &RouteRequest<'_>,
) -> Result<RouteDecision, RouteError> {
    route_reserved(snapshots, request, &Reservations::none())
}

/// Choose a node for one request, counting what this fabric already placed.
///
/// Still pure: `reserved` is a snapshot of the fabric's own bookkeeping, so the
/// decision remains a function of its arguments and a dry run with an empty
/// `reserved` reproduces [`route`] exactly.
pub fn route_reserved(
    snapshots: &[NodeSnapshot],
    request: &RouteRequest<'_>,
    reserved: &Reservations,
) -> Result<RouteDecision, RouteError> {
    route_reserved_with_estimates(
        snapshots,
        request,
        reserved,
        &ServiceTimeEstimates::default(),
    )
}

pub(crate) fn route_reserved_with_estimates(
    snapshots: &[NodeSnapshot],
    request: &RouteRequest<'_>,
    reserved: &Reservations,
    estimates: &ServiceTimeEstimates,
) -> Result<RouteDecision, RouteError> {
    route_reserved_with_estimates_at(
        snapshots,
        request,
        reserved,
        estimates,
        std::time::Instant::now(),
    )
}

fn route_reserved_with_estimates_at(
    snapshots: &[NodeSnapshot],
    request: &RouteRequest<'_>,
    reserved: &Reservations,
    estimates: &ServiceTimeEstimates,
    now: std::time::Instant,
) -> Result<RouteDecision, RouteError> {
    if snapshots.is_empty() {
        return Err(RouteError::NoNodesConfigured);
    }

    let admitted_nodes: Vec<&NodeSnapshot> = snapshots
        .iter()
        .filter(|snapshot| admitted(snapshot, request))
        .collect();

    if admitted_nodes.is_empty() {
        // A request that names a model only an unplaced engine holds is
        // refused as that, settled and naming the node: what changes it is
        // the operator's flag, and "no node can serve" would say nothing
        // about which node holds it or why it was not used.
        if let Some(model) = request.model {
            let mut held_by_unplaced: Vec<String> = snapshots
                .iter()
                .filter(|s| s.status.is_ready())
                .filter(|s| serves(s, local_id(request, s.label(), model).0).is_some())
                .map(|s| s.label().to_string())
                .collect();
            if !held_by_unplaced.is_empty() {
                held_by_unplaced.sort();
                return Err(RouteError::ModelUnavailable {
                    model: model.to_string(),
                    serving: Vec::new(),
                    unobserved: snapshots.iter().filter(|s| !s.status.is_ready()).count(),
                    held_by_unplaced,
                });
            }
        }
        let unreachable = snapshots
            .iter()
            .filter(|s| matches!(s.status, NodeStatus::Unreachable { .. }))
            .count();
        // A healthy node running an unplaceable engine is neither unreachable
        // nor not-ready, and folding it into either would send an operator to
        // fix a machine that is working perfectly.
        let not_placeable = snapshots
            .iter()
            .filter(|s| s.status.is_ready() && !admitted(s, request))
            .count();
        return Err(RouteError::AllNodesUnavailable {
            unreachable,
            not_ready: snapshots.len() - unreachable - not_placeable,
            not_placeable,
        });
    }

    // Only a node that answered as ready was asked what it holds. A ready node
    // this fabric does not place on still answered, so it is not unaccounted
    // for: counting it would turn a refusal only a flag can clear into one a
    // client is told to retry.
    let unobserved = snapshots.iter().filter(|s| !s.status.is_ready()).count();

    let holders: Vec<Holder<'_>> = match request.model {
        Some(model) => {
            let matched: Vec<Holder<'_>> = admitted_nodes
                .iter()
                .filter_map(|snapshot| {
                    let (local, identity) = local_id(request, snapshot.label(), model);
                    serves(snapshot, local).map(|residency| Holder {
                        snapshot,
                        model: Some(local.to_string()),
                        identity: Some(identity),
                        residency: Some(residency),
                    })
                })
                .collect();
            if matched.is_empty() {
                let mut serving: Vec<String> = admitted_nodes
                    .iter()
                    .filter_map(|s| s.active_model_id().map(str::to_string))
                    .collect();
                serving.sort();
                serving.dedup();
                let mut held_by_unplaced: Vec<String> = snapshots
                    .iter()
                    .filter(|s| s.status.is_ready() && !admitted(s, request))
                    .filter(|s| serves(s, local_id(request, s.label(), model).0).is_some())
                    .map(|s| s.label().to_string())
                    .collect();
                held_by_unplaced.sort();
                return Err(RouteError::ModelUnavailable {
                    model: model.to_string(),
                    serving,
                    unobserved,
                    held_by_unplaced,
                });
            }
            matched
        }
        // Our own engine serves whatever it loaded to a request naming no
        // model; another engine answers that it needs one. So under mixed
        // placement such a request goes only where the model it will get is
        // known, and that id is what is sent.
        None if request.mixed == MixedEngines::Allowed => {
            let matched: Vec<Holder<'_>> = admitted_nodes
                .iter()
                .filter_map(|snapshot| {
                    snapshot.active_model_id().map(|active| Holder {
                        snapshot,
                        model: Some(active.to_string()),
                        identity: None,
                        residency: Some(Residency::Resident),
                    })
                })
                .collect();
            if matched.is_empty() {
                return Err(RouteError::ModelUnspecified { unobserved });
            }
            matched
        }
        None => admitted_nodes
            .iter()
            .map(|snapshot| Holder {
                snapshot,
                model: None,
                identity: None,
                residency: None,
            })
            .collect(),
    };

    // Asked of every engine, ours included. A holder that cannot do what the
    // request needs is refused for exactly that, never as an unplaceable
    // engine: the operator's next step is a different node or a measurement,
    // not a restart.
    let capable: Vec<&Holder<'_>> = holders
        .iter()
        .filter(|holder| meets(&holder.snapshot.capabilities(), request.requires).is_ok())
        .collect();
    if capable.is_empty() {
        let requirement = holders
            .iter()
            .find_map(|holder| meets(&holder.snapshot.capabilities(), request.requires).err())
            .unwrap_or("tool_calls");
        let mut unmet: Vec<(String, String)> = holders
            .iter()
            .map(|holder| {
                (
                    holder.snapshot.label().to_string(),
                    holder.snapshot.engine_and_version(),
                )
            })
            .collect();
        unmet.sort();
        let mut capable_elsewhere: Vec<String> = admitted_nodes
            .iter()
            .filter(|s| meets(&s.capabilities(), request.requires).is_ok())
            .filter(|s| {
                !holders
                    .iter()
                    .any(|holder| holder.snapshot.label() == s.label())
            })
            .map(|s| s.label().to_string())
            .collect();
        capable_elsewhere.sort();
        return Err(RouteError::RequirementUnmet {
            requirement,
            model: request.model.map(str::to_string),
            unmet,
            capable_elsewhere,
            unobserved,
        });
    }

    // Every ready node serving the model is eligible. The fabric cannot tell
    // whether a node is at its bound — `/v1/health` publishes load, not capacity
    // — so it ranks by load and lets a genuinely full node answer its own 503.
    let candidates: Vec<Candidate<'_>> = capable
        .iter()
        .filter_map(|holder| {
            let snapshot = holder.snapshot;
            snapshot.status.ready().map(|ready| {
                let estimate = estimates.get(snapshot, request.model, request.service_class);
                let current = estimate.filter(|estimate| {
                    estimate.last_observed.is_some_and(|observed| {
                        now.saturating_duration_since(observed) <= MAX_SERVICE_TIME_AGE
                    })
                });
                // A node that publishes no load is not an idle one. Left at
                // the reservation count it would outrank every node that
                // honestly reported work and win every placement, so it
                // carries a stated fixed cost instead. This is a deliberate
                // guess about an unobservable node, which is why mixed
                // placement is off by default.
                let carrying = match ready.in_flight() {
                    Some(observed) => load_of(observed, reserved.get(snapshot.label())),
                    None => UNREPORTED_LOAD_COST.saturating_add(reserved.get(snapshot.label())),
                };
                let loading = match holder.residency {
                    Some(Residency::Resident) | None => 0,
                    Some(Residency::NotResident | Residency::Unknown) => COLD_LOAD_COST,
                };
                Candidate {
                    label: snapshot.label(),
                    engine: snapshot.engine(),
                    load: carrying.saturating_add(loading),
                    service_nanos: current
                        .filter(|estimate| estimate.samples >= MIN_SERVICE_TIME_SAMPLES)
                        .map(|estimate| estimate.mean_nanos),
                    last_selected: estimate.and_then(|estimate| estimate.last_selected),
                    model: holder.model.clone(),
                    identity: holder.identity,
                    residency: holder.residency,
                }
            })
        })
        .collect();

    if candidates.is_empty() {
        return Err(RouteError::AllNodesUnavailable {
            unreachable: 0,
            not_ready: capable.len(),
            not_placeable: 0,
        });
    }

    if request.mode == RouteMode::Affinity {
        if let Some(sticky) = request.sticky {
            // Affinity is only worth honouring if the node can say the prefix
            // is still warm. A node that cannot is refused rather than
            // silently treated as a cache hit it never claimed.
            if let Some(hit) = candidates.iter().find(|c| c.label == sticky) {
                let attests = capable
                    .iter()
                    .find(|holder| holder.snapshot.label() == sticky)
                    .is_some_and(|holder| {
                        holder.snapshot.capabilities().warm_prefix.supported == Some(true)
                    });
                if !attests {
                    return Err(RouteError::AffinityUnsupported {
                        label: sticky.to_string(),
                    });
                }
                return Ok(RouteDecision {
                    label: hit.label.to_string(),
                    engine: hit.engine,
                    reason: RouteReason::Affinity,
                    affinity_lost: None,
                    model: hit.model.clone(),
                    model_identity: hit.identity,
                    residency: hit.residency,
                });
            }
        }
    }

    let has_complete_estimates = request.mode == RouteMode::CompletionTime
        && candidates
            .iter()
            .all(|candidate| candidate.service_nanos.is_some());

    // Ties break on label so a dry run predicts the real decision. Completion
    // scores use u128 and saturating multiplication: neither a pathological
    // duration nor load may wrap around and make a node look fast.
    let chosen = if has_complete_estimates {
        candidates
            .iter()
            .min_by(|a, b| {
                let score = |candidate: &Candidate<'_>| {
                    candidate
                        .service_nanos
                        .expect("all estimates are complete")
                        .saturating_mul((candidate.load as u128).saturating_add(1))
                };
                score(a)
                    .cmp(&score(b))
                    .then_with(|| a.load.cmp(&b.load))
                    .then_with(|| a.label.cmp(b.label))
            })
            .expect("candidates is non-empty")
    } else {
        candidates
            .iter()
            .min_by(|a, b| {
                a.load
                    .cmp(&b.load)
                    .then_with(|| {
                        if request.mode == RouteMode::CompletionTime {
                            a.last_selected.cmp(&b.last_selected)
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .then_with(|| a.label.cmp(b.label))
            })
            .expect("candidates is non-empty")
    };

    let reason = if candidates.len() == 1 {
        RouteReason::OnlyCandidate
    } else if has_complete_estimates {
        RouteReason::EstimatedCompletion
    } else {
        RouteReason::LeastLoaded
    };

    let affinity_lost = request
        .sticky
        .filter(|sticky| request.mode == RouteMode::Affinity && *sticky != chosen.label)
        .map(str::to_string);

    Ok(RouteDecision {
        label: chosen.label.to_string(),
        engine: chosen.engine,
        reason,
        affinity_lost,
        model: chosen.model.clone(),
        model_identity: chosen.identity,
        residency: chosen.residency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::engine::NodeEngine;
    use crate::fabric::node::{NodeLoad, NodeReady, NodeSpec, NodeStatus};

    fn spec(label: &str) -> NodeSpec {
        NodeSpec::camelid(label, "127.0.0.1", 8181)
    }

    fn ready(label: &str, model: Option<&str>, in_flight: usize) -> NodeSnapshot {
        NodeSnapshot {
            spec: spec(label),
            status: NodeStatus::Ready(NodeReady {
                engine: NodeEngine::Camelid,
                active_model_id: model.map(str::to_string),
                models: model.map(str::to_string).into_iter().collect(),
                resident_models: model.map(|model| vec![model.to_string()]),
                backend: Some("llama".to_string()),
                version: Some("0.5.4".to_string()),
                load: Some(NodeLoad {
                    in_flight,
                    waiting: in_flight.saturating_sub(1),
                }),
            }),
            latency: None,
        }
    }

    /// A healthy foreign node: serving, and publishing no load, because that
    /// is what the engines this fabric reads actually do.
    fn foreign(label: &str, engine: NodeEngine, model: &str) -> NodeSnapshot {
        NodeSnapshot {
            spec: NodeSpec {
                label: label.to_string(),
                host: "127.0.0.1".to_string(),
                port: engine.default_port(),
                engine,
            },
            status: NodeStatus::Ready(NodeReady {
                engine,
                active_model_id: Some(model.to_string()),
                models: vec![model.to_string()],
                resident_models: Some(vec![model.to_string()]),
                backend: None,
                version: None,
                load: None,
            }),
            latency: None,
        }
    }

    const MIXED: MixedEngines = MixedEngines::Allowed;

    #[test]
    fn a_foreign_node_is_refused_by_default_and_eligible_only_when_mixed_is_asked_for() {
        let nodes = [foreign("studio", NodeEngine::Ollama, "m")];

        let refused = route(&nodes, &RouteRequest::new(RouteMode::Throughput))
            .expect_err("default routing places on Camelid only");
        assert_eq!(
            refused,
            RouteError::AllNodesUnavailable {
                unreachable: 0,
                not_ready: 0,
                not_placeable: 1
            }
        );

        let placed = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput).with_mixed_engines(MIXED),
        )
        .expect("mixed placement was asked for");
        assert_eq!(placed.label, "studio");
        assert_eq!(placed.engine, NodeEngine::Ollama);
    }

    #[test]
    fn a_node_that_publishes_no_load_does_not_outrank_one_that_reported_being_busy() {
        // The guard that matters most: charged nothing, an unobservable node
        // looks permanently idle, wins every placement, and the fabric piles
        // work exactly where it can see least.
        let nodes = [
            ready("win", Some("m"), 1),
            foreign("studio", NodeEngine::Ollama, "m"),
        ];
        let decision = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .with_mixed_engines(MIXED),
        )
        .expect("both are eligible");
        assert_eq!(
            decision.label, "win",
            "one job in flight is better evidence than no evidence at all"
        );
    }

    #[test]
    fn a_node_that_publishes_no_load_still_wins_when_the_observable_nodes_are_busier() {
        // The other half: the fixed cost is a price, not a ban.
        let nodes = [
            ready("win", Some("m"), UNREPORTED_LOAD_COST + 1),
            foreign("studio", NodeEngine::Ollama, "m"),
        ];
        let decision = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .with_mixed_engines(MIXED),
        )
        .expect("both are eligible");
        assert_eq!(decision.label, "studio");
    }

    #[test]
    fn a_tool_calling_request_never_lands_on_a_backend_nobody_measured() {
        let nodes = [foreign("studio", NodeEngine::Ollama, "m")];
        let needs_tools = RouteRequest::new(RouteMode::Throughput)
            .with_mixed_engines(MIXED)
            .requiring(Requirements {
                tool_calls: true,
                ..Requirements::default()
            });

        assert!(
            route(&nodes, &needs_tools).is_err(),
            "an unmeasured backend is not eligible for tools however well documented"
        );
        assert!(
            route(
                &nodes,
                &RouteRequest::new(RouteMode::Throughput).with_mixed_engines(MIXED)
            )
            .is_ok(),
            "and the same node still takes ordinary work"
        );
    }

    #[test]
    fn our_own_engine_is_eligible_for_tools_and_a_camelid_only_fabric_is_unaffected() {
        let nodes = [ready("win", Some("m"), 0)];
        let decision = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .requiring(Requirements {
                    tool_calls: true,
                    ..Requirements::default()
                }),
        )
        .expect("our own engine's tool contract is covered by its test suite");
        assert_eq!(decision.label, "win");
    }

    #[test]
    fn affinity_to_a_node_that_cannot_attest_a_warm_prefix_is_refused_not_degraded() {
        // Placing it there anyway would be a slower random choice wearing the
        // word "affinity", and the caller would never learn the difference.
        let nodes = [
            foreign("studio", NodeEngine::Ollama, "m"),
            ready("win", Some("m"), 0),
        ];
        let error = route(
            &nodes,
            &RouteRequest::new(RouteMode::Affinity)
                .with_model(Some("m"))
                .with_sticky(Some("studio"))
                .with_mixed_engines(MIXED),
        )
        .expect_err("affinity to an unattesting node is refused");

        assert_eq!(
            error,
            RouteError::AffinityUnsupported {
                label: "studio".to_string()
            }
        );
        let message = error.to_string();
        assert!(message.contains("retry without a sticky node"), "{message}");
    }

    #[test]
    fn affinity_to_our_own_engine_is_unchanged_by_any_of_this() {
        let nodes = [ready("warm", Some("m"), 3), ready("idle", Some("m"), 0)];
        let decision = route(
            &nodes,
            &RouteRequest::new(RouteMode::Affinity)
                .with_model(Some("m"))
                .with_sticky(Some("warm")),
        )
        .expect("a Camelid node attests its own warmth");
        assert_eq!(decision.label, "warm");
        assert_eq!(decision.reason, RouteReason::Affinity);
    }

    fn unreachable(label: &str) -> NodeSnapshot {
        NodeSnapshot {
            spec: spec(label),
            status: NodeStatus::Unreachable {
                reason: "connection refused".to_string(),
            },
            latency: None,
        }
    }

    fn not_ready(label: &str) -> NodeSnapshot {
        NodeSnapshot {
            spec: spec(label),
            status: NodeStatus::NotReady {
                reason: "no model loaded".to_string(),
            },
            latency: None,
        }
    }

    /// The flag accepts an engine this fabric cannot fully observe; it does
    /// not accept a node that cannot answer. Folded together, a model request
    /// against a dead fabric came back as a permanent `ModelUnavailable` —
    /// a 404 a client is told never to retry — instead of the refusal that
    /// can clear.
    #[test]
    fn mixed_placement_refuses_an_unreachable_node_exactly_as_camelid_only_placement_does() {
        let nodes = [unreachable("gone")];
        let camelid_only = RouteRequest::new(RouteMode::Throughput).with_model(Some("m"));
        let mixed = camelid_only.with_mixed_engines(MIXED);

        let expected = route(&nodes, &camelid_only).expect_err("nothing answered");
        assert!(
            matches!(
                expected,
                RouteError::AllNodesUnavailable { unreachable: 1, .. }
            ),
            "{expected:?}"
        );
        assert_eq!(route(&nodes, &mixed), Err(expected));
    }

    #[test]
    fn mixed_placement_does_not_make_a_node_that_is_still_loading_eligible() {
        let warming = not_ready("warming");
        let mixed = RouteRequest::new(RouteMode::Throughput).with_mixed_engines(MIXED);
        assert!(!admitted(&warming, &mixed));

        // And what that changes on the wire: while the loading node is
        // unaccounted for it may be the one that owns the model, so the
        // refusal has to stay the retryable one.
        let nodes = [warming, ready("win", Some("other"), 0)];
        let camelid_only = RouteRequest::new(RouteMode::Throughput).with_model(Some("m"));
        let refused = route(&nodes, &camelid_only.with_mixed_engines(MIXED))
            .expect_err("nobody ready serves m");
        assert_eq!(refused, route(&nodes, &camelid_only).expect_err("same"));
        assert!(
            matches!(refused, RouteError::ModelUnavailable { unobserved: 1, .. }),
            "{refused:?}"
        );
    }

    #[test]
    fn an_empty_fabric_is_refused_distinctly_from_a_dead_one() {
        assert_eq!(
            route(&[], &RouteRequest::new(RouteMode::Throughput)),
            Err(RouteError::NoNodesConfigured)
        );
    }

    #[test]
    fn reserving_and_releasing_returns_to_nothing_outstanding() {
        let mut reserved = Reservations::none();
        assert_eq!(reserved.get("a"), 0);
        reserved.take("a");
        reserved.take("a");
        assert_eq!(reserved.get("a"), 2);
        reserved.release("a");
        assert_eq!(reserved.get("a"), 1);
        reserved.release("a");
        assert_eq!(reserved.get("a"), 0);
        // Back to genuinely empty, not a lingering zero.
        assert!(reserved.is_empty());
    }

    #[test]
    fn releasing_more_than_was_taken_cannot_underflow() {
        let mut reserved = Reservations::none();
        reserved.release("a");
        reserved.take("a");
        reserved.release("a");
        reserved.release("a");
        assert_eq!(reserved.get("a"), 0);
    }

    /// The defect this exists to fix: with every node idle, `in_flight` is 0
    /// everywhere, so the label tie-break sends a whole burst to one node.
    /// Reservations make the second request see the first.
    #[test]
    fn a_request_already_placed_moves_the_next_one_along() {
        let nodes = vec![
            ready("a", Some("m"), 0),
            ready("b", Some("m"), 0),
            ready("c", Some("m"), 0),
        ];
        let request = RouteRequest::new(RouteMode::Throughput);

        let mut reserved = Reservations::none();
        let mut chosen = Vec::new();
        for _ in 0..6 {
            let decision = route_reserved(&nodes, &request, &reserved).expect("routes");
            reserved.take(&decision.label);
            chosen.push(decision.label);
        }
        // Without reservations every one of these is "a".
        assert_eq!(chosen, vec!["a", "b", "c", "a", "b", "c"]);
    }

    /// A node's own report and this fabric's bookkeeping describe overlapping
    /// work, not separate work. Summing them would rank a node carrying this
    /// fabric's request below an equally busy node carrying someone else's.
    #[test]
    fn a_nodes_own_load_and_our_reservation_are_not_added_together() {
        // `a` is busy with the one request we placed; `b` is equally busy with
        // traffic from elsewhere. They are carrying the same amount, so the tie
        // must break on label.
        let nodes = vec![ready("a", Some("m"), 1), ready("b", Some("m"), 1)];
        let mut reserved = Reservations::none();
        reserved.take("a");

        let request = RouteRequest::new(RouteMode::Throughput);
        let decision = route_reserved(&nodes, &request, &reserved).expect("routes");
        assert_eq!(
            decision.label, "a",
            "adding load to a reservation would have made `a` look twice as busy"
        );
    }

    #[test]
    fn an_empty_reservation_set_reproduces_the_dry_run_exactly() {
        let nodes = vec![ready("a", Some("m"), 3), ready("b", Some("m"), 1)];
        let request = RouteRequest::new(RouteMode::Throughput);
        assert_eq!(
            route_reserved(&nodes, &request, &Reservations::none()),
            route(&nodes, &request)
        );
    }

    /// A reservation ranks a node, it never disqualifies one: a single reserved
    /// node is still the only candidate rather than becoming no candidate.
    #[test]
    fn reservations_never_make_a_fabric_unroutable() {
        let nodes = vec![ready("only", Some("m"), 0)];
        let mut reserved = Reservations::none();
        for _ in 0..50 {
            reserved.take("only");
        }
        let decision = route_reserved(&nodes, &RouteRequest::new(RouteMode::Throughput), &reserved)
            .expect("a reserved node is still eligible");
        assert_eq!(decision.label, "only");
        assert_eq!(decision.reason, RouteReason::OnlyCandidate);
    }

    /// Affinity is a deliberate request for one node; a reservation on it is
    /// not a reason to send the session somewhere cold.
    #[test]
    fn affinity_still_wins_over_a_reservation() {
        let nodes = vec![ready("warm", Some("m"), 0), ready("cold", Some("m"), 0)];
        let mut reserved = Reservations::none();
        reserved.take("warm");
        reserved.take("warm");

        let request = RouteRequest::new(RouteMode::Affinity).with_sticky(Some("warm"));
        let decision = route_reserved(&nodes, &request, &reserved).expect("routes");
        assert_eq!(decision.label, "warm");
        assert_eq!(decision.reason, RouteReason::Affinity);
    }

    /// These strings reach clients over HTTP and in `--json` output, so they
    /// are part of the contract and a variant rename must not move them.
    #[test]
    fn the_reason_names_clients_see_are_pinned() {
        assert_eq!(RouteReason::Affinity.as_str(), "Affinity");
        assert_eq!(RouteReason::OnlyCandidate.as_str(), "OnlyCandidate");
        assert_eq!(RouteReason::LeastLoaded.as_str(), "LeastLoaded");
        assert_eq!(
            RouteReason::EstimatedCompletion.as_str(),
            "EstimatedCompletion"
        );
    }

    fn observe_n(estimates: &mut ServiceTimeEstimates, node: &NodeSnapshot, elapsed_ms: u64) {
        for _ in 0..MIN_SERVICE_TIME_SAMPLES {
            estimates.observe(
                node,
                "m",
                "/v1/chat/completions",
                std::time::Duration::from_millis(elapsed_ms),
            );
        }
    }

    #[test]
    fn completion_time_waits_for_every_candidate_before_using_estimates() {
        let fast = ready("fast", Some("m"), 4);
        let slow = ready("slow", Some("m"), 0);
        let nodes = vec![fast.clone(), slow];
        let mut estimates = ServiceTimeEstimates::default();
        observe_n(&mut estimates, &fast, 100);

        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("/v1/chat/completions"));
        let decision =
            route_reserved_with_estimates(&nodes, &request, &Reservations::none(), &estimates)
                .expect("routes");

        assert_eq!(decision.label, "slow", "cold estimates must fall back");
        assert_eq!(decision.reason, RouteReason::LeastLoaded);
    }

    /// The point of this mode is that idle is not the same as fast. Given real
    /// service costs, a node three requests deep still wins if it is ten times
    /// quicker than the idle one.
    #[test]
    fn a_fast_busy_node_beats_a_slow_idle_one() {
        let fast = ready("fast", Some("m"), 3);
        let slow = ready("slow", Some("m"), 0);
        let nodes = vec![fast.clone(), slow.clone()];
        let mut estimates = ServiceTimeEstimates::default();
        observe_n(&mut estimates, &fast, 40);
        observe_n(&mut estimates, &slow, 400);

        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("/v1/chat/completions"));
        let decision =
            route_reserved_with_estimates(&nodes, &request, &Reservations::none(), &estimates)
                .expect("routes");

        assert_eq!(decision.label, "fast");
        assert_eq!(decision.reason, RouteReason::EstimatedCompletion);
    }

    #[test]
    fn completion_time_explores_cold_nodes_without_overriding_load() {
        let nodes = vec![ready("a", Some("m"), 0), ready("b", Some("m"), 0)];
        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("/v1/chat/completions"));
        let mut estimates = ServiceTimeEstimates::default();
        let mut chosen = Vec::new();

        for _ in 0..(MIN_SERVICE_TIME_SAMPLES * 2) {
            let decision =
                route_reserved_with_estimates(&nodes, &request, &Reservations::none(), &estimates)
                    .expect("routes");
            chosen.push(decision.label.clone());
            let node = nodes
                .iter()
                .find(|node| node.label() == decision.label)
                .expect("chosen node exists");
            estimates.selected(node, "m", "/v1/chat/completions");
            estimates.observe(
                node,
                "m",
                "/v1/chat/completions",
                std::time::Duration::from_millis(100),
            );
        }

        assert_eq!(
            chosen,
            vec!["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]
        );

        let busy_cold = ready("c", Some("m"), 1);
        let decision = route_reserved_with_estimates(
            &[nodes[0].clone(), busy_cold],
            &request,
            &Reservations::none(),
            &estimates,
        )
        .expect("routes");
        assert_eq!(
            decision.label, "a",
            "exploration may break an equal-load tie but must not override queue depth"
        );
    }

    #[test]
    fn a_failed_cold_selection_rotates_without_becoming_a_speed_sample() {
        let nodes = vec![ready("a", Some("m"), 0), ready("b", Some("m"), 0)];
        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("/v1/chat/completions"));
        let mut estimates = ServiceTimeEstimates::default();

        let first =
            route_reserved_with_estimates(&nodes, &request, &Reservations::none(), &estimates)
                .expect("routes");
        assert_eq!(first.label, "a");
        estimates.selected(&nodes[0], "m", "/v1/chat/completions");

        let second =
            route_reserved_with_estimates(&nodes, &request, &Reservations::none(), &estimates)
                .expect("routes");
        assert_eq!(
            second.label, "b",
            "a failed selection must not monopolise cold traffic"
        );
        assert_eq!(
            estimates
                .get(&nodes[0], Some("m"), Some("/v1/chat/completions"))
                .expect("selection is tracked")
                .samples,
            0,
            "selection recency must not masquerade as a successful service sample"
        );
    }

    #[test]
    fn invalidating_a_degraded_node_returns_the_class_to_cold_fallback() {
        let fast = ready("fast", Some("m"), 4);
        let slow = ready("slow", Some("m"), 0);
        let nodes = vec![fast.clone(), slow.clone()];
        let mut estimates = ServiceTimeEstimates::default();
        observe_n(&mut estimates, &fast, 100);
        observe_n(&mut estimates, &slow, 1_000);
        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("/v1/chat/completions"));

        let learned =
            route_reserved_with_estimates(&nodes, &request, &Reservations::none(), &estimates)
                .expect("routes");
        assert_eq!(learned.label, "fast");

        estimates.invalidate(&fast, "m", "/v1/chat/completions");
        let after_failure =
            route_reserved_with_estimates(&nodes, &request, &Reservations::none(), &estimates)
                .expect("routes");
        assert_eq!(after_failure.label, "slow");
        assert_eq!(after_failure.reason, RouteReason::LeastLoaded);
    }

    #[test]
    fn a_stale_estimate_is_explored_before_it_decides_placement_again() {
        let now = std::time::Instant::now();
        let fast = ready("fast", Some("m"), 4);
        let slow = ready("slow", Some("m"), 0);
        let nodes = vec![fast.clone(), slow.clone()];
        let mut estimates = ServiceTimeEstimates::default();
        for _ in 0..MIN_SERVICE_TIME_SAMPLES {
            estimates.observe_at(
                &fast,
                "m",
                "/v1/chat/completions",
                std::time::Duration::from_millis(100),
                now,
            );
            estimates.observe_at(
                &slow,
                "m",
                "/v1/chat/completions",
                std::time::Duration::from_millis(1_000),
                now,
            );
        }
        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("/v1/chat/completions"));

        let learned = route_reserved_with_estimates_at(
            &nodes,
            &request,
            &Reservations::none(),
            &estimates,
            now,
        )
        .expect("routes");
        assert_eq!(learned.label, "fast");

        let stale = route_reserved_with_estimates_at(
            &nodes,
            &request,
            &Reservations::none(),
            &estimates,
            now + MAX_SERVICE_TIME_AGE + std::time::Duration::from_nanos(1),
        )
        .expect("routes");
        assert_eq!(
            stale.label, "slow",
            "stale estimates must fall back to load"
        );
        assert_eq!(stale.reason, RouteReason::LeastLoaded);

        estimates.observe_at(
            &fast,
            "m",
            "/v1/chat/completions",
            std::time::Duration::from_millis(100),
            now + MAX_SERVICE_TIME_AGE + std::time::Duration::from_nanos(1),
        );
        assert_eq!(
            estimates
                .get(&fast, Some("m"), Some("/v1/chat/completions"))
                .expect("the class remains present")
                .samples,
            1,
            "one fresh sample must not revive a stale mature estimate"
        );
    }

    #[test]
    fn completion_time_accounts_for_both_speed_and_outstanding_work() {
        let fast = ready("fast", Some("m"), 4);
        let slow = ready("slow", Some("m"), 0);
        let nodes = vec![fast.clone(), slow.clone()];
        let mut estimates = ServiceTimeEstimates::default();
        observe_n(&mut estimates, &fast, 100);
        observe_n(&mut estimates, &slow, 1_000);
        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("/v1/chat/completions"));

        let decision =
            route_reserved_with_estimates(&nodes, &request, &Reservations::none(), &estimates)
                .expect("routes");
        assert_eq!(decision.label, "fast");
        assert_eq!(decision.reason, RouteReason::EstimatedCompletion);

        let fast = ready("fast", Some("m"), 9);
        let decision = route_reserved_with_estimates(
            &[fast, slow],
            &request,
            &Reservations::none(),
            &estimates,
        )
        .expect("routes");
        assert_eq!(
            decision.label, "slow",
            "the slow node must receive work once the fast node's queue erases its advantage"
        );
    }

    #[test]
    fn service_time_is_scoped_to_node_identity_model_and_workload() {
        let measured = ready("fast", Some("m"), 4);
        let other = ready("slow", Some("m"), 0);
        let mut estimates = ServiceTimeEstimates::default();
        observe_n(&mut estimates, &measured, 100);
        observe_n(&mut estimates, &other, 1_000);

        let another_fast = ready("fast", Some("another-model"), 4);
        let another_slow = ready("slow", Some("another-model"), 0);
        let another_model = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("another-model"))
            .with_service_class(Some("/v1/chat/completions"));
        let decision = route_reserved_with_estimates(
            &[another_fast, another_slow],
            &another_model,
            &Reservations::none(),
            &estimates,
        )
        .expect("routes");
        assert_eq!(decision.label, "slow");
        assert_eq!(decision.reason, RouteReason::LeastLoaded);

        let another_route = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("/v1/embeddings"));
        let decision = route_reserved_with_estimates(
            &[measured.clone(), other.clone()],
            &another_route,
            &Reservations::none(),
            &estimates,
        )
        .expect("routes");
        assert_eq!(decision.label, "slow");
        assert_eq!(decision.reason, RouteReason::LeastLoaded);

        let mut moved = measured.clone();
        moved.spec.port += 1;
        let request = RouteRequest::new(RouteMode::CompletionTime)
            .with_model(Some("m"))
            .with_service_class(Some("/v1/chat/completions"));
        let decision = route_reserved_with_estimates(
            &[moved, other],
            &request,
            &Reservations::none(),
            &estimates,
        )
        .expect("routes");
        assert_eq!(decision.reason, RouteReason::LeastLoaded);
    }

    #[test]
    fn service_time_state_is_bounded() {
        let node = ready("node", Some("m"), 0);
        let now = std::time::Instant::now();
        let mut estimates = ServiceTimeEstimates::default();
        for index in 0..=MAX_SERVICE_CLASSES {
            estimates.selected_at(
                &node,
                "m",
                &format!("workload-{index}"),
                now + std::time::Duration::from_nanos(index as u64),
            );
        }

        assert_eq!(estimates.0.len(), MAX_SERVICE_CLASSES);
        assert!(
            estimates
                .get(&node, Some("m"), Some("workload-0"))
                .is_none(),
            "the least-recent entry was not pruned"
        );
        assert!(
            estimates
                .get(&node, Some("m"), Some("workload-1024"))
                .is_some(),
            "the newest entry was pruned"
        );
    }

    #[test]
    fn a_dead_fabric_reports_how_it_died() {
        let nodes = vec![unreachable("a"), not_ready("b"), unreachable("c")];
        assert_eq!(
            route(&nodes, &RouteRequest::new(RouteMode::Throughput)),
            Err(RouteError::AllNodesUnavailable {
                unreachable: 2,
                not_ready: 1,
                not_placeable: 0,
            })
        );
    }

    #[test]
    fn an_unreachable_node_is_never_selected() {
        let nodes = vec![unreachable("a"), ready("b", None, 0)];
        let decision =
            route(&nodes, &RouteRequest::new(RouteMode::Throughput)).expect("b can serve");
        assert_eq!(decision.label, "b");
        assert_eq!(decision.reason, RouteReason::OnlyCandidate);
    }

    #[test]
    fn a_wholly_idle_fabric_routes() {
        // Regression: `engine_queue_depth` is a load gauge, not a capacity bound.
        // Reading it as a bound made two healthy idle nodes both look "full" and
        // every request was refused. Verified against two live nodes.
        let nodes = vec![ready("windows", None, 0), ready("mac", None, 0)];
        let decision = route(&nodes, &RouteRequest::new(RouteMode::Throughput))
            .expect("an idle fabric must be routable");
        assert_eq!(decision.label, "mac", "ties break on label");
    }

    #[test]
    fn the_least_loaded_eligible_node_wins() {
        let nodes = vec![
            ready("a", None, 3),
            ready("b", None, 1),
            ready("c", None, 2),
        ];
        let decision = route(&nodes, &RouteRequest::new(RouteMode::Throughput)).expect("routes");
        assert_eq!(decision.label, "b");
        assert_eq!(decision.reason, RouteReason::LeastLoaded);
    }

    #[test]
    fn ties_break_on_label_so_a_dry_run_predicts_the_real_decision() {
        let forward = vec![ready("zulu", None, 1), ready("alpha", None, 1)];
        let reversed = vec![ready("alpha", None, 1), ready("zulu", None, 1)];
        let request = RouteRequest::new(RouteMode::Throughput);
        assert_eq!(route(&forward, &request).expect("routes").label, "alpha");
        assert_eq!(route(&reversed, &request).expect("routes").label, "alpha");
    }

    #[test]
    fn a_model_request_only_matches_a_node_actually_serving_it() {
        let nodes = vec![
            ready("a", Some("llama-3b"), 0),
            ready("b", Some("qwen-4b"), 0),
        ];
        let request = RouteRequest::new(RouteMode::Throughput).with_model(Some("qwen-4b"));
        assert_eq!(route(&nodes, &request).expect("routes").label, "b");
    }

    #[test]
    fn a_missing_model_names_what_the_fabric_does_serve() {
        let nodes = vec![
            ready("a", Some("llama-3b"), 0),
            ready("b", Some("qwen-4b"), 0),
        ];
        let request = RouteRequest::new(RouteMode::Throughput).with_model(Some("gemma-27b"));
        assert_eq!(
            route(&nodes, &request),
            Err(RouteError::ModelUnavailable {
                model: "gemma-27b".to_string(),
                serving: vec!["llama-3b".to_string(), "qwen-4b".to_string()],
                unobserved: 0,
                held_by_unplaced: Vec::new(),
            })
        );
    }

    #[test]
    fn a_busy_node_still_receives_work_when_it_is_the_only_one() {
        // The fabric cannot know a node's bound, so it must not invent one and
        // refuse. A genuinely full node answers its own 503.
        let nodes = vec![ready("a", None, 99)];
        let decision = route(&nodes, &RouteRequest::new(RouteMode::Throughput)).expect("routes");
        assert_eq!(decision.label, "a");
    }

    /// A node that never answered may be the one holding the model, so the
    /// refusal has to record that the fabric could not see the whole picture.
    /// A caller reads this to tell a permanent refusal from a temporary one.
    #[test]
    fn a_refusal_records_the_nodes_it_could_not_consult() {
        let nodes = vec![
            ready("up", Some("llama-3b"), 0),
            unreachable("down"),
            not_ready("loading"),
        ];
        let request = RouteRequest::new(RouteMode::Throughput).with_model(Some("qwen-4b"));
        match route(&nodes, &request).expect_err("nothing ready serves it") {
            RouteError::ModelUnavailable { unobserved, .. } => assert_eq!(unobserved, 2),
            other => panic!("expected ModelUnavailable, got {other:?}"),
        }
    }

    /// The control: with every node accounted for, the same refusal is final.
    #[test]
    fn a_refusal_from_a_fully_observed_fabric_records_nothing_missing() {
        let nodes = vec![ready("a", Some("llama-3b"), 0), ready("b", None, 0)];
        let request = RouteRequest::new(RouteMode::Throughput).with_model(Some("qwen-4b"));
        match route(&nodes, &request).expect_err("nothing serves it") {
            RouteError::ModelUnavailable { unobserved, .. } => assert_eq!(unobserved, 0),
            other => panic!("expected ModelUnavailable, got {other:?}"),
        }
    }

    /// The message is what an operator acts on, so it has to say the fabric
    /// could not see everything rather than implying the model is simply gone.
    #[test]
    fn an_incomplete_refusal_says_so_in_its_message() {
        let nodes = vec![ready("up", Some("llama-3b"), 0), unreachable("down")];
        let request = RouteRequest::new(RouteMode::Throughput).with_model(Some("qwen-4b"));
        let message = route(&nodes, &request)
            .expect_err("nothing ready serves it")
            .to_string();
        assert!(message.contains("llama-3b"), "{message}");
        assert!(
            message.contains("1 node could not be consulted"),
            "{message}"
        );
    }

    #[test]
    fn affinity_keeps_a_session_on_its_warm_node_even_when_busier() {
        let nodes = vec![ready("warm", None, 3), ready("idle", None, 0)];
        let request = RouteRequest::new(RouteMode::Affinity).with_sticky(Some("warm"));
        let decision = route(&nodes, &request).expect("routes");
        assert_eq!(decision.label, "warm");
        assert_eq!(decision.reason, RouteReason::Affinity);
        assert_eq!(decision.affinity_lost, None);
    }

    #[test]
    fn affinity_degrades_instead_of_failing_when_the_warm_node_dies() {
        let nodes = vec![unreachable("warm"), ready("idle", None, 0)];
        let request = RouteRequest::new(RouteMode::Affinity).with_sticky(Some("warm"));
        let decision = route(&nodes, &request).expect("falls back rather than failing");
        assert_eq!(decision.label, "idle");
        assert_eq!(decision.affinity_lost.as_deref(), Some("warm"));
    }

    #[test]
    fn affinity_degrades_when_the_warm_node_no_longer_holds_the_model() {
        let nodes = vec![
            ready("warm", Some("llama-3b"), 0),
            ready("other", Some("qwen-4b"), 5),
        ];
        let request = RouteRequest::new(RouteMode::Affinity)
            .with_sticky(Some("warm"))
            .with_model(Some("qwen-4b"));
        let decision = route(&nodes, &request).expect("routes");
        assert_eq!(decision.label, "other");
        assert_eq!(decision.affinity_lost.as_deref(), Some("warm"));
    }

    #[test]
    fn throughput_mode_ignores_a_sticky_hint_entirely() {
        let nodes = vec![ready("warm", None, 3), ready("idle", None, 0)];
        let request = RouteRequest::new(RouteMode::Throughput).with_sticky(Some("warm"));
        let decision = route(&nodes, &request).expect("routes");
        assert_eq!(decision.label, "idle");
        // Affinity was never requested, so nothing was lost.
        assert_eq!(decision.affinity_lost, None);
    }

    #[test]
    fn a_node_with_no_model_cannot_satisfy_a_model_request() {
        let nodes = vec![ready("a", None, 0)];
        let request = RouteRequest::new(RouteMode::Throughput).with_model(Some("llama-3b"));
        assert_eq!(
            route(&nodes, &request),
            Err(RouteError::ModelUnavailable {
                model: "llama-3b".to_string(),
                serving: Vec::new(),
                unobserved: 0,
                held_by_unplaced: Vec::new(),
            })
        );
    }

    /// A foreign node holding `installed`, with `resident` as its own listing
    /// said it (`None` when the listing said nothing). Its active model is the
    /// one resident model, when there is exactly one, as the adapters report.
    fn holding(
        label: &str,
        engine: NodeEngine,
        installed: &[&str],
        resident: Option<&[&str]>,
    ) -> NodeSnapshot {
        let resident: Option<Vec<String>> =
            resident.map(|models| models.iter().map(|m| m.to_string()).collect());
        NodeSnapshot {
            spec: NodeSpec {
                label: label.to_string(),
                host: "127.0.0.1".to_string(),
                port: engine.default_port(),
                engine,
            },
            status: NodeStatus::Ready(NodeReady {
                engine,
                active_model_id: match resident.as_deref() {
                    Some([only]) => Some(only.clone()),
                    _ => None,
                },
                models: installed.iter().map(|m| m.to_string()).collect(),
                resident_models: resident,
                backend: None,
                version: (engine == NodeEngine::Ollama).then(|| "0.33.2".to_string()),
                load: None,
            }),
            latency: None,
        }
    }

    fn tools() -> Requirements {
        Requirements {
            tool_calls: true,
            ..Requirements::default()
        }
    }

    fn rerank() -> Requirements {
        Requirements {
            rerank_route: true,
            ..Requirements::default()
        }
    }

    /// An unmet requirement is a fact about what the node can do, and saying
    /// "this fabric does not place on that engine" instead sends an operator
    /// to a flag they have already given.
    #[test]
    fn a_tool_calling_request_is_refused_for_the_capability_it_lacks_not_as_an_unplaceable_engine()
    {
        let nodes = [foreign("studio", NodeEngine::Ollama, "m")];
        let error = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .with_mixed_engines(MIXED)
                .requiring(tools()),
        )
        .expect_err("nobody measured it");
        match &error {
            RouteError::RequirementUnmet {
                requirement,
                unmet,
                unobserved,
                ..
            } => {
                assert_eq!(*requirement, "tool_calls");
                assert_eq!(unmet.len(), 1);
                assert_eq!(unmet[0].0, "studio");
                assert_eq!(*unobserved, 0);
            }
            other => panic!("expected RequirementUnmet, got {other:?}"),
        }
        let message = error.to_string();
        assert!(message.contains("studio"), "{message}");
        assert!(message.contains("measured"), "{message}");
    }

    #[test]
    fn a_tool_calling_request_goes_to_our_engine_even_when_the_foreign_node_is_idler() {
        let nodes = [
            ready("win", Some("m"), 5),
            foreign("studio", NodeEngine::Ollama, "m"),
        ];
        let plain = RouteRequest::new(RouteMode::Throughput)
            .with_model(Some("m"))
            .with_mixed_engines(MIXED);
        assert_eq!(
            route(&nodes, &plain).expect("routes").label,
            "studio",
            "without tools the idler node wins, so the next assertion is the guard's doing"
        );
        let decision = route(&nodes, &plain.requiring(tools())).expect("our engine takes tools");
        assert_eq!(decision.label, "win");
        assert_eq!(decision.engine, NodeEngine::Camelid);
    }

    #[test]
    fn an_installed_model_on_an_engine_that_loads_on_demand_is_matched_and_says_so() {
        let nodes = [holding("studio", NodeEngine::Ollama, &["m"], Some(&[]))];
        let decision = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .with_mixed_engines(MIXED),
        )
        .expect("an engine that loads on demand serves what it has installed");
        assert_eq!(decision.label, "studio");
        assert_eq!(decision.residency, Some(Residency::NotResident));
        assert_eq!(decision.model.as_deref(), Some("m"));
    }

    #[test]
    fn an_installed_model_on_an_engine_whose_loading_is_unprobed_is_not_matched() {
        // LM Studio's listing says "not-loaded", and whether naming it loads it
        // is a setting its API does not publish.
        let nodes = [holding(
            "desk",
            NodeEngine::LmStudio,
            &["m", "other"],
            Some(&["other"]),
        )];
        let error = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .with_mixed_engines(MIXED),
        )
        .expect_err("installed is not servable without a declaration");
        assert!(
            matches!(error, RouteError::ModelUnavailable { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn unknown_residency_is_never_read_as_resident() {
        let unknown = holding("a-unknown", NodeEngine::Ollama, &["m"], None);
        let only = route(
            std::slice::from_ref(&unknown),
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .with_mixed_engines(MIXED),
        )
        .expect("still placed");
        assert_eq!(only.residency, Some(Residency::Unknown));

        // Equal load, and the unknown one sorts first: only the price can put
        // the resident node ahead.
        let nodes = [
            unknown,
            holding("b-warm", NodeEngine::Ollama, &["m"], Some(&["m"])),
        ];
        let decision = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .with_mixed_engines(MIXED),
        )
        .expect("routes");
        assert_eq!(decision.label, "b-warm");
        assert_eq!(decision.residency, Some(Residency::Resident));
    }

    #[test]
    fn a_node_holding_two_resident_models_serves_either() {
        let nodes = [holding(
            "studio",
            NodeEngine::Ollama,
            &["a", "b"],
            Some(&["a", "b"]),
        )];
        for model in ["a", "b"] {
            let decision = route(
                &nodes,
                &RouteRequest::new(RouteMode::Throughput)
                    .with_model(Some(model))
                    .with_mixed_engines(MIXED),
            )
            .unwrap_or_else(|error| panic!("{model}: {error}"));
            assert_eq!(decision.label, "studio");
            assert_eq!(decision.residency, Some(Residency::Resident), "{model}");
        }
    }

    #[test]
    fn a_resident_holder_outranks_a_cold_one_at_equal_load() {
        // The cold node sorts first, so a tie would go to it.
        let nodes = [
            holding("a-cold", NodeEngine::Ollama, &["m"], Some(&[])),
            holding("b-warm", NodeEngine::Ollama, &["m"], Some(&["m"])),
        ];
        let decision = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .with_mixed_engines(MIXED),
        )
        .expect("routes");
        assert_eq!(decision.label, "b-warm");
    }

    /// A price, not a tier: a warm holder that publishes no load could
    /// otherwise take every request for its model forever.
    #[test]
    fn a_cold_holder_is_chosen_once_the_resident_one_is_carrying_more_than_the_cold_price() {
        // The warm node sorts first, so at a tie it keeps the work.
        let nodes = [
            holding("a-warm", NodeEngine::Ollama, &["m"], Some(&["m"])),
            holding("b-cold", NodeEngine::Ollama, &["m"], Some(&[])),
        ];
        let request = RouteRequest::new(RouteMode::Throughput)
            .with_model(Some("m"))
            .with_mixed_engines(MIXED);

        let mut reserved = Reservations::none();
        for _ in 0..COLD_LOAD_COST {
            reserved.take("a-warm");
        }
        assert_eq!(
            route_reserved(&nodes, &request, &reserved)
                .expect("routes")
                .label,
            "a-warm",
            "carrying exactly the cold price is a tie, and ties keep the warm node"
        );

        reserved.take("a-warm");
        let spilled = route_reserved(&nodes, &request, &reserved).expect("routes");
        assert_eq!(spilled.label, "b-cold");
        assert_eq!(spilled.residency, Some(Residency::NotResident));
    }

    #[test]
    fn a_declared_alias_places_on_the_local_id_and_records_the_claim() {
        let nodes = [foreign("studio", NodeEngine::Ollama, "llama1b-q8:latest")];
        let aliases = crate::fabric::aliases::parse_model_aliases(&[
            "llama-1b=studio:llama1b-q8:latest".to_string(),
        ])
        .expect("parses");
        let decision = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("llama-1b"))
                .with_mixed_engines(MIXED)
                .with_aliases(&aliases),
        )
        .expect("the alias names what studio holds");
        assert_eq!(decision.model.as_deref(), Some("llama1b-q8:latest"));
        assert_eq!(
            decision.model_identity,
            Some(ModelIdentity::AssertedByOperator)
        );

        let direct = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("llama1b-q8:latest"))
                .with_mixed_engines(MIXED),
        )
        .expect("its own id");
        assert_eq!(direct.model.as_deref(), Some("llama1b-q8:latest"));
        assert_eq!(direct.model_identity, Some(ModelIdentity::SameId));

        assert!(
            route(
                &nodes,
                &RouteRequest::new(RouteMode::Throughput)
                    .with_model(Some("llama-1b"))
                    .with_mixed_engines(MIXED),
            )
            .is_err(),
            "without the declaration the canonical id names nothing studio holds"
        );
    }

    #[test]
    fn a_request_naming_no_model_is_placed_only_where_the_model_it_will_get_is_known() {
        let two = holding("a-two", NodeEngine::Ollama, &["x", "y"], Some(&["x", "y"]));
        let one = holding("b-one", NodeEngine::Ollama, &["z"], Some(&["z"]));
        let mixed = RouteRequest::new(RouteMode::Throughput).with_mixed_engines(MIXED);

        let decision = route(&[two.clone(), one], &mixed).expect("b-one is known");
        assert_eq!(
            decision.label, "b-one",
            "a-two sorts first and would win a tie, but which of its models a request would get is unknown"
        );
        assert_eq!(decision.model.as_deref(), Some("z"));
        assert_eq!(
            route(std::slice::from_ref(&two), &mixed),
            Err(RouteError::ModelUnspecified { unobserved: 0 })
        );

        // Camelid-only placement of a request naming no model is untouched:
        // every ready node is a candidate and nothing is written.
        let nodes = [ready("a", None, 0), ready("b", Some("m"), 0)];
        let decision = route(&nodes, &RouteRequest::new(RouteMode::Throughput)).expect("routes");
        assert_eq!(decision.label, "a");
        assert_eq!(decision.reason, RouteReason::LeastLoaded);
        assert_eq!(decision.model, None, "nothing is rewritten");
        assert_eq!(decision.residency, None);
    }

    /// S5. Every Camelid-only decision this module made before, remade with
    /// each requirement set: none of the new stages may move one.
    #[test]
    fn a_camelid_only_fabric_decides_exactly_as_before() {
        type Expected = Result<(&'static str, RouteReason, Option<&'static str>), RouteError>;
        let mut reserved_a = Reservations::none();
        reserved_a.take("a");
        let fast = ready("fast", Some("m"), 3);
        let slow = ready("slow", Some("m"), 0);
        let mut estimates = ServiceTimeEstimates::default();
        observe_n(&mut estimates, &fast, 40);
        observe_n(&mut estimates, &slow, 400);

        let cases: Vec<(
            &str,
            Vec<NodeSnapshot>,
            RouteRequest<'_>,
            Reservations,
            Expected,
        )> = vec![
            (
                "an unreachable node is skipped",
                vec![unreachable("a"), ready("b", None, 0)],
                RouteRequest::new(RouteMode::Throughput),
                Reservations::none(),
                Ok(("b", RouteReason::OnlyCandidate, None)),
            ),
            (
                "an idle fabric ties on label",
                vec![ready("windows", None, 0), ready("mac", None, 0)],
                RouteRequest::new(RouteMode::Throughput),
                Reservations::none(),
                Ok(("mac", RouteReason::LeastLoaded, None)),
            ),
            (
                "least loaded wins",
                vec![
                    ready("a", None, 3),
                    ready("b", None, 1),
                    ready("c", None, 2),
                ],
                RouteRequest::new(RouteMode::Throughput),
                Reservations::none(),
                Ok(("b", RouteReason::LeastLoaded, None)),
            ),
            (
                "a model matches only its server",
                vec![
                    ready("a", Some("llama-3b"), 0),
                    ready("b", Some("qwen-4b"), 0),
                ],
                RouteRequest::new(RouteMode::Throughput).with_model(Some("qwen-4b")),
                Reservations::none(),
                Ok(("b", RouteReason::OnlyCandidate, None)),
            ),
            (
                "a not-ready node is skipped",
                vec![not_ready("a"), ready("b", Some("m"), 0)],
                RouteRequest::new(RouteMode::Throughput).with_model(Some("m")),
                Reservations::none(),
                Ok(("b", RouteReason::OnlyCandidate, None)),
            ),
            (
                "affinity holds on a busier warm node",
                vec![ready("warm", None, 3), ready("idle", None, 0)],
                RouteRequest::new(RouteMode::Affinity).with_sticky(Some("warm")),
                Reservations::none(),
                Ok(("warm", RouteReason::Affinity, None)),
            ),
            (
                "affinity degrades when the warm node dies",
                vec![unreachable("warm"), ready("idle", None, 0)],
                RouteRequest::new(RouteMode::Affinity).with_sticky(Some("warm")),
                Reservations::none(),
                Ok(("idle", RouteReason::OnlyCandidate, Some("warm"))),
            ),
            (
                "a reservation moves the next request along",
                vec![
                    ready("a", Some("m"), 0),
                    ready("b", Some("m"), 0),
                    ready("c", Some("m"), 0),
                ],
                RouteRequest::new(RouteMode::Throughput),
                reserved_a.clone(),
                Ok(("b", RouteReason::LeastLoaded, None)),
            ),
            (
                "completion time prefers the fast busy node",
                vec![fast.clone(), slow.clone()],
                RouteRequest::new(RouteMode::CompletionTime)
                    .with_model(Some("m"))
                    .with_service_class(Some("/v1/chat/completions")),
                Reservations::none(),
                Ok(("fast", RouteReason::EstimatedCompletion, None)),
            ),
            (
                "a dead fabric says how it died",
                vec![unreachable("a"), not_ready("b")],
                RouteRequest::new(RouteMode::Throughput),
                Reservations::none(),
                Err(RouteError::AllNodesUnavailable {
                    unreachable: 1,
                    not_ready: 1,
                    not_placeable: 0,
                }),
            ),
            (
                "a missing model names what is served and what could not be asked",
                vec![ready("a", Some("llama-3b"), 0), unreachable("b")],
                RouteRequest::new(RouteMode::Throughput).with_model(Some("qwen-4b")),
                Reservations::none(),
                Err(RouteError::ModelUnavailable {
                    model: "qwen-4b".to_string(),
                    serving: vec!["llama-3b".to_string()],
                    unobserved: 1,
                    held_by_unplaced: Vec::new(),
                }),
            ),
        ];

        for (name, nodes, request, reserved, expected) in &cases {
            for requires in [Requirements::default(), tools(), rerank()] {
                let got = route_reserved_with_estimates(
                    nodes,
                    &request.requiring(requires),
                    reserved,
                    &estimates,
                )
                .map(|decision| {
                    (
                        decision.label,
                        decision.reason,
                        decision.affinity_lost,
                        decision.model,
                    )
                });
                let expected = expected.clone().map(|(label, reason, lost)| {
                    (
                        label.to_string(),
                        reason,
                        lost.map(str::to_string),
                        request.model.map(str::to_string),
                    )
                });
                assert_eq!(got, expected, "{name} with {requires:?}");
            }
        }
    }

    #[test]
    fn a_rerank_request_never_lands_on_an_engine_without_the_route() {
        let nodes = [
            ready("win", Some("m"), 5),
            foreign("studio", NodeEngine::Ollama, "m"),
        ];
        let mixed = RouteRequest::new(RouteMode::Throughput)
            .with_model(Some("m"))
            .with_mixed_engines(MIXED);
        assert_eq!(route(&nodes, &mixed).expect("routes").label, "studio");
        assert_eq!(
            route(&nodes, &mixed.requiring(rerank()))
                .expect("our engine has the route")
                .label,
            "win"
        );

        let error = route(&nodes[1..], &mixed.requiring(rerank())).expect_err("no route");
        assert!(
            matches!(
                error,
                RouteError::RequirementUnmet {
                    requirement: "rerank_route",
                    ..
                }
            ),
            "{error:?}"
        );
        let message = error.to_string();
        assert!(message.contains("exposes a rerank route"), "{message}");
        assert!(!message.contains("supports"), "{message}");
    }

    #[test]
    fn a_model_only_an_unplaced_engine_holds_is_refused_as_settled_and_names_it() {
        let nodes = [
            ready("a-camelid", Some("a"), 0),
            foreign("b-ollama", NodeEngine::Ollama, "x"),
        ];
        let error = route(
            &nodes,
            &RouteRequest::new(RouteMode::Throughput).with_model(Some("x")),
        )
        .expect_err("refused: only a foreign node holds it");
        assert_eq!(
            error,
            RouteError::ModelUnavailable {
                model: "x".to_string(),
                serving: vec!["a".to_string()],
                unobserved: 0,
                held_by_unplaced: vec!["b-ollama".to_string()],
            },
            "a node that answered was consulted, so the refusal is settled"
        );
        let message = error.to_string();
        assert!(message.contains("b-ollama"), "{message}");
        assert!(message.contains(MIXED_ENGINES_FLAG), "{message}");

        let public = error.public_message();
        assert!(!public.contains("b-ollama"), "{public}");
        assert!(!public.contains(MIXED_ENGINES_FLAG), "{public}");
        assert!(
            public.contains("no ready node is serving model `x`"),
            "{public}"
        );
    }

    #[test]
    fn a_refusal_off_the_machine_names_no_node_and_no_engine_version() {
        let error = route(
            &[foreign("b-ollama", NodeEngine::Ollama, "m")],
            &RouteRequest::new(RouteMode::Throughput)
                .with_model(Some("m"))
                .with_mixed_engines(MIXED)
                .requiring(tools()),
        )
        .expect_err("unmeasured");
        let public = error.public_message();
        assert!(!public.contains("b-ollama"), "{public}");
        assert!(!public.contains("ollama"), "{public}");
        assert!(public.contains("1 node holding this model"), "{public}");
    }
}
