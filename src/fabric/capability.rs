//! What an engine can be asked, and **how we know**.
//!
//! P2 gave the fabric engines it reads but does not place on. The question that
//! immediately follows is *why not*, and the answer has to be more specific than
//! a hard-coded list of names — otherwise every new backend is a fresh argument
//! rather than a fresh row.
//!
//! So each capability carries its own provenance:
//!
//! * [`Provenance::Declared`] — it follows from the engine's documented API
//!   surface. There is no endpoint for queue depth in Ollama's API, so a node
//!   running it reports no load; that is a fact about the API, not a guess.
//! * [`Provenance::Measured`] — we ran it against **that exact backend and
//!   version** and recorded what happened.
//! * [`Provenance::NotProbed`] — nobody has checked, and this build will not
//!   guess on the operator's behalf.
//!
//! Two rules keep this from drifting into marketing:
//!
//! 1. **A measurement is about the build it was taken on.** The table below is
//!    keyed on an exact version string and matched exactly. `ollama 0.33.1`
//!    behaving one way is not a claim about `ollama 0.33.3`, which reads as *not
//!    probed* until somebody measures it.
//! 2. **A foreign engine is never credited with tool calling on documentation
//!    alone.** Vendors document it; we have measured it failing anyway. Until
//!    there is a measurement for that exact version the answer is *not probed*,
//!    never *declared yes*.

use serde::{Serialize, Serializer};

use super::engine::NodeEngine;

/// How a capability answer was arrived at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    Measured,
    Declared,
    NotProbed,
}

impl Provenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Declared => "declared",
            Self::NotProbed => "not_probed",
        }
    }
}

impl Serialize for Provenance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

/// One answer, and how we came by it. `supported` is `None` exactly when the
/// provenance is [`Provenance::NotProbed`] — an unknown is never a `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Capability {
    pub supported: Option<bool>,
    pub provenance: Provenance,
    /// Where the answer came from, in the operator's language.
    pub detail: &'static str,
}

impl Capability {
    const fn declared(supported: bool, detail: &'static str) -> Self {
        Self {
            supported: Some(supported),
            provenance: Provenance::Declared,
            detail,
        }
    }

    const fn measured(supported: bool, detail: &'static str) -> Self {
        Self {
            supported: Some(supported),
            provenance: Provenance::Measured,
            detail,
        }
    }

    const fn not_probed(detail: &'static str) -> Self {
        Self {
            supported: None,
            provenance: Provenance::NotProbed,
            detail,
        }
    }
}

/// Everything the fabric wants to know about an engine before trusting it with
/// anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Capabilities {
    /// Publishes a load figure placement could rank on.
    pub load_reporting: Capability,
    /// Refuses a full queue with a typed code the fabric can re-place on a
    /// sibling, rather than an error indistinguishable from a real failure.
    pub typed_backpressure: Capability,
    /// Can attest that a session's prefix is still warm, which is what makes
    /// affinity worth honouring rather than a guess.
    pub warm_prefix: Capability,
    pub tool_calls: Capability,
    pub embeddings: Capability,
    /// Answers a request the same way whatever requests it served before —
    /// that a prompt cache, a reused KV prefix or a slot left over from an
    /// earlier request never changes the bytes. A comparison's
    /// back-to-back self-consistency check cannot see an engine that fails
    /// this: it repeats itself run after run from the same history.
    pub history_neutral: Capability,
}

impl Capabilities {
    /// Name/answer pairs in a fixed order, for rendering.
    pub fn entries(&self) -> [(&'static str, Capability); 6] {
        [
            ("load_reporting", self.load_reporting),
            ("typed_backpressure", self.typed_backpressure),
            ("warm_prefix", self.warm_prefix),
            ("tool_calls", self.tool_calls),
            ("embeddings", self.embeddings),
            ("history_neutral", self.history_neutral),
        ]
    }

    /// The reasons this engine cannot be placed on, in the order they matter.
    /// Empty for an engine the fabric routes to.
    pub fn placement_blockers(&self) -> Vec<&'static str> {
        let mut blockers = Vec::new();
        if self.load_reporting.supported != Some(true) {
            blockers.push("publishes no load to rank on");
        }
        if self.typed_backpressure.supported != Some(true) {
            blockers.push("a full queue is indistinguishable from a failure");
        }
        if self.warm_prefix.supported != Some(true) {
            blockers.push("cannot attest a warm prefix, so affinity would be a guess");
        }
        blockers
    }
}

/// One thing we actually ran, against one exact build.
struct Measurement {
    engine: NodeEngine,
    version: &'static str,
    supported: bool,
    detail: &'static str,
}

/// Tool-calling measurements. Exact version match only — see rule 1 above.
const TOOL_CALL_MEASUREMENTS: &[Measurement] = &[
    Measurement {
        engine: NodeEngine::Camelid,
        version: "v0.6.1-267",
        supported: true,
        detail: "measured on this build: 180 requests, 0 non-200, tool arguments parsed on every call",
    },
    Measurement {
        engine: NodeEngine::Ollama,
        version: "0.33.1",
        supported: false,
        detail: "measured on this exact version: all 15 tool items failed across 3 repetitions (45 of 180 requests answered non-200)",
    },
];

fn tool_calls_for(engine: NodeEngine, version: Option<&str>) -> Capability {
    if let Some(version) = version {
        for measurement in TOOL_CALL_MEASUREMENTS {
            if measurement.engine == engine && measurement.version == version {
                return Capability::measured(measurement.supported, measurement.detail);
            }
        }
    }
    match engine {
        // Our own engine, and the contract its test suite pins.
        NodeEngine::Camelid => {
            Capability::declared(true, "the engine's own tool-call contract is covered by its test suite")
        }
        // Never credited on documentation alone.
        _ => Capability::not_probed(
            "this backend and version has not been measured here; vendor documentation is not taken as evidence",
        ),
    }
}

/// Request-history measurements. Exact version match only, like the table
/// above. Only a `false` is recorded from a single counterexample: one request
/// answered two ways is proof of dependence, while no number of agreeing runs
/// proves independence, so a `true` needs more than this table can hold.
const HISTORY_MEASUREMENTS: &[Measurement] = &[Measurement {
    engine: NodeEngine::Ollama,
    version: "0.33.2",
    supported: false,
    detail: "measured on this exact version: the same temperature-0, seed-42 request on an unchanged Llama-3.2-1B-Instruct Q8_0 answered two different ways depending on the requests served before it",
}];

fn history_neutral_for(engine: NodeEngine, version: Option<&str>) -> Capability {
    if let Some(version) = version {
        for measurement in HISTORY_MEASUREMENTS {
            if measurement.engine == engine && measurement.version == version {
                return Capability::measured(measurement.supported, measurement.detail);
            }
        }
    }
    // Our own engine included. It keeps prompt caches, and nothing here has
    // shown a cache hit and a cold prefill produce the same bytes on every
    // lane, so it is not declared neutral on its own say-so either.
    Capability::not_probed(
        "no measurement here shows this engine and version answers the same whatever requests it served before",
    )
}

/// What an engine can be asked, given the version it reported.
///
/// `version` is `Option` because not every engine publishes one over HTTP:
/// LM Studio has no version endpoint, so a measurement can never be matched to
/// one of its nodes, and everything version-keyed stays *not probed*.
pub fn capabilities_of(engine: NodeEngine, version: Option<&str>) -> Capabilities {
    match engine {
        NodeEngine::Camelid => Capabilities {
            load_reporting: Capability::declared(
                true,
                "`/v1/health` publishes jobs in flight and jobs waiting",
            ),
            typed_backpressure: Capability::declared(
                true,
                "a full queue answers 503 with the typed code `engine_queue_full`, which the fabric re-places",
            ),
            warm_prefix: Capability::declared(
                true,
                "one node owns one model and one session, so a sticky label really is a warm prefix",
            ),
            tool_calls: tool_calls_for(engine, version),
            embeddings: Capability::declared(true, "`/v1/embeddings` is served by the engine"),
            history_neutral: history_neutral_for(engine, version),
        },
        NodeEngine::Ollama => Capabilities {
            load_reporting: Capability::declared(
                false,
                "the API has no queue-depth or capacity endpoint",
            ),
            typed_backpressure: Capability::declared(
                false,
                "no typed queue-full refusal, so a busy node cannot be told from a broken one",
            ),
            warm_prefix: Capability::declared(
                false,
                "`/api/ps` says which models are resident, not whether a session's prefix survived",
            ),
            tool_calls: tool_calls_for(engine, version),
            embeddings: Capability::declared(true, "`/api/embed` is documented"),
            history_neutral: history_neutral_for(engine, version),
        },
        NodeEngine::LmStudio => Capabilities {
            load_reporting: Capability::declared(
                false,
                "the API has no queue-depth or capacity endpoint",
            ),
            typed_backpressure: Capability::declared(
                false,
                "no typed queue-full refusal, so a busy node cannot be told from a broken one",
            ),
            warm_prefix: Capability::declared(
                false,
                "model state is loaded or not loaded; nothing attests a session's prefix",
            ),
            tool_calls: tool_calls_for(engine, version),
            embeddings: Capability::declared(true, "`/api/v0/embeddings` is documented"),
            history_neutral: history_neutral_for(engine, version),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_answer_is_never_a_no() {
        // The whole point of the provenance: `not probed` and `no` must not be
        // the same value, or an unmeasured backend reads as a broken one.
        let unmeasured = capabilities_of(NodeEngine::Ollama, Some("99.0.0"));
        assert_eq!(unmeasured.tool_calls.supported, None);
        assert_eq!(unmeasured.tool_calls.provenance, Provenance::NotProbed);

        let measured = capabilities_of(NodeEngine::Ollama, Some("0.33.1"));
        assert_eq!(measured.tool_calls.supported, Some(false));
        assert_eq!(measured.tool_calls.provenance, Provenance::Measured);
    }

    #[test]
    fn a_measurement_is_about_the_exact_build_it_was_taken_on() {
        // 0.33.1 was measured; 0.33.3 is a different build and inherits nothing.
        assert_eq!(
            capabilities_of(NodeEngine::Ollama, Some("0.33.1"))
                .tool_calls
                .provenance,
            Provenance::Measured
        );
        assert_eq!(
            capabilities_of(NodeEngine::Ollama, Some("0.33.3"))
                .tool_calls
                .provenance,
            Provenance::NotProbed,
            "a neighbouring version inherits no measurement"
        );
    }

    #[test]
    fn an_engine_that_publishes_no_version_can_match_no_measurement() {
        // LM Studio has no version endpoint, so this is its permanent state
        // until one exists, and it must degrade to unknown rather than to no.
        let capabilities = capabilities_of(NodeEngine::LmStudio, None);
        assert_eq!(capabilities.tool_calls.supported, None);
        assert_eq!(capabilities.tool_calls.provenance, Provenance::NotProbed);
    }

    #[test]
    fn a_foreign_engine_is_never_credited_with_tool_calls_by_declaration() {
        for engine in [NodeEngine::Ollama, NodeEngine::LmStudio] {
            let capability = capabilities_of(engine, None).tool_calls;
            assert_ne!(
                capability.provenance,
                Provenance::Declared,
                "{engine} must not be credited from documentation"
            );
            assert_ne!(capability.supported, Some(true));
        }
    }

    #[test]
    fn every_unknown_carries_the_not_probed_provenance_and_nothing_else_does() {
        for engine in [
            NodeEngine::Camelid,
            NodeEngine::Ollama,
            NodeEngine::LmStudio,
        ] {
            for version in [None, Some("0.33.1"), Some("v0.6.1-267")] {
                for (name, capability) in capabilities_of(engine, version).entries() {
                    assert_eq!(
                        capability.supported.is_none(),
                        capability.provenance == Provenance::NotProbed,
                        "{engine}/{name}: unknown and not-probed must be the same state"
                    );
                    assert!(
                        !capability.detail.is_empty(),
                        "{engine}/{name} needs a reason"
                    );
                }
            }
        }
    }

    #[test]
    fn the_blockers_explain_exactly_why_an_engine_is_not_placed_on() {
        assert!(capabilities_of(NodeEngine::Camelid, None)
            .placement_blockers()
            .is_empty());
        for engine in [NodeEngine::Ollama, NodeEngine::LmStudio] {
            let blockers = capabilities_of(engine, None).placement_blockers();
            assert_eq!(
                blockers.len(),
                3,
                "{engine} is blocked on load, backpressure and warmth"
            );
        }
    }

    /// The seam's own invariant: placeability is not a hard-coded list of
    /// names, it is what the capabilities say.
    #[test]
    fn placeability_agrees_with_the_capabilities_that_justify_it() {
        for engine in [
            NodeEngine::Camelid,
            NodeEngine::Ollama,
            NodeEngine::LmStudio,
        ] {
            assert_eq!(
                engine.is_placeable(),
                capabilities_of(engine, None)
                    .placement_blockers()
                    .is_empty(),
                "{engine}: placeability and its stated reasons disagree"
            );
            assert_eq!(
                engine.reports_load(),
                capabilities_of(engine, None).load_reporting.supported == Some(true),
                "{engine}: load reporting and its capability disagree"
            );
        }
    }

    #[test]
    fn a_foreign_engine_is_never_credited_with_tool_calls_without_a_measurement() {
        // Placement depends on this. `policy::eligible` gates a tool-calling
        // request on `tool_calls.supported == Some(true)` alone, which is only
        // safe while a `true` cannot be reached from documentation. If this
        // ever fails, that gate has quietly become a vendor's word.
        let versions = [
            None,
            Some("0.33.1"),
            Some("0.33.3"),
            Some("1.0.0"),
            Some(""),
            Some("v0.6.1-267"),
        ];
        for engine in [NodeEngine::Ollama, NodeEngine::LmStudio] {
            for version in versions {
                let tools = capabilities_of(engine, version).tool_calls;
                if tools.supported == Some(true) {
                    assert_eq!(
                        tools.provenance,
                        Provenance::Measured,
                        "{engine} {version:?} was credited with tool calls without a measurement"
                    );
                }
            }
        }
    }

    /// C2. Neutrality across request history is never declared, for any
    /// engine, ours included: back-to-back agreement cannot show it, and the
    /// one measurement recorded is a counterexample on one exact version.
    #[test]
    fn history_neutrality_is_never_declared_and_only_measured_on_its_exact_version() {
        for engine in [
            NodeEngine::Camelid,
            NodeEngine::Ollama,
            NodeEngine::LmStudio,
        ] {
            for version in [None, Some("0.33.1"), Some("0.33.3"), Some("v0.7.3-1")] {
                let history = capabilities_of(engine, version).history_neutral;
                assert_eq!(
                    history.provenance,
                    Provenance::NotProbed,
                    "{engine} {version:?} must not be credited with neutrality"
                );
                assert_eq!(history.supported, None);
            }
        }
        let measured = capabilities_of(NodeEngine::Ollama, Some("0.33.2")).history_neutral;
        assert_eq!(measured.provenance, Provenance::Measured);
        assert_eq!(measured.supported, Some(false));
        assert_eq!(
            capabilities_of(NodeEngine::Camelid, Some("0.33.2"))
                .history_neutral
                .provenance,
            Provenance::NotProbed,
            "a measurement belongs to the engine it was taken on"
        );
    }

    #[test]
    fn our_own_engine_is_tool_capable_on_every_build_not_only_the_measured_one() {
        // Its tool-call contract is covered by its own test suite, so gating on
        // the measurement table would refuse every build that is not the single
        // version in it.
        for version in [None, Some("0.5.4"), Some("v0.6.1-267"), Some("v9.9.9")] {
            assert_eq!(
                capabilities_of(NodeEngine::Camelid, version)
                    .tool_calls
                    .supported,
                Some(true),
                "camelid {version:?}"
            );
        }
    }
}
