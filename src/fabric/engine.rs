//! Which inference engine a node runs, and what that engine can be asked.
//!
//! A fabric used to be Camelid nodes only, so the engine was implied. It is now
//! declared, because the alternative is to guess: probing an address and
//! inferring the engine from what comes back would make a typo look like a
//! different product, and would send a Camelid bearer token to whatever answered.
//!
//! The rule this module exists to hold: **an engine declares what it cannot
//! answer.** Ollama publishes no queue depth, so a node running it reports no
//! load at all rather than a plausible zero, and placement is told that rather
//! than left to infer it from a number that was never measured.

use std::fmt;

use serde::{Serialize, Serializer};

use super::identify::{Answers, EngineVerdict};

/// An inference engine the fabric knows how to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeEngine {
    /// A `camelid serve` process. The only engine this fabric places on today.
    #[default]
    Camelid,
    /// An Ollama server. Read and reported, never placed on — see
    /// [`NodeEngine::is_placeable`].
    Ollama,
    /// An LM Studio server. Read and reported, never placed on.
    LmStudio,
}

impl NodeEngine {
    /// The scheme an operator writes in a node specification, and the value the
    /// engine is reported under in health and status output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Camelid => "camelid",
            Self::Ollama => "ollama",
            Self::LmStudio => "lmstudio",
        }
    }

    /// Parse a scheme from a node specification. `None` for anything else, so an
    /// unknown engine is refused with the list of known ones rather than
    /// silently treated as Camelid.
    pub fn from_scheme(scheme: &str) -> Option<Self> {
        match scheme {
            "camelid" => Some(Self::Camelid),
            "ollama" => Some(Self::Ollama),
            "lmstudio" => Some(Self::LmStudio),
            _ => None,
        }
    }

    /// Every engine an operator may name, for error messages.
    pub fn known() -> &'static [&'static str] {
        &["camelid", "ollama", "lmstudio"]
    }

    /// Every engine this build knows, in the order they are reported.
    ///
    /// Discovery iterates this rather than listing engines of its own, which is
    /// what keeps a fourth engine a new file plus a row here (S6).
    pub const ALL: [NodeEngine; 3] = [Self::Camelid, Self::Ollama, Self::LmStudio];

    /// The paths that identify this engine, asked of a machine nobody has
    /// declared yet.
    ///
    /// Separate from what [`super::probe`] reads, and answered by a stricter
    /// signature: a probe reads a node an operator already named and may be
    /// lenient, while this decides *which* engine an address is, where the
    /// same leniency would match everything at once.
    pub(crate) fn identification_paths(self) -> &'static [&'static str] {
        match self {
            Self::Camelid => super::camelid::IDENTIFICATION_PATHS,
            Self::Ollama => super::ollama::IDENTIFICATION_PATHS,
            Self::LmStudio => super::lmstudio::IDENTIFICATION_PATHS,
        }
    }

    /// Whether the answers from one address match this engine's signature.
    pub(crate) fn identify(self, answers: &Answers<'_>) -> EngineVerdict {
        match self {
            Self::Camelid => super::camelid::identify(answers),
            Self::Ollama => super::ollama::identify(answers),
            Self::LmStudio => super::lmstudio::identify(answers),
        }
    }

    /// Whether a node declared as this engine is ever shown the fabric's
    /// bearer, once it is in the file.
    ///
    /// The same answer [`Self::fabric_bearer`] gives, as a fact rather than a
    /// token: discovery holds no credential at all, and still has to be able to
    /// warn a person what joining a machine will mean.
    pub(crate) fn receives_fabric_bearer(self) -> bool {
        self.fabric_bearer(Some("")).is_some()
    }

    /// The port to assume when a specification names no port.
    pub fn default_port(self) -> u16 {
        match self {
            Self::Camelid => 8181,
            Self::Ollama => 11434,
            Self::LmStudio => 1234,
        }
    }

    /// What this engine can be asked, given the version it reported.
    pub fn capabilities(self, version: Option<&str>) -> super::capability::Capabilities {
        super::capability::capabilities_of(self, version)
    }

    /// Whether this fabric may place a request on the engine.
    ///
    /// Not a list of names: an engine is placeable exactly when its capabilities
    /// leave nothing blocking. Placement ranks on reported load, re-places a
    /// typed queue-full refusal, and honours session affinity — an engine that
    /// answers none of those three would mean ranking a node whose load was
    /// never measured, relaying a refusal that might have been retried, and
    /// claiming a warm prefix nobody attested. `capabilities().placement_blockers()`
    /// names which of those is missing.
    pub fn is_placeable(self) -> bool {
        matches!(self, Self::Camelid)
    }

    /// Whether the engine publishes a load figure placement could rank on.
    pub fn reports_load(self) -> bool {
        matches!(self, Self::Camelid)
    }

    /// The fabric's bearer, where this engine may be shown it at all.
    ///
    /// That token is a Camelid API key. Any other engine has no place for it,
    /// and presenting it hands this fabric's credential to a foreign process —
    /// which mixed placement must never make possible, however the token was
    /// configured and whichever path the request took.
    pub(crate) fn fabric_bearer(self, bearer: Option<&str>) -> Option<&str> {
        bearer.filter(|_| matches!(self, Self::Camelid))
    }

    /// Which bearer-receiving warning a proposal for this engine carries, given
    /// whether the proxy that would probe it holds a bearer at all.
    ///
    /// Both halves are facts: one about the engine, one about the process. A
    /// warning from either alone would be a claim about the other.
    pub(crate) fn bearer_warning(self, bearer_configured: Option<bool>) -> Option<&'static str> {
        if !self.receives_fabric_bearer() {
            return None;
        }
        match bearer_configured {
            Some(true) => Some("bearer_will_be_sent"),
            Some(false) => None,
            // The CLI never reads a key, so it cannot know. It says so.
            None => Some("bearer_sent_if_configured"),
        }
    }
}

impl fmt::Display for NodeEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for NodeEngine {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_known_scheme_parses_and_round_trips() {
        for name in NodeEngine::known() {
            let engine = NodeEngine::from_scheme(name).expect("known scheme parses");
            assert_eq!(engine.as_str(), *name);
        }
    }

    #[test]
    fn an_unknown_engine_is_refused_rather_than_assumed_to_be_camelid() {
        assert_eq!(NodeEngine::from_scheme("vllm"), None);
        assert_eq!(NodeEngine::from_scheme("CAMELID"), None);
        assert_eq!(NodeEngine::from_scheme(""), None);
    }

    #[test]
    fn only_camelid_is_placeable_and_only_camelid_reports_load() {
        assert!(NodeEngine::Camelid.is_placeable());
        assert!(NodeEngine::Camelid.reports_load());
        for foreign in [NodeEngine::Ollama, NodeEngine::LmStudio] {
            assert!(!foreign.is_placeable());
            assert!(!foreign.reports_load());
        }
    }

    #[test]
    fn only_our_own_engine_is_ever_shown_the_fabric_bearer() {
        assert_eq!(NodeEngine::Camelid.fabric_bearer(Some("k")), Some("k"));
        assert_eq!(NodeEngine::Camelid.fabric_bearer(None), None);
        for foreign in [NodeEngine::Ollama, NodeEngine::LmStudio] {
            assert_eq!(foreign.fabric_bearer(Some("k")), None, "{foreign}");
        }
    }

    /// `ALL` is what discovery iterates. If it ever falls behind `known()`, a
    /// scan silently stops looking for one of this build's own engines.
    #[test]
    fn every_engine_this_build_knows_is_one_discovery_looks_for() {
        let named: Vec<&str> = NodeEngine::ALL.iter().map(|engine| engine.as_str()).collect();
        assert_eq!(named, NodeEngine::known());
        for engine in NodeEngine::ALL {
            assert!(
                !engine.identification_paths().is_empty(),
                "{engine} has no way to be recognised"
            );
        }
    }

    /// The warning a person is shown before joining a machine has to follow the
    /// same rule the probe follows afterwards, or it becomes false the moment
    /// one of them changes.
    #[test]
    fn only_the_engine_that_is_shown_the_bearer_warns_about_it() {
        assert_eq!(
            NodeEngine::Camelid.bearer_warning(Some(true)),
            Some("bearer_will_be_sent")
        );
        assert_eq!(NodeEngine::Camelid.bearer_warning(Some(false)), None);
        assert_eq!(
            NodeEngine::Camelid.bearer_warning(None),
            Some("bearer_sent_if_configured"),
            "a caller that never reads a key must not claim one is configured"
        );
        for foreign in [NodeEngine::Ollama, NodeEngine::LmStudio] {
            assert!(!foreign.receives_fabric_bearer(), "{foreign}");
            for configured in [Some(true), Some(false), None] {
                assert_eq!(foreign.bearer_warning(configured), None, "{foreign}");
            }
        }
    }

    #[test]
    fn each_engine_carries_its_own_default_port() {
        assert_eq!(NodeEngine::Camelid.default_port(), 8181);
        assert_eq!(NodeEngine::Ollama.default_port(), 11434);
        assert_eq!(NodeEngine::LmStudio.default_port(), 1234);
    }

    #[test]
    fn every_engine_can_say_why_it_is_or_is_not_placed_on() {
        for name in NodeEngine::known() {
            let engine = NodeEngine::from_scheme(name).expect("known scheme parses");
            let blockers = engine.capabilities(None).placement_blockers();
            assert_eq!(
                engine.is_placeable(),
                blockers.is_empty(),
                "{name}: placeability must be what the capabilities say"
            );
        }
    }
}
