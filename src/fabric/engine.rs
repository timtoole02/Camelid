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
