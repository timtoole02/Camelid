//! Operator-declared model identity across engines.
//!
//! Engines do not agree on what to call the same weights. Ollama suffixes
//! `:latest`; LM Studio does not; Camelid uses its catalog id. None of them
//! publishes a digest this fabric could compare — Ollama's `/api/tags` carries
//! a *manifest* digest, not the GGUF's, and LM Studio publishes none at all.
//!
//! So this build will never conclude on its own that two names mean the same
//! weights. Stripping a `:latest` suffix to make two ids match would be exactly
//! that inference, and it would be wrong the first time someone has two
//! genuinely different builds under similar names.
//!
//! What it does instead is let an operator say so, once, in the nodes file:
//!
//! ```text
//! alias llama-3.2-1b-instruct=studio:llama-3.2-1b-instruct:latest
//! alias llama-3.2-1b-instruct=desk:llama-3.2-1b-instruct
//! ```
//!
//! `fabric compare --model llama-3.2-1b-instruct` then asks each node for the
//! name that node knows. The claim is recorded as *asserted, not verified*
//! wherever it is used, because that is what it is.
//!
//! A per-invocation flag would have been enough for comparison, where a human
//! is present every time. It is not enough for placement, which has to answer
//! "which nodes serve this model" with nobody in the room — which is why this
//! is a stored table rather than an argument.

use std::collections::HashMap;

/// The `alias` keyword that introduces one of these lines in a nodes file.
pub(crate) const ALIAS_PREFIX: &str = "alias ";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AliasParseError {
    Malformed(String),
    EmptyField { field: &'static str, raw: String },
    Duplicate { canonical: String, label: String },
}

impl std::fmt::Display for AliasParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(raw) => write!(
                f,
                "`{raw}` is not a model alias; write `alias CANONICAL=LABEL:LOCAL`, \
                 for example `alias llama-3.2-1b-instruct=studio:llama-3.2-1b-instruct:latest`"
            ),
            Self::EmptyField { field, raw } => {
                write!(f, "`{raw}` has an empty {field}")
            }
            Self::Duplicate { canonical, label } => write!(
                f,
                "{label} already has a name declared for {canonical}; one node can only \
                 know one local name for a model"
            ),
        }
    }
}

impl std::error::Error for AliasParseError {}

/// What one node calls one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAlias {
    pub canonical: String,
    pub label: String,
    pub local: String,
}

/// Every declared name, keyed by the node it applies to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelAliases {
    by_node: HashMap<(String, String), String>,
}

impl ModelAliases {
    pub fn is_empty(&self) -> bool {
        self.by_node.is_empty()
    }

    pub fn len(&self) -> usize {
        self.by_node.len()
    }

    /// The name `label` knows `canonical` by.
    ///
    /// Falls back to `canonical` itself, so a fabric with no aliases behaves
    /// exactly as it did before this existed — and an operator only has to
    /// declare the names that actually differ.
    pub fn resolve<'a>(&'a self, label: &str, canonical: &'a str) -> &'a str {
        self.by_node
            .get(&(label.to_string(), canonical.to_string()))
            .map(String::as_str)
            .unwrap_or(canonical)
    }

    /// Whether a declaration was used to answer for this node, which is what
    /// makes a result rest on somebody's word rather than on a match.
    pub fn declares(&self, label: &str, canonical: &str) -> bool {
        self.by_node
            .contains_key(&(label.to_string(), canonical.to_string()))
    }

    /// [`Self::resolve`], plus what the answer rests on: the id as given, or
    /// an operator's word that a different id is the same weights.
    pub fn resolve_with_identity<'a>(
        &'a self,
        label: &str,
        canonical: &'a str,
    ) -> (&'a str, super::divergence::ModelIdentity) {
        let local = self.resolve(label, canonical);
        let identity = if local == canonical {
            super::divergence::ModelIdentity::SameId
        } else {
            super::divergence::ModelIdentity::AssertedByOperator
        };
        (local, identity)
    }

    /// Every declaration, ordered by canonical id then node, for listing what
    /// is in force.
    pub fn declared(&self) -> Vec<ModelAlias> {
        let mut declared: Vec<ModelAlias> = self
            .by_node
            .iter()
            .map(|((label, canonical), local)| ModelAlias {
                canonical: canonical.clone(),
                label: label.clone(),
                local: local.clone(),
            })
            .collect();
        declared.sort_by(|a, b| (&a.canonical, &a.label).cmp(&(&b.canonical, &b.label)));
        declared
    }

    pub fn insert(&mut self, alias: ModelAlias) -> Result<(), AliasParseError> {
        let key = (alias.label.clone(), alias.canonical.clone());
        if self.by_node.contains_key(&key) {
            return Err(AliasParseError::Duplicate {
                canonical: alias.canonical,
                label: alias.label,
            });
        }
        self.by_node.insert(key, alias.local);
        Ok(())
    }
}

/// Parse `CANONICAL=LABEL:LOCAL`, with or without the leading `alias `.
///
/// `LOCAL` is everything after the first colon because a model id may contain
/// colons — `llama-3.2-1b-instruct:latest` is the case this whole module
/// exists for. A label may not.
pub fn parse_model_alias(raw: &str) -> Result<ModelAlias, AliasParseError> {
    let body = raw.trim().strip_prefix(ALIAS_PREFIX).unwrap_or(raw.trim());

    let (canonical, rest) = body
        .split_once('=')
        .ok_or_else(|| AliasParseError::Malformed(raw.trim().to_string()))?;
    let (label, local) = rest
        .split_once(':')
        .ok_or_else(|| AliasParseError::Malformed(raw.trim().to_string()))?;

    let alias = ModelAlias {
        canonical: canonical.trim().to_string(),
        label: label.trim().to_string(),
        local: local.trim().to_string(),
    };
    for (field, value) in [
        ("canonical model id", &alias.canonical),
        ("node label", &alias.label),
        ("local model id", &alias.local),
    ] {
        if value.is_empty() {
            return Err(AliasParseError::EmptyField {
                field,
                raw: raw.trim().to_string(),
            });
        }
    }
    Ok(alias)
}

/// Build a table from whole lines, each `CANONICAL=LABEL:LOCAL`.
pub fn parse_model_aliases(raws: &[String]) -> Result<ModelAliases, AliasParseError> {
    let mut aliases = ModelAliases::default();
    for raw in raws {
        aliases.insert(parse_model_alias(raw)?)?;
    }
    Ok(aliases)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(lines: &[&str]) -> ModelAliases {
        parse_model_aliases(&lines.iter().map(|l| l.to_string()).collect::<Vec<_>>())
            .expect("parses")
    }

    #[test]
    fn a_local_id_may_contain_colons_because_that_is_the_whole_problem() {
        let alias = parse_model_alias("llama-3.2-1b-instruct=studio:llama-3.2-1b-instruct:latest")
            .expect("parses");
        assert_eq!(alias.canonical, "llama-3.2-1b-instruct");
        assert_eq!(alias.label, "studio");
        assert_eq!(
            alias.local, "llama-3.2-1b-instruct:latest",
            "everything after the first colon is the model's own name"
        );
    }

    #[test]
    fn the_alias_keyword_is_accepted_so_a_nodes_file_line_parses_unchanged() {
        assert_eq!(
            parse_model_alias("alias m=studio:m:latest").expect("parses"),
            parse_model_alias("m=studio:m:latest").expect("parses")
        );
    }

    #[test]
    fn a_fabric_with_no_aliases_answers_exactly_what_it_was_asked() {
        // The no-declaration path has to be identical to the behaviour before
        // this module existed, or adding it would change every existing setup.
        let empty = ModelAliases::default();
        assert!(empty.is_empty());
        assert_eq!(empty.resolve("studio", "qwen3:8b"), "qwen3:8b");
        assert!(!empty.declares("studio", "qwen3:8b"));
    }

    #[test]
    fn a_declaration_answers_only_for_the_node_it_names() {
        let aliases = table(&["llama-3.2-1b-instruct=studio:llama-3.2-1b-instruct:latest"]);
        assert_eq!(
            aliases.resolve("studio", "llama-3.2-1b-instruct"),
            "llama-3.2-1b-instruct:latest"
        );
        // A node with no declaration is asked for the canonical name, not for
        // some other node's local one.
        assert_eq!(
            aliases.resolve("desk", "llama-3.2-1b-instruct"),
            "llama-3.2-1b-instruct"
        );
        assert!(aliases.declares("studio", "llama-3.2-1b-instruct"));
        assert!(!aliases.declares("desk", "llama-3.2-1b-instruct"));
    }

    #[test]
    fn one_node_may_declare_names_for_several_models() {
        let aliases = table(&["a=studio:a:latest", "b=studio:b:latest"]);
        assert_eq!(aliases.len(), 2);
        assert_eq!(aliases.resolve("studio", "a"), "a:latest");
        assert_eq!(aliases.resolve("studio", "b"), "b:latest");
    }

    #[test]
    fn two_names_for_one_model_on_one_node_is_refused_rather_than_last_wins() {
        let error =
            parse_model_aliases(&["m=studio:first".to_string(), "m=studio:second".to_string()])
                .expect_err("a node cannot know one model by two names");
        assert_eq!(
            error,
            AliasParseError::Duplicate {
                canonical: "m".to_string(),
                label: "studio".to_string()
            }
        );
    }

    #[test]
    fn a_malformed_line_says_what_the_shape_is() {
        for raw in ["nonsense", "m=studio", "m:studio:local"] {
            let message = parse_model_alias(raw).expect_err(raw).to_string();
            assert!(
                message.contains("alias CANONICAL=LABEL:LOCAL"),
                "{raw} -> {message}"
            );
        }
    }

    #[test]
    fn an_empty_field_names_which_one_is_empty() {
        for (raw, field) in [
            ("=studio:local", "canonical model id"),
            ("m=:local", "node label"),
            ("m=studio:", "local model id"),
        ] {
            let message = parse_model_alias(raw).expect_err(raw).to_string();
            assert!(message.contains(field), "{raw} -> {message}");
        }
    }

    #[test]
    fn surrounding_whitespace_in_a_file_line_is_not_part_of_a_model_id() {
        let alias = parse_model_alias("  alias  m = studio : local  ").expect("parses");
        assert_eq!(alias.canonical, "m");
        assert_eq!(alias.label, "studio");
        assert_eq!(alias.local, "local");
    }
}
