//! Divergence: the same prompt on two nodes, and an honest account of what came
//! back.
//!
//! This is the part of the fabric that exists because of a measured fact: an
//! identical GGUF, asked *"What is 7 plus 5?"* greedily, answers **12** on
//! Camelid and **7** on llama.cpp / Ollama / LM Studio, because the backends
//! apply different chat templates. Mixed-engine routing makes that our problem;
//! this module makes it visible instead of deniable.
//!
//! Four rules do the real work, and every one of them exists to stop a
//! comparison from claiming more than it measured:
//!
//! 1. **A difference is only attributable when each side agrees with itself.**
//!    One run per side proves nothing: engines are not obliged to be
//!    deterministic, and a backend that varies run to run would otherwise be
//!    reported as diverging from the other one. Each side is run repeatedly and
//!    a cross-side verdict is withheld unless both sides are self-consistent.
//! 2. **What we asked for is not what was applied.** LM Studio's documented
//!    completion API has no seed parameter, so a comparison against it is not
//!    seeded no matter what the operator typed. The plan and the per-side
//!    reality are recorded separately.
//! 3. **Different weights are not a divergence.** Two nodes serving different
//!    model identities can differ for the most boring reason there is. That is
//!    reported as its own verdict, not as evidence about templates.
//! 4. **Never say which side is right.** This module reports difference,
//!    provenance and reproducibility. Correctness is not ours to award.

use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::engine::NodeEngine;
use super::textdiff::{diff_lines, Diff};

/// One question, fully specified: what to ask and how to sample it.
///
/// Bundled rather than passed as a parameter list because every engine adapter
/// needs exactly this set, and a positional `f32, Option<u64>, u32` tail is the
/// kind of signature where a temperature and a token count eventually get
/// swapped without the compiler noticing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ask<'a> {
    pub(crate) model: &'a str,
    pub(crate) prompt: &'a str,
    pub(crate) temperature: f32,
    pub(crate) seed: Option<u64>,
    pub(crate) max_tokens: u32,
}

/// What the operator asked for. Recorded verbatim so a receipt can be replayed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SamplingPlan {
    pub temperature: f32,
    pub seed: Option<u64>,
    pub max_tokens: u32,
    /// How many times each side is run. One is permitted, and is why
    /// [`Stability::Unmeasured`] exists.
    pub repetitions: usize,
}

impl Default for SamplingPlan {
    fn default() -> Self {
        // Greedy, seeded, short. The default is the setting under which the
        // §4 template divergence was originally measured.
        Self {
            temperature: 0.0,
            seed: Some(0),
            max_tokens: 64,
            repetitions: 2,
        }
    }
}

/// Whether a requested sampling control actually reached the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Honoured {
    /// The engine's documented API takes this parameter and we sent it.
    Sent,
    /// The engine's documented API has no such parameter. Not a failure — but
    /// it must not be reported as a controlled variable.
    Unsupported,
}

/// What sampling actually looked like for one side.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AppliedSampling {
    pub temperature: Honoured,
    pub seed: Honoured,
}

impl AppliedSampling {
    pub fn for_engine(engine: NodeEngine) -> Self {
        match engine {
            // `temperature` and `seed` are both fields of our own
            // `ChatCompletionRequest`.
            NodeEngine::Camelid => Self {
                temperature: Honoured::Sent,
                seed: Honoured::Sent,
            },
            // `POST /api/chat` takes `options.temperature` and `options.seed`.
            NodeEngine::Ollama => Self {
                temperature: Honoured::Sent,
                seed: Honoured::Sent,
            },
            // `POST /api/v0/chat/completions` documents `temperature`; it
            // documents no seed parameter, so the run is unseeded.
            NodeEngine::LmStudio => Self {
                temperature: Honoured::Sent,
                seed: Honoured::Unsupported,
            },
        }
    }

    /// True when every control the plan relies on reached the engine.
    pub fn fully_controlled(&self) -> bool {
        self.temperature == Honoured::Sent && self.seed == Honoured::Sent
    }
}

/// One answer from one run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sample {
    pub text: String,
    pub sha256: String,
    #[serde(rename = "elapsed_ms")]
    pub elapsed_ms: u128,
}

impl Sample {
    pub fn new(text: String, elapsed: Duration) -> Self {
        let sha256 = sha256_hex(text.as_bytes());
        Self {
            text,
            sha256,
            elapsed_ms: elapsed.as_millis(),
        }
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut acc, byte| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{byte:02x}");
            acc
        })
}

/// Whether one side agreed with itself across its own repetitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Stability {
    /// Every repetition produced the same bytes.
    Stable,
    /// Repetitions disagreed. Nothing can be attributed to the *other* side
    /// while this is true, so the distinct digests are carried for the report.
    Unstable { digests: Vec<String> },
    /// Only one run was requested, so self-consistency was never tested. This
    /// is not stability; it is the absence of the measurement.
    Unmeasured,
}

/// The template a backend applied, where it can be obtained at all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TemplateEvidence {
    Captured {
        source: String,
        template: String,
    },
    /// The engine exposes no way to ask. Distinct from "asked and got nothing".
    NotExposed {
        detail: String,
    },
    /// We asked and the request failed.
    Unavailable {
        detail: String,
    },
}

/// Everything measured about one node in one comparison.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Side {
    pub label: String,
    pub engine: NodeEngine,
    pub engine_version: Option<String>,
    /// The inference runtime the engine named, where it names one. Kept apart
    /// from `engine_version` on purpose: LM Studio publishes no application
    /// version but does name a llama.cpp build, and reporting that as its
    /// version would be inventing the fact P3 refused to invent.
    pub runtime: Option<String>,
    /// The model the node reported serving, which is not necessarily the model
    /// the operator named.
    pub model: Option<String>,
    pub applied_sampling: AppliedSampling,
    pub samples: Vec<Sample>,
    pub stability: Stability,
    pub template: TemplateEvidence,
}

impl Side {
    /// The digest this side is represented by, which only exists when the side
    /// agreed with itself.
    pub fn settled_digest(&self) -> Option<&str> {
        match self.stability {
            Stability::Stable => self.samples.first().map(|sample| sample.sha256.as_str()),
            _ => None,
        }
    }

    pub fn settled_text(&self) -> Option<&str> {
        match self.stability {
            Stability::Stable => self.samples.first().map(|sample| sample.text.as_str()),
            _ => None,
        }
    }
}

/// Decide whether a side agreed with itself.
pub fn stability_of(samples: &[Sample]) -> Stability {
    match samples.len() {
        0 | 1 => Stability::Unmeasured,
        _ => {
            let mut digests: Vec<String> =
                samples.iter().map(|sample| sample.sha256.clone()).collect();
            digests.sort();
            digests.dedup();
            if digests.len() == 1 {
                Stability::Stable
            } else {
                Stability::Unstable { digests }
            }
        }
    }
}

/// What the comparison is entitled to conclude.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Verdict {
    /// Both sides settled on the same bytes.
    Identical,
    /// Both sides settled, on different bytes. The only verdict that is
    /// evidence about the engines.
    Divergent,
    /// The two nodes are not serving the same model identity, so a difference
    /// says nothing about the backends. Carries both names.
    DifferentModels { left: String, right: String },
    /// At least one side did not agree with itself, so a cross-side difference
    /// cannot be attributed to the engines.
    NotAttributable { reason: String },
}

impl Verdict {
    /// Whether this verdict supports a claim about the engines. Deliberately
    /// false for `DifferentModels` and `NotAttributable`.
    pub fn is_attributable(&self) -> bool {
        matches!(self, Self::Identical | Self::Divergent)
    }
}

/// The whole comparison, in the shape it is reported and exported.
/// How the two sides came to be treated as the same model.
///
/// Engines do not agree on how to name weights: Ollama suffixes `:latest`, LM
/// Studio does not, and neither publishes a digest this fabric can compare. So
/// an exactly-equal id is the only thing this build will *conclude* on its own,
/// and anything else has to be a human saying so out loud.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelIdentity {
    /// Both sides were asked for the same id.
    SameId,
    /// The ids differ and an operator stated they are the same weights. Not
    /// verified, and never inferred — it travels with the receipt so a reader
    /// knows the comparison rests on someone's word.
    AssertedByOperator,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Comparison {
    pub prompt: String,
    pub prompt_sha256: String,
    pub plan: SamplingPlan,
    pub left: Side,
    pub right: Side,
    pub verdict: Verdict,
    pub diff: Diff,
    pub model_identity: ModelIdentity,
    /// Controls the plan asked for that at least one side could not honour.
    /// Empty when both sides were fully controlled.
    pub uncontrolled: Vec<String>,
}

/// Build the verdict and diff from two completed sides.
///
/// Ordering matters and is the point of the function: identity of the weights
/// is checked before stability, and stability before any comparison of bytes.
/// A caller that reordered these would start reporting model swaps as template
/// divergence.
pub fn conclude(
    prompt: &str,
    plan: SamplingPlan,
    left: Side,
    right: Side,
    model_identity: ModelIdentity,
) -> Comparison {
    let verdict = verdict_for(&left, &right, model_identity);
    let diff = match (&verdict, left.settled_text(), right.settled_text()) {
        (Verdict::Divergent, Some(l), Some(r)) => diff_lines(l, r),
        (Verdict::Identical, _, _) => Diff::Identical,
        // Anything unattributable gets no diff: rendering one would invite
        // exactly the reading the verdict just refused.
        _ => Diff::Declined {
            reason: "the two sides are not comparable, so no diff is shown".to_string(),
        },
    };

    let mut uncontrolled = Vec::new();
    if left.applied_sampling.seed == Honoured::Unsupported
        || right.applied_sampling.seed == Honoured::Unsupported
    {
        uncontrolled.push("seed".to_string());
    }
    if left.applied_sampling.temperature == Honoured::Unsupported
        || right.applied_sampling.temperature == Honoured::Unsupported
    {
        uncontrolled.push("temperature".to_string());
    }
    // The biggest uncontrolled variable of all when it applies: nobody checked
    // that these are the same weights.
    if model_identity == ModelIdentity::AssertedByOperator {
        uncontrolled.push("model identity".to_string());
    }

    Comparison {
        prompt: prompt.to_string(),
        prompt_sha256: sha256_hex(prompt.as_bytes()),
        plan,
        left,
        right,
        verdict,
        diff,
        model_identity,
        uncontrolled,
    }
}

fn verdict_for(left: &Side, right: &Side, model_identity: ModelIdentity) -> Verdict {
    if model_identity == ModelIdentity::SameId {
        if let (Some(l), Some(r)) = (left.model.as_deref(), right.model.as_deref()) {
            if l != r {
                return Verdict::DifferentModels {
                    left: l.to_string(),
                    right: r.to_string(),
                };
            }
        }
    }
    match (&left.stability, &right.stability) {
        (Stability::Stable, Stability::Stable) => {
            if left.settled_digest() == right.settled_digest() {
                Verdict::Identical
            } else {
                Verdict::Divergent
            }
        }
        (Stability::Unmeasured, _) | (_, Stability::Unmeasured) => Verdict::NotAttributable {
            reason: "each side was run once, so neither was shown to agree with itself"
                .to_string(),
        },
        (Stability::Unstable { .. }, Stability::Unstable { .. }) => Verdict::NotAttributable {
            reason: "neither side repeated its own answer, so a difference between them says nothing about the engines"
                .to_string(),
        },
        (Stability::Unstable { .. }, _) => Verdict::NotAttributable {
            reason: format!(
                "{} did not repeat its own answer, so a difference from {} cannot be attributed to the engines",
                left.label, right.label
            ),
        },
        (_, Stability::Unstable { .. }) => Verdict::NotAttributable {
            reason: format!(
                "{} did not repeat its own answer, so a difference from {} cannot be attributed to the engines",
                right.label, left.label
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(text: &str) -> Sample {
        Sample::new(text.to_string(), Duration::from_millis(1))
    }

    fn side(label: &str, engine: NodeEngine, model: &str, answers: &[&str]) -> Side {
        let samples: Vec<Sample> = answers.iter().map(|text| sample(text)).collect();
        Side {
            label: label.to_string(),
            engine,
            engine_version: None,
            runtime: None,
            model: Some(model.to_string()),
            applied_sampling: AppliedSampling::for_engine(engine),
            stability: stability_of(&samples),
            samples,
            template: TemplateEvidence::NotExposed {
                detail: "test".to_string(),
            },
        }
    }

    #[test]
    fn a_known_digest_is_reproduced_exactly() {
        // Pins the hash function itself: the whole verdict rests on it, and a
        // silent change of algorithm would invalidate every stored receipt.
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn one_run_a_side_is_unmeasured_rather_than_stable() {
        assert_eq!(stability_of(&[]), Stability::Unmeasured);
        assert_eq!(stability_of(&[sample("a")]), Stability::Unmeasured);
    }

    #[test]
    fn a_side_that_repeated_itself_is_stable_and_one_that_did_not_is_not() {
        assert_eq!(stability_of(&[sample("a"), sample("a")]), Stability::Stable);
        match stability_of(&[sample("a"), sample("b"), sample("a")]) {
            Stability::Unstable { digests } => {
                assert_eq!(digests.len(), 2, "distinct digests, not repetitions");
            }
            other => panic!("expected instability, got {other:?}"),
        }
    }

    #[test]
    fn the_measured_template_divergence_is_reported_as_divergent_and_diffed() {
        let left = side("win", NodeEngine::Camelid, "llama-3.2-1b", &["12", "12"]);
        let right = side("studio", NodeEngine::Ollama, "llama-3.2-1b", &["7", "7"]);
        let comparison = conclude(
            "What is 7 plus 5?",
            SamplingPlan::default(),
            left,
            right,
            ModelIdentity::SameId,
        );

        assert_eq!(comparison.verdict, Verdict::Divergent);
        assert!(comparison.verdict.is_attributable());
        assert_eq!(comparison.diff.changed_lines(), 2);
        assert_eq!(
            comparison.prompt_sha256,
            sha256_hex(b"What is 7 plus 5?"),
            "the prompt is hashed so a receipt names the exact question asked"
        );
    }

    #[test]
    fn two_sides_that_agree_are_identical_with_nothing_to_diff() {
        let left = side("win", NodeEngine::Camelid, "m", &["12", "12"]);
        let right = side("mac", NodeEngine::Camelid, "m", &["12", "12"]);
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            left,
            right,
            ModelIdentity::SameId,
        );
        assert_eq!(comparison.verdict, Verdict::Identical);
        assert_eq!(comparison.diff, Diff::Identical);
    }

    #[test]
    fn an_unstable_side_makes_a_difference_unattributable_and_suppresses_the_diff() {
        // The heart of the module: without this, a backend that simply is not
        // deterministic would be reported as diverging from the other one.
        let left = side("win", NodeEngine::Camelid, "m", &["12", "12"]);
        let right = side("studio", NodeEngine::Ollama, "m", &["7", "seven"]);
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            left,
            right,
            ModelIdentity::SameId,
        );

        match &comparison.verdict {
            Verdict::NotAttributable { reason } => {
                assert!(
                    reason.contains("studio"),
                    "the unstable side is named: {reason}"
                );
            }
            other => panic!("expected no attribution, got {other:?}"),
        }
        assert!(!comparison.verdict.is_attributable());
        assert!(
            matches!(comparison.diff, Diff::Declined { .. }),
            "a diff here would invite exactly the reading the verdict refused"
        );
    }

    #[test]
    fn a_single_repetition_yields_no_attribution_however_different_the_answers() {
        let left = side("win", NodeEngine::Camelid, "m", &["12"]);
        let right = side("studio", NodeEngine::Ollama, "m", &["7"]);
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            left,
            right,
            ModelIdentity::SameId,
        );
        match &comparison.verdict {
            Verdict::NotAttributable { reason } => assert!(reason.contains("once"), "{reason}"),
            other => panic!("expected no attribution, got {other:?}"),
        }
    }

    #[test]
    fn different_model_identities_are_reported_as_such_and_never_as_divergence() {
        // Otherwise the view's headline finding could be produced by pointing
        // it at two different sets of weights.
        let left = side("win", NodeEngine::Camelid, "llama-3.2-1b", &["12", "12"]);
        let right = side("studio", NodeEngine::Ollama, "qwen3:8b", &["7", "7"]);
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            left,
            right,
            ModelIdentity::SameId,
        );

        assert_eq!(
            comparison.verdict,
            Verdict::DifferentModels {
                left: "llama-3.2-1b".to_string(),
                right: "qwen3:8b".to_string()
            }
        );
        assert!(!comparison.verdict.is_attributable());
    }

    #[test]
    fn model_identity_is_checked_before_stability_so_a_swap_is_never_read_as_flakiness() {
        let left = side("win", NodeEngine::Camelid, "a", &["12", "12"]);
        let right = side("studio", NodeEngine::Ollama, "b", &["7", "seven"]);
        assert!(matches!(
            conclude(
                "q",
                SamplingPlan::default(),
                left,
                right,
                ModelIdentity::SameId
            )
            .verdict,
            Verdict::DifferentModels { .. }
        ));
    }

    #[test]
    fn an_operator_may_assert_two_ids_are_the_same_weights_and_it_is_recorded_as_unverified() {
        // Ollama suffixes `:latest` and LM Studio does not, so a cross-engine
        // comparison is impossible unless a human says the two names mean the
        // same thing. That assertion is honoured and then disclosed as the
        // uncontrolled variable it is.
        let left = side(
            "studio",
            NodeEngine::Ollama,
            "llama-3.2-1b-instruct:latest",
            &["12", "12"],
        );
        let right = side(
            "desk",
            NodeEngine::LmStudio,
            "llama-3.2-1b-instruct",
            &["7", "7"],
        );
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            left,
            right,
            ModelIdentity::AssertedByOperator,
        );

        assert_eq!(comparison.verdict, Verdict::Divergent);
        assert_eq!(comparison.model_identity, ModelIdentity::AssertedByOperator);
        assert!(
            comparison
                .uncontrolled
                .contains(&"model identity".to_string()),
            "an unverified equivalence is the biggest uncontrolled variable there is: {:?}",
            comparison.uncontrolled
        );
    }

    #[test]
    fn without_that_assertion_two_ids_that_differ_are_still_refused() {
        let left = side(
            "studio",
            NodeEngine::Ollama,
            "llama-3.2-1b-instruct:latest",
            &["12", "12"],
        );
        let right = side(
            "desk",
            NodeEngine::LmStudio,
            "llama-3.2-1b-instruct",
            &["7", "7"],
        );
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            left,
            right,
            ModelIdentity::SameId,
        );
        assert!(matches!(
            comparison.verdict,
            Verdict::DifferentModels { .. }
        ));
        assert!(!comparison
            .uncontrolled
            .contains(&"model identity".to_string()));
    }

    #[test]
    fn an_engine_without_a_seed_parameter_is_recorded_as_uncontrolled() {
        let applied = AppliedSampling::for_engine(NodeEngine::LmStudio);
        assert_eq!(applied.seed, Honoured::Unsupported);
        assert!(!applied.fully_controlled());

        let left = side("win", NodeEngine::Camelid, "m", &["12", "12"]);
        let right = side("desk", NodeEngine::LmStudio, "m", &["7", "7"]);
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            left,
            right,
            ModelIdentity::SameId,
        );
        assert_eq!(comparison.uncontrolled, ["seed"]);
        assert_eq!(
            comparison.verdict,
            Verdict::Divergent,
            "an uncontrolled seed is disclosed, not a reason to withhold a verdict both sides earned"
        );
    }

    #[test]
    fn our_own_engine_claims_no_more_control_than_its_request_type_offers() {
        assert!(AppliedSampling::for_engine(NodeEngine::Camelid).fully_controlled());
        assert!(AppliedSampling::for_engine(NodeEngine::Ollama).fully_controlled());
    }

    #[test]
    fn no_verdict_ever_names_a_winner() {
        // The product rule, asserted rather than trusted to review. Only the
        // prose a human reads is scanned: `left`/`right` are positions in a
        // comparison, not judgements, and appear as field names.
        let cases = [
            Verdict::Identical,
            Verdict::Divergent,
            Verdict::DifferentModels {
                left: "a".into(),
                right: "b".into(),
            },
            Verdict::NotAttributable { reason: "x".into() },
        ];
        for verdict in cases {
            let rendered = serde_json::to_value(&verdict).expect("serialises");
            let prose = [
                rendered
                    .get("kind")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
                rendered
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
            ]
            .join(" ");
            for banned in [
                "correct",
                "incorrect",
                "wrong",
                "better",
                "worse",
                "accurate",
                "winner",
                "expected",
            ] {
                assert!(
                    !prose.contains(banned),
                    "a verdict must not judge correctness, found {banned:?} in {prose:?}"
                );
            }
        }
    }

    #[test]
    fn no_reason_this_module_can_produce_judges_a_side() {
        // The verdicts above are hand-built; these are the ones the code
        // actually emits, which is what a user would read.
        let unstable = side("studio", NodeEngine::Ollama, "m", &["7", "seven"]);
        let stable = side("win", NodeEngine::Camelid, "m", &["12", "12"]);
        let once = side("solo", NodeEngine::Camelid, "m", &["12"]);
        let produced = [
            conclude(
                "q",
                SamplingPlan::default(),
                stable.clone(),
                unstable.clone(),
                ModelIdentity::SameId,
            )
            .verdict,
            conclude(
                "q",
                SamplingPlan::default(),
                unstable.clone(),
                stable.clone(),
                ModelIdentity::SameId,
            )
            .verdict,
            conclude(
                "q",
                SamplingPlan::default(),
                unstable.clone(),
                unstable,
                ModelIdentity::SameId,
            )
            .verdict,
            conclude(
                "q",
                SamplingPlan::default(),
                once.clone(),
                stable,
                ModelIdentity::SameId,
            )
            .verdict,
        ];
        for verdict in produced {
            let Verdict::NotAttributable { reason } = &verdict else {
                panic!("expected no attribution, got {verdict:?}");
            };
            for banned in [
                "correct",
                "incorrect",
                "wrong",
                "better",
                "worse",
                "accurate",
                "broken",
            ] {
                assert!(!reason.contains(banned), "{reason}");
            }
        }
    }
}
