//! Divergence: the same prompt on two nodes, and an honest account of what came
//! back.
//!
//! This is the part of the fabric that exists because of a measured fact: an
//! identical GGUF, asked *"What is 7 plus 5?"* greedily, answers **12** on
//! Camelid and **7** on llama.cpp / Ollama / LM Studio, a difference traced to
//! the prompts the backends build. Mixed-engine routing makes that our problem;
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

impl Ask<'_> {
    /// The chat messages every adapter sends for this question.
    ///
    /// One definition, because a rendered prompt is only evidence about a
    /// comparison when it was rendered from exactly the messages that
    /// comparison generated from.
    pub(crate) fn messages(&self) -> serde_json::Value {
        serde_json::json!([{ "role": "user", "content": self.prompt }])
    }
}

/// One reply, as the engine itself described it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Answer {
    pub(crate) text: String,
    /// The model the response named. `None` when it named none — never filled
    /// in from the request, because comparing the two is the only evidence
    /// that a node served what it was asked for.
    pub(crate) model: Option<String>,
    /// The inference runtime the response named, for an engine that names one
    /// apart from its own version; see [`Side::runtime`].
    pub(crate) runtime: Option<String>,
}

/// Most runs a side may be asked for. Each is a full generation on a real
/// node, so this bounds how long one comparison can occupy two machines.
pub const MAX_COMPARE_REPETITIONS: usize = 5;

/// Hottest temperature a comparison accepts: the top of the range the
/// OpenAI-shaped completion APIs define. Outside it a node may refuse the
/// request or adjust the value silently, and an adjusted run would sit under a
/// recorded plan that never ran.
pub const MAX_COMPARE_TEMPERATURE: f32 = 2.0;

/// Refuse a temperature a comparison cannot honestly record as applied.
pub fn check_temperature(temperature: f32) -> Result<(), String> {
    if temperature.is_finite() && (0.0..=MAX_COMPARE_TEMPERATURE).contains(&temperature) {
        Ok(())
    } else {
        Err(format!(
            "temperature must be a finite number from 0 to {MAX_COMPARE_TEMPERATURE}; got {temperature}"
        ))
    }
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

impl SamplingPlan {
    /// This plan with its repetitions inside what a comparison will run, so
    /// the plan a receipt records is the plan that ran. Zero reads as one: a
    /// side is always asked at least once, and recording zero beside one
    /// sample would misstate the receipt.
    pub fn bounded(mut self) -> Self {
        self.repetitions = self.repetitions.clamp(1, MAX_COMPARE_REPETITIONS);
        self
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

/// The chat template an engine **advertises** for a model: the text it
/// publishes as that model's template, where it publishes one at all.
///
/// Never evidence of what the engine applied. Measured live: Camelid and
/// Ollama advertised byte-identical templates for one GGUF, and Camelid's own
/// renderer still produced a prompt without the system block that template
/// emits. What an engine applied is only shown by a rendered prompt; see
/// [`RenderedPrompt`].
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

/// The prompt an engine rendered from exactly the messages a comparison sent,
/// without generating from it.
///
/// This, and not [`TemplateEvidence`], is what shows the text an engine built.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RenderedPrompt {
    Captured {
        source: String,
        text: String,
    },
    /// No render was obtained. The reason says whether the engine has no way
    /// to render without generating or whether asking failed.
    Unavailable {
        reason: String,
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
    /// The model id this side was asked for: the id local to this node, after
    /// any alias. What the node says it actually served is `reported_model`.
    pub model: Option<String>,
    /// The model the node's own generation responses named. `None` when no
    /// response named one; never copied from `model`, because a difference
    /// between the two is the only evidence here that a node answered from
    /// weights other than the ones it was asked for.
    pub reported_model: Option<String>,
    pub applied_sampling: AppliedSampling,
    pub samples: Vec<Sample>,
    pub stability: Stability,
    /// Named for what it is on the wire as well. It was `template` beside a
    /// view titled "templates applied", and a reader took the advertised text
    /// for the prompt the engine built.
    pub advertised_template: TemplateEvidence,
    pub rendered_prompt: RenderedPrompt,
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
    /// What this comparison did not control, by name. Empty when nothing is
    /// listed. The same names, in the same order, as `uncontrolled_detail`,
    /// which says why each one is here; kept as bare names so a reader of the
    /// original wire shape still gets them.
    pub uncontrolled: Vec<String>,
    pub uncontrolled_detail: Vec<Uncontrolled>,
}

/// One thing a comparison did not control, and why.
///
/// Each item has its own reason because they are different kinds of gap: a
/// seed is a parameter an engine may lack, while model identity is a check
/// nobody ran. One sentence for all of them printed "model identity — at
/// least one engine has no such parameter", which is false.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Uncontrolled {
    pub name: String,
    pub reason: String,
}

/// Whether two captured texts are the same bytes, when both were captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextMatch {
    Identical,
    Different,
    /// At least one side has no captured text, so nothing can be said.
    NotComparable,
}

impl Comparison {
    /// How the two advertised templates compare.
    pub fn advertised_templates(&self) -> TextMatch {
        match (
            &self.left.advertised_template,
            &self.right.advertised_template,
        ) {
            (
                TemplateEvidence::Captured { template: l, .. },
                TemplateEvidence::Captured { template: r, .. },
            ) => text_match(l, r),
            _ => TextMatch::NotComparable,
        }
    }

    /// How the two rendered prompts compare.
    pub fn rendered_prompts(&self) -> TextMatch {
        match (&self.left.rendered_prompt, &self.right.rendered_prompt) {
            (
                RenderedPrompt::Captured { text: l, .. },
                RenderedPrompt::Captured { text: r, .. },
            ) => text_match(l, r),
            _ => TextMatch::NotComparable,
        }
    }

    /// The sentence a reader needs when the advertised templates cannot be
    /// the explanation for an established difference.
    ///
    /// Without it, two identical templates shown beside a divergence invite
    /// the reading that the templates were checked and are the cause.
    pub fn unexplained_by_advertised_template(&self) -> Option<&'static str> {
        (self.verdict == Verdict::Divergent && self.advertised_templates() == TextMatch::Identical)
            .then_some(
                "both nodes advertise byte-identical chat templates, so the advertised template does not explain this difference",
            )
    }
}

fn text_match(left: &str, right: &str) -> TextMatch {
    if left == right {
        TextMatch::Identical
    } else {
        TextMatch::Different
    }
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

    let mut uncontrolled_detail = Vec::new();
    for (name, missing) in [
        ("seed", missing_parameter(&left, &right, |s| s.seed, "seed")),
        (
            "temperature",
            missing_parameter(&left, &right, |s| s.temperature, "temperature"),
        ),
    ] {
        if let Some(reason) = missing {
            uncontrolled_detail.push(Uncontrolled {
                name: name.to_string(),
                reason,
            });
        }
    }
    // The biggest uncontrolled variable of all when it applies: nobody checked
    // that these are the same weights. An operator's word is one way that
    // happens; an equal id on two engines is the other, because each engine
    // resolves a name by its own rules and an equal name is not equal weights.
    // Not added beside a verdict that already says the models differ.
    let identity_reason = match model_identity {
        ModelIdentity::AssertedByOperator => Some(format!(
            "the operator declared `{}` and `{}` to be the same weights, and nothing here checked it",
            left.model.as_deref().unwrap_or("-"),
            right.model.as_deref().unwrap_or("-"),
        )),
        ModelIdentity::SameId
            if left.engine != right.engine
                && !matches!(verdict, Verdict::DifferentModels { .. }) =>
        {
            Some(format!(
                "both sides were asked for `{}`, but {} and {} each resolve a name by their own rules, so an equal name is not shown to be the same weights",
                left.model.as_deref().unwrap_or("-"),
                left.engine,
                right.engine,
            ))
        }
        ModelIdentity::SameId => None,
    };
    if let Some(reason) = identity_reason {
        uncontrolled_detail.push(Uncontrolled {
            name: "model identity".to_string(),
            reason,
        });
    }
    let uncontrolled = uncontrolled_detail
        .iter()
        .map(|item| item.name.clone())
        .collect();

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
        uncontrolled_detail,
    }
}

/// Why a sampling parameter was not controlled, naming the side or sides
/// whose engine has no such parameter; `None` when both sent it.
fn missing_parameter(
    left: &Side,
    right: &Side,
    pick: fn(&AppliedSampling) -> Honoured,
    parameter: &str,
) -> Option<String> {
    let lacking: Vec<String> = [left, right]
        .into_iter()
        .filter(|side| pick(&side.applied_sampling) == Honoured::Unsupported)
        .map(|side| format!("{} ({})", side.label, side.engine))
        .collect();
    match lacking.as_slice() {
        [] => None,
        [one] => Some(format!(
            "{one} runs an engine whose documented completion API has no {parameter} parameter, so its runs were sent none"
        )),
        _ => Some(format!(
            "{} run engines whose documented completion APIs have no {parameter} parameter, so their runs were sent none",
            lacking.join(" and ")
        )),
    }
}

fn verdict_for(left: &Side, right: &Side, model_identity: ModelIdentity) -> Verdict {
    // What a node's own response says it served outranks every other piece of
    // identity evidence here, an operator's assertion included: that assertion
    // is about the ids asked for, and a node that answered from other weights
    // is not serving the model under comparison at all.
    if served_other_than_asked(left) || served_other_than_asked(right) {
        return Verdict::DifferentModels {
            left: served(left),
            right: served(right),
        };
    }
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

fn served_other_than_asked(side: &Side) -> bool {
    matches!(
        (side.reported_model.as_deref(), side.model.as_deref()),
        (Some(reported), Some(asked)) if reported != asked
    )
}

/// The best name for what a side served: what it said, else what it was asked.
fn served(side: &Side) -> String {
    side.reported_model
        .as_deref()
        .or(side.model.as_deref())
        .unwrap_or("an unnamed model")
        .to_string()
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
            reported_model: None,
            applied_sampling: AppliedSampling::for_engine(engine),
            stability: stability_of(&samples),
            samples,
            advertised_template: TemplateEvidence::NotExposed {
                detail: "test".to_string(),
            },
            rendered_prompt: RenderedPrompt::Unavailable {
                reason: "test".to_string(),
            },
        }
    }

    fn advertising(mut side: Side, template: &str) -> Side {
        side.advertised_template = TemplateEvidence::Captured {
            source: "GET /props".to_string(),
            template: template.to_string(),
        };
        side
    }

    /// C1. The captured template is what the engine publishes, and the wire
    /// has to say so: under the old `template` key, beside a view titled
    /// "templates applied", a reader took it for the prompt the engine built.
    #[test]
    fn a_captured_template_is_serialised_as_advertised_never_as_applied() {
        let side = advertising(
            side("win", NodeEngine::Camelid, "m", &["12", "12"]),
            "{{ x }}",
        );
        let wire = serde_json::to_value(&side).expect("serialises");
        let object = wire.as_object().expect("an object");
        assert_eq!(
            wire["advertised_template"]["kind"], "captured",
            "the capture travels under its own name: {wire}"
        );
        assert!(
            !object.contains_key("template"),
            "an unqualified `template` key reads as the one applied: {wire}"
        );
        for key in object.keys() {
            assert!(
                key == "applied_sampling" || !key.contains("applied"),
                "nothing but a rendered prompt may be called applied: {key}"
            );
        }
        assert_eq!(wire["rendered_prompt"]["kind"], "unavailable");
    }

    /// The live receipt: byte-identical advertised templates, and still a
    /// divergence. The comparison has to say the template is not the cause.
    #[test]
    fn identical_advertised_templates_beside_a_divergence_are_said_not_to_explain_it() {
        let comparison = conclude(
            "Say hi.",
            SamplingPlan::default(),
            advertising(
                side("win", NodeEngine::Camelid, "m", &["Hi!", "Hi!"]),
                "{{ same }}",
            ),
            advertising(
                side(
                    "studio",
                    NodeEngine::Ollama,
                    "m",
                    &["Hi. How can I assist you today?"; 2],
                ),
                "{{ same }}",
            ),
            ModelIdentity::SameId,
        );
        assert_eq!(comparison.verdict, Verdict::Divergent);
        assert_eq!(comparison.advertised_templates(), TextMatch::Identical);
        let note = comparison
            .unexplained_by_advertised_template()
            .expect("a divergence beside identical templates is called out");
        assert!(note.contains("does not explain"), "{note}");
    }

    /// Paired: templates that differ, or a verdict that is not a divergence,
    /// carry no such sentence.
    #[test]
    fn the_unexplained_note_appears_only_for_identical_templates_and_a_divergence() {
        let differing = conclude(
            "q",
            SamplingPlan::default(),
            advertising(
                side("win", NodeEngine::Camelid, "m", &["a", "a"]),
                "{{ l }}",
            ),
            advertising(
                side("mac", NodeEngine::Camelid, "m", &["b", "b"]),
                "{{ r }}",
            ),
            ModelIdentity::SameId,
        );
        assert_eq!(differing.advertised_templates(), TextMatch::Different);
        assert_eq!(differing.unexplained_by_advertised_template(), None);

        let identical = conclude(
            "q",
            SamplingPlan::default(),
            advertising(
                side("win", NodeEngine::Camelid, "m", &["a", "a"]),
                "{{ t }}",
            ),
            advertising(
                side("mac", NodeEngine::Camelid, "m", &["a", "a"]),
                "{{ t }}",
            ),
            ModelIdentity::SameId,
        );
        assert_eq!(identical.verdict, Verdict::Identical);
        assert_eq!(identical.unexplained_by_advertised_template(), None);

        let uncaptured = conclude(
            "q",
            SamplingPlan::default(),
            side("win", NodeEngine::Camelid, "m", &["a", "a"]),
            side("mac", NodeEngine::Camelid, "m", &["b", "b"]),
            ModelIdentity::SameId,
        );
        assert_eq!(uncaptured.advertised_templates(), TextMatch::NotComparable);
        assert_eq!(uncaptured.unexplained_by_advertised_template(), None);
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
        // Seed, because LM Studio takes none. Model identity, because an
        // equal id on two engines is two names agreeing rather than one set of
        // weights (pinned on its own below). Nothing else is invented.
        assert_eq!(comparison.uncontrolled, ["seed", "model identity"]);
        assert_eq!(
            comparison.verdict,
            Verdict::Divergent,
            "an uncontrolled seed is disclosed, not a reason to withhold a verdict both sides earned"
        );
    }

    fn reporting(mut side: Side, named: &str) -> Side {
        side.reported_model = Some(named.to_string());
        side
    }

    /// D3 could only fire from a hand-built side while `model` held the id
    /// that was asked for. The id a node's response names is the evidence
    /// that actually reaches this function in production.
    #[test]
    fn a_node_naming_other_weights_than_it_was_asked_for_is_a_different_model_never_a_divergence() {
        let left = reporting(side("win", NodeEngine::Camelid, "m", &["12", "12"]), "m");
        let right = reporting(
            side("studio", NodeEngine::Ollama, "m", &["7", "7"]),
            "other:latest",
        );
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
                left: "m".to_string(),
                right: "other:latest".to_string()
            }
        );
        assert!(matches!(comparison.diff, Diff::Declined { .. }));
    }

    #[test]
    fn an_operator_assertion_does_not_cover_a_node_that_served_something_else() {
        let left = reporting(
            side("studio", NodeEngine::Ollama, "m:latest", &["12", "12"]),
            "m:latest",
        );
        let right = reporting(side("desk", NodeEngine::LmStudio, "m", &["7", "7"]), "b");
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            left,
            right,
            ModelIdentity::AssertedByOperator,
        );
        assert_eq!(
            comparison.verdict,
            Verdict::DifferentModels {
                left: "m:latest".to_string(),
                right: "b".to_string()
            }
        );
    }

    /// The paired "unaffected" case: a response naming exactly what was asked
    /// for, or naming nothing at all, leaves the verdict to the answers.
    #[test]
    fn a_response_naming_the_model_asked_for_or_none_at_all_changes_nothing() {
        let named = conclude(
            "q",
            SamplingPlan::default(),
            reporting(side("win", NodeEngine::Camelid, "m", &["12", "12"]), "m"),
            reporting(side("mac", NodeEngine::Camelid, "m", &["7", "7"]), "m"),
            ModelIdentity::SameId,
        );
        assert_eq!(named.verdict, Verdict::Divergent);

        let silent = conclude(
            "q",
            SamplingPlan::default(),
            side("win", NodeEngine::Camelid, "m", &["12", "12"]),
            side("mac", NodeEngine::Camelid, "m", &["7", "7"]),
            ModelIdentity::SameId,
        );
        assert_eq!(silent.verdict, Verdict::Divergent);
    }

    /// I14: an equal name is not equal weights. Across two engines the name is
    /// resolved twice, by two sets of rules, so the comparison rests on it.
    #[test]
    fn an_equal_id_on_two_engines_is_disclosed_as_an_uncontrolled_identity() {
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            side("win", NodeEngine::Camelid, "m", &["12", "12"]),
            side("studio", NodeEngine::Ollama, "m", &["7", "7"]),
            ModelIdentity::SameId,
        );
        assert_eq!(comparison.uncontrolled, ["model identity"]);
        assert_eq!(
            comparison.verdict,
            Verdict::Divergent,
            "disclosed, not a reason to withhold the verdict"
        );
    }

    /// C3. Every uncontrolled item says why it is there, in its own words. A
    /// shared sentence called model identity a missing engine parameter.
    #[test]
    fn every_uncontrolled_item_carries_its_own_reason() {
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

        assert_eq!(comparison.uncontrolled, ["seed", "model identity"]);
        let names: Vec<&str> = comparison
            .uncontrolled_detail
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(
            names, comparison.uncontrolled,
            "the bare names and the detail stay parallel"
        );

        let reason = |name: &str| {
            comparison
                .uncontrolled_detail
                .iter()
                .find(|item| item.name == name)
                .map(|item| item.reason.as_str())
                .expect("listed")
        };
        let seed = reason("seed");
        assert!(
            seed.contains("desk (lmstudio)") && seed.contains("no seed parameter"),
            "{seed}"
        );
        assert!(
            !seed.contains("studio (ollama)"),
            "ollama sends a seed: {seed}"
        );
        let identity = reason("model identity");
        assert!(
            identity.contains("declared")
                && identity.contains("`llama-3.2-1b-instruct:latest`")
                && identity.contains("`llama-3.2-1b-instruct`"),
            "{identity}"
        );
        assert!(
            !identity.contains("parameter"),
            "model identity is not a parameter any engine could lack: {identity}"
        );
        assert_ne!(seed, identity);
    }

    #[test]
    fn an_equal_id_on_two_engines_says_the_name_is_all_that_matched() {
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            side("win", NodeEngine::Camelid, "m", &["12", "12"]),
            side("studio", NodeEngine::Ollama, "m", &["7", "7"]),
            ModelIdentity::SameId,
        );
        let [only] = comparison.uncontrolled_detail.as_slice() else {
            panic!("{:?}", comparison.uncontrolled_detail);
        };
        assert_eq!(only.name, "model identity");
        assert!(
            only.reason.contains("both sides were asked for `m`")
                && only
                    .reason
                    .contains("camelid and ollama each resolve a name"),
            "{}",
            only.reason
        );
        assert!(
            !only.reason.contains("declared"),
            "nobody declared anything here"
        );
    }

    #[test]
    fn an_equal_id_on_one_engine_carries_no_identity_caveat() {
        let comparison = conclude(
            "q",
            SamplingPlan::default(),
            side("win", NodeEngine::Camelid, "m", &["12", "12"]),
            side("mac", NodeEngine::Camelid, "m", &["7", "7"]),
            ModelIdentity::SameId,
        );
        assert!(
            comparison.uncontrolled.is_empty(),
            "{:?}",
            comparison.uncontrolled
        );
    }

    #[test]
    fn a_plan_records_the_repetitions_that_actually_run() {
        let asked = |repetitions| SamplingPlan {
            repetitions,
            ..SamplingPlan::default()
        };
        assert_eq!(asked(0).bounded().repetitions, 1);
        assert_eq!(
            asked(1_000_000).bounded().repetitions,
            MAX_COMPARE_REPETITIONS
        );
        assert_eq!(SamplingPlan::default().bounded(), SamplingPlan::default());
    }

    #[test]
    fn a_temperature_outside_the_documented_range_is_refused() {
        for accepted in [0.0, 0.7, MAX_COMPARE_TEMPERATURE] {
            assert!(check_temperature(accepted).is_ok(), "{accepted}");
        }
        for refused in [-0.1, 2.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let reason = check_temperature(refused).expect_err("out of range");
            assert!(reason.contains("from 0 to 2"), "{reason}");
        }
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
