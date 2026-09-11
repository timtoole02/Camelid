//! Running one prompt on one node, whatever engine it runs.
//!
//! This is the divergence view's counterpart to [`super::probe`]: the only
//! place that knows which engine speaks which wire format when we are asking a
//! question rather than reading a status. Placement stays entirely unaware of
//! it, which is what lets a foreign engine be *measured* without being *routed
//! to* — and measuring is exactly how a foreign engine could ever earn a
//! `measured` capability provenance.

use std::time::{Duration, Instant};

use super::camelid;
use super::divergence::{
    stability_of, AppliedSampling, Ask, RenderedPrompt, Sample, SamplingPlan, Side,
    TemplateEvidence, MAX_COMPARE_REPETITIONS,
};
use super::engine::NodeEngine;
use super::lmstudio;
use super::node::{NodeSpec, NodeStatus};
use super::ollama;
use super::probe::probe_node_with_transport;
use super::transport::NodeTransport;

/// Why one side of a comparison could not be measured at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleError {
    /// The node did not answer a status probe, so there is nothing to compare.
    NotServing { label: String, reason: String },
    /// The node is serving, but not the model the comparison is about.
    ModelAbsent {
        label: String,
        model: String,
        available: Vec<String>,
    },
    /// A generation request failed.
    Failed { label: String, detail: String },
}

impl std::fmt::Display for SampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotServing { label, reason } => {
                write!(f, "{label} is not serving: {reason}")
            }
            Self::ModelAbsent {
                label,
                model,
                available,
            } => write!(
                f,
                "{label} does not hold {model}; it holds {}",
                if available.is_empty() {
                    "nothing".to_string()
                } else {
                    available.join(", ")
                }
            ),
            Self::Failed { label, detail } => write!(f, "{label} failed to answer: {detail}"),
        }
    }
}

impl std::error::Error for SampleError {}

/// Everything one side needs, gathered so the argument list stays honest about
/// what a measurement depends on.
pub(crate) struct SideRequest<'a> {
    pub(crate) spec: &'a NodeSpec,
    pub(crate) model: &'a str,
    pub(crate) prompt: &'a str,
    pub(crate) plan: &'a SamplingPlan,
    pub(crate) bearer: Option<&'a str>,
    /// For what a node answers without generating: its status and its
    /// template.
    pub(crate) probe_timeout: Duration,
    /// For each generation, which can legitimately take minutes. Held apart
    /// from the probe budget because a resident proxy's is two seconds, and a
    /// node that needed longer than a health read to generate was reported as
    /// having failed.
    pub(crate) generation_timeout: Duration,
}

/// Probe the node, check it holds the model, run the prompt `plan.repetitions`
/// times, and capture the template the engine advertises and the prompt it
/// renders, where the engine exposes either.
pub(crate) fn measure(
    request: &SideRequest<'_>,
    transport: &NodeTransport,
) -> Result<Side, SampleError> {
    let label = request.spec.label.clone();
    // Through the configured transport, like every other read of a node. The
    // default one allows cleartext to loopback only, so a node `fabric status`
    // reached under the operator's flags was refused here under the same ones.
    let snapshot = probe_node_with_transport(
        request.spec,
        request.bearer,
        request.probe_timeout,
        transport,
    );
    let ready = match snapshot.status {
        NodeStatus::Ready(ready) => ready,
        NodeStatus::NotReady { reason } | NodeStatus::Unreachable { reason } => {
            return Err(SampleError::NotServing { label, reason })
        }
    };

    // Refusing here is the difference between measuring a divergence and
    // measuring a typo: a node that does not hold the model would otherwise
    // answer with whatever it does hold, or with an error we would then diff.
    if !ready.models.iter().any(|held| held == request.model) {
        return Err(SampleError::ModelAbsent {
            label,
            model: request.model.to_string(),
            available: ready.models,
        });
    }

    let engine = ready.engine;
    let runs = request.plan.repetitions.clamp(1, MAX_COMPARE_REPETITIONS);
    let mut samples = Vec::with_capacity(runs);
    let mut named = Vec::with_capacity(runs);
    let mut runtime_note = None;
    let ask = Ask {
        model: request.model,
        prompt: request.prompt,
        temperature: request.plan.temperature,
        seed: request.plan.seed,
        max_tokens: request.plan.max_tokens,
    };

    for _ in 0..runs {
        let started = Instant::now();
        let answer = match engine {
            NodeEngine::Camelid => camelid::complete(
                request.spec,
                &ask,
                request.bearer,
                request.generation_timeout,
                transport,
            ),
            NodeEngine::Ollama => {
                ollama::complete(request.spec, &ask, request.generation_timeout, transport)
            }
            NodeEngine::LmStudio => {
                lmstudio::complete(request.spec, &ask, request.generation_timeout, transport)
            }
        }
        .map_err(|detail| SampleError::Failed {
            label: label.clone(),
            detail,
        })?;
        let elapsed = started.elapsed();
        runtime_note = answer.runtime;
        named.push(answer.model);
        samples.push(Sample::new(answer.text, elapsed));
    }

    Ok(Side {
        label,
        engine,
        engine_version: ready.version,
        // Named separately from the version: LM Studio publishes no application
        // version, and this is a llama.cpp build, not that.
        runtime: runtime_note,
        model: Some(request.model.to_string()),
        reported_model: reported_model(request.model, named),
        applied_sampling: AppliedSampling::for_engine(engine),
        stability: stability_of(&samples),
        samples,
        advertised_template: capture_template(request, engine, transport),
        rendered_prompt: capture_rendered_prompt(request, &ask, engine, transport),
    })
}

/// The prompt the engine builds from the messages this comparison sent, where
/// the engine can say so without generating.
fn capture_rendered_prompt(
    request: &SideRequest<'_>,
    ask: &Ask<'_>,
    engine: NodeEngine,
    transport: &NodeTransport,
) -> RenderedPrompt {
    const SOURCE: &str = "POST /apply-template";
    match engine {
        NodeEngine::Camelid => match camelid::rendered_prompt(
            request.spec,
            ask,
            request.bearer,
            request.probe_timeout,
            transport,
        ) {
            Ok(text) => RenderedPrompt::Captured {
                source: SOURCE.to_string(),
                text,
            },
            Err(detail) => RenderedPrompt::Unavailable {
                reason: format!("{SOURCE} could not be read: {detail}"),
            },
        },
        // Neither publishes a way to render a chat prompt without generating,
        // so what either applied cannot be read back; its advertised template
        // is not a substitute.
        NodeEngine::Ollama => RenderedPrompt::Unavailable {
            reason: "Ollama's documented API has no route that renders a chat prompt without generating"
                .to_string(),
        },
        NodeEngine::LmStudio => RenderedPrompt::Unavailable {
            reason: "LM Studio's documented API has no route that renders a chat prompt without generating"
                .to_string(),
        },
    }
}

/// What a node's own answers said it served, as one value.
///
/// A run naming something other than what was asked for outranks runs that
/// matched, so a node that swapped weights on one run of several is still
/// caught. Runs that named nothing leave this unknown rather than assuming the
/// request was honoured.
fn reported_model(asked: &str, named: Vec<Option<String>>) -> Option<String> {
    let named: Vec<String> = named.into_iter().flatten().collect();
    named
        .iter()
        .find(|model| model.as_str() != asked)
        .or_else(|| named.first())
        .cloned()
}

fn capture_template(
    request: &SideRequest<'_>,
    engine: NodeEngine,
    transport: &NodeTransport,
) -> TemplateEvidence {
    let (source, captured) = match engine {
        NodeEngine::Camelid => (
            "GET /props",
            camelid::template(
                request.spec,
                request.bearer,
                request.probe_timeout,
                transport,
            ),
        ),
        NodeEngine::Ollama => (
            "POST /api/show",
            ollama::template(
                request.spec,
                request.model,
                request.probe_timeout,
                transport,
            ),
        ),
        // Its documented API has no endpoint that returns a prompt template.
        // Saying so is the honest answer; an empty string would read as "no
        // template", which is a different and false claim.
        NodeEngine::LmStudio => {
            return TemplateEvidence::NotExposed {
                detail: "LM Studio's documented API exposes no prompt template".to_string(),
            }
        }
    };
    match captured {
        Ok(template) => TemplateEvidence::Captured {
            source: source.to_string(),
            template,
        },
        Err(detail) => TemplateEvidence::Unavailable { detail },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_model_names_what_the_node_does_hold() {
        let error = SampleError::ModelAbsent {
            label: "studio".to_string(),
            model: "llama-3.2-1b".to_string(),
            available: vec!["qwen3:8b".to_string(), "mistral:latest".to_string()],
        };
        let message = error.to_string();
        assert!(message.contains("does not hold llama-3.2-1b"), "{message}");
        assert!(message.contains("qwen3:8b"), "{message}");
    }

    #[test]
    fn a_node_holding_nothing_says_so_rather_than_listing_an_empty_set() {
        let error = SampleError::ModelAbsent {
            label: "studio".to_string(),
            model: "m".to_string(),
            available: Vec::new(),
        };
        assert!(error.to_string().contains("it holds nothing"), "{error}");
    }

    #[test]
    fn a_run_that_named_other_weights_outranks_runs_that_matched() {
        let named = vec![
            Some("m".to_string()),
            Some("other".to_string()),
            Some("m".to_string()),
        ];
        assert_eq!(reported_model("m", named).as_deref(), Some("other"));
    }

    #[test]
    fn runs_that_named_nothing_leave_what_was_served_unknown() {
        assert_eq!(reported_model("m", vec![None, None]), None);
        assert_eq!(
            reported_model("m", vec![None, Some("m".to_string())]).as_deref(),
            Some("m")
        );
    }
}
