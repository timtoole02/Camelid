//! Asking our own engine a question, for measurement rather than placement.
//!
//! Placement goes through [`super::forward`], which carries a request the fabric
//! was given. This is the other direction: the fabric composes a request of its
//! own to find out what a node would say. Keeping it separate means the
//! divergence view can never accidentally acquire the retry, affinity and
//! failover behaviour that belongs to serving real traffic — a comparison that
//! silently failed over to a second node would be measuring the wrong machine.

use std::time::Duration;

use serde::Deserialize;

use super::cancel::Cancel;
use super::divergence::Answer;
use super::http;
use super::node::NodeSpec;
use super::transport::NodeTransport;

const MAX_ANSWER_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    message: Option<ChoiceMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatPayload {
    /// Our own engine names its loaded model here — the same id its health
    /// reports, which is what a comparison asks it for.
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<Choice>,
}

fn request(
    spec: &NodeSpec,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<Vec<u8>, String> {
    // Only ever reached for a Camelid node today; gated anyway, so that
    // stays true if a caller ever routes another engine through here.
    let bearer = spec.engine.fabric_bearer(bearer);
    let response = http::request_with_transport(
        &spec.host,
        spec.port,
        method,
        path,
        body,
        bearer,
        timeout,
        MAX_ANSWER_BYTES,
        &Cancel::never(),
        transport,
    )
    .map_err(|error| error.to_string())?;
    if response.status != 200 {
        return Err(format!("{path} answered HTTP {}", response.status));
    }
    Ok(response.body)
}

/// Ask this node one prompt. `temperature`, `seed` and `max_tokens` are all
/// fields of our own `ChatCompletionRequest`, so a comparison against a Camelid
/// node is fully controlled.
#[allow(clippy::too_many_arguments)]
pub(crate) fn complete(
    spec: &NodeSpec,
    ask: &super::divergence::Ask<'_>,
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<Answer, String> {
    let mut body = serde_json::json!({
        "model": ask.model,
        "messages": ask.messages(),
        "temperature": ask.temperature,
        "max_tokens": ask.max_tokens,
        "stream": false,
    });
    if let Some(seed) = ask.seed {
        body["seed"] = serde_json::json!(seed);
    }
    let encoded = serde_json::to_vec(&body).map_err(|error| error.to_string())?;
    let raw = request(
        spec,
        "POST",
        "/v1/chat/completions",
        Some(&encoded),
        bearer,
        timeout,
        transport,
    )?;
    let payload: ChatPayload = serde_json::from_slice(&raw)
        .map_err(|error| format!("/v1/chat/completions was not readable: {error}"))?;
    let text = payload
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message)
        .map(|message| message.content)
        .ok_or_else(|| "/v1/chat/completions answered without a choice".to_string())?;
    Ok(Answer {
        text,
        model: payload.model.filter(|model| !model.is_empty()),
        runtime: None,
    })
}

#[derive(Debug, Deserialize)]
struct RenderedPayload {
    #[serde(default)]
    prompt: Option<String>,
}

/// The prompt this node renders for the question, from `POST /apply-template`,
/// which renders without generating.
///
/// Sent exactly the messages [`complete`] sends, because a render of any
/// other conversation says nothing about the comparison. It renders for the
/// node's active model, which for a Camelid node is the only one its health
/// lists, so it is the model the comparison asked for. It renders with
/// thinking off, which is also what a chat request that does not ask for
/// thinking gets.
pub(crate) fn rendered_prompt(
    spec: &NodeSpec,
    ask: &super::divergence::Ask<'_>,
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<String, String> {
    let body = serde_json::json!({ "messages": ask.messages() });
    let encoded = serde_json::to_vec(&body).map_err(|error| error.to_string())?;
    let raw = request(
        spec,
        "POST",
        "/apply-template",
        Some(&encoded),
        bearer,
        timeout,
        transport,
    )?;
    let payload: RenderedPayload = serde_json::from_slice(&raw)
        .map_err(|error| format!("/apply-template was not readable: {error}"))?;
    payload
        .prompt
        .ok_or_else(|| "/apply-template answered without a prompt".to_string())
}

#[derive(Debug, Deserialize)]
struct PropsPayload {
    #[serde(default)]
    chat_template: Option<String>,
}

/// The chat template this node advertises, from `GET /props`.
pub(crate) fn template(
    spec: &NodeSpec,
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<String, String> {
    let raw = request(spec, "GET", "/props", None, bearer, timeout, transport)?;
    let payload: PropsPayload = serde_json::from_slice(&raw)
        .map_err(|error| format!("/props was not readable: {error}"))?;
    match payload.chat_template {
        Some(template) if !template.is_empty() => Ok(template),
        // A loaded model with no template is a real state, not a failed read.
        _ => Err("/props reports no chat template for the loaded model".to_string()),
    }
}
