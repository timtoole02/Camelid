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

fn send(
    spec: &NodeSpec,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<http::HttpResponse, String> {
    // Only ever reached for a Camelid node today; gated anyway, so that
    // stays true if a caller ever routes another engine through here.
    let bearer = spec.engine.fabric_bearer(bearer);
    http::request_with_transport(
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
    .map_err(|error| error.to_string())
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
    let response = send(spec, method, path, body, bearer, timeout, transport)?;
    if response.status != 200 {
        return Err(format!("{path} answered HTTP {}", response.status));
    }
    Ok(response.body)
}

/// Why a Camelid node gave no answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AskError {
    /// It refused a request bound to a weights digest because its loaded GGUF
    /// is other bytes. That answers the comparison's identity question.
    OtherWeights(String),
    Failed(String),
}

/// The typed code the engine answers when `camelid_expected_gguf_sha256`
/// names bytes other than the ones it has loaded.
const ARTIFACT_MISMATCH_CODE: &str = "model_artifact_mismatch";

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    #[serde(default)]
    error: Option<ErrorBody>,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

/// Ask this node one prompt. `temperature`, `seed` and `max_tokens` are all
/// fields of our own `ChatCompletionRequest`, so a comparison against a Camelid
/// node is fully controlled.
///
/// `expected_gguf_sha256` binds the request to those exact GGUF bytes, and the
/// engine enforces it against what it has loaded; see [`AskError::OtherWeights`].
pub(crate) fn complete(
    spec: &NodeSpec,
    ask: &super::divergence::Ask<'_>,
    expected_gguf_sha256: Option<&str>,
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<Answer, AskError> {
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
    if let Some(expected) = expected_gguf_sha256 {
        body["camelid_expected_gguf_sha256"] = serde_json::json!(expected);
    }
    let encoded = serde_json::to_vec(&body).map_err(|error| AskError::Failed(error.to_string()))?;
    const PATH: &str = "/v1/chat/completions";
    let response = send(
        spec,
        "POST",
        PATH,
        Some(&encoded),
        bearer,
        timeout,
        transport,
    )
    .map_err(AskError::Failed)?;
    if response.status != 200 {
        return Err(refusal(response.status, &response.body));
    }
    let raw = response.body;
    let payload: ChatPayload = serde_json::from_slice(&raw)
        .map_err(|error| AskError::Failed(format!("{PATH} was not readable: {error}")))?;
    let text = payload
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message)
        .map(|message| message.content)
        .ok_or_else(|| AskError::Failed(format!("{PATH} answered without a choice")))?;
    Ok(Answer {
        text,
        model: payload.model.filter(|model| !model.is_empty()),
        runtime: None,
    })
}

/// Read a refusal: the one typed code that means "other weights" is kept
/// apart from every other failure, which stays a failure.
fn refusal(status: u16, body: &[u8]) -> AskError {
    let error = serde_json::from_slice::<ErrorEnvelope>(body)
        .ok()
        .and_then(|envelope| envelope.error);
    let code = error.as_ref().and_then(|error| error.code.as_deref());
    let message = error
        .as_ref()
        .and_then(|error| error.message.clone())
        .unwrap_or_default();
    if status == 409 && code == Some(ARTIFACT_MISMATCH_CODE) {
        return AskError::OtherWeights(format!("{ARTIFACT_MISMATCH_CODE}: {message}"));
    }
    AskError::Failed(format!(
        "/v1/chat/completions answered HTTP {status}{}",
        code.map(|code| format!(" ({code}: {message})"))
            .unwrap_or_default()
    ))
}

#[derive(Debug, Deserialize)]
struct ModelListing {
    #[serde(default)]
    data: Vec<ListedModel>,
}

#[derive(Debug, Deserialize)]
struct ListedModel {
    #[serde(default)]
    id: String,
    #[serde(default)]
    gguf_sha256: Option<String>,
}

/// The SHA-256 of the GGUF this node serves as `model`, from `GET /v1/models`,
/// where every loaded model carries the digest of its exact file.
pub(crate) fn weights_digest(
    spec: &NodeSpec,
    model: &str,
    bearer: Option<&str>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<String, String> {
    let raw = request(spec, "GET", "/v1/models", None, bearer, timeout, transport)?;
    digest_in_listing(&raw, model)
}

fn digest_in_listing(raw: &[u8], model: &str) -> Result<String, String> {
    let listing: ModelListing = serde_json::from_slice(raw)
        .map_err(|error| format!("/v1/models was not readable: {error}"))?;
    let entry = listing
        .data
        .into_iter()
        .find(|entry| entry.id == model)
        .ok_or_else(|| format!("/v1/models does not list {model}"))?;
    entry
        .gguf_sha256
        .as_deref()
        .and_then(super::divergence::normalized_sha256)
        .ok_or_else(|| format!("/v1/models lists {model} without a usable gguf_sha256"))
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

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3";

    /// The shape `GET /v1/models` answers with: every loaded model, each with
    /// the SHA-256 of its exact GGUF.
    #[test]
    fn the_weights_digest_is_the_listed_gguf_sha256_of_the_model_asked_for() {
        let listing = format!(
            r#"{{"object":"list","data":[
                {{"id":"other","gguf_sha256":"{}","filename":"o.gguf"}},
                {{"id":"m","gguf_sha256":"{}","filename":"m.gguf","meta":null}}
            ]}}"#,
            "b".repeat(64),
            DIGEST.to_ascii_uppercase()
        );
        assert_eq!(
            digest_in_listing(listing.as_bytes(), "m").as_deref(),
            Ok(DIGEST)
        );
    }

    #[test]
    fn a_listing_without_a_usable_digest_for_the_model_publishes_none() {
        for listing in [
            r#"{"data":[]}"#.to_string(),
            r#"{"data":[{"id":"m"}]}"#.to_string(),
            r#"{"data":[{"id":"m","gguf_sha256":""}]}"#.to_string(),
            r#"{"data":[{"id":"m","gguf_sha256":"not-a-digest"}]}"#.to_string(),
            format!(r#"{{"data":[{{"id":"other","gguf_sha256":"{DIGEST}"}}]}}"#),
            "not json".to_string(),
        ] {
            assert!(
                digest_in_listing(listing.as_bytes(), "m").is_err(),
                "{listing}"
            );
        }
    }

    /// Only the engine's typed mismatch means other weights. Every other
    /// refusal, including one about the binding itself, stays a failure.
    #[test]
    fn only_the_typed_artifact_mismatch_reads_as_other_weights() {
        let mismatch = br#"{"error":{"message":"the selected model id now refers to different GGUF bytes","type":"invalid_request","code":"model_artifact_mismatch","param":"camelid_expected_gguf_sha256"}}"#;
        assert!(matches!(
            refusal(409, mismatch),
            AskError::OtherWeights(detail) if detail.contains("different GGUF bytes")
        ));
        let unavailable = br#"{"error":{"message":"no usable identity","code":"model_artifact_identity_unavailable"}}"#;
        assert!(matches!(refusal(409, unavailable), AskError::Failed(_)));
        assert!(matches!(refusal(500, mismatch), AskError::Failed(_)));
        assert!(matches!(refusal(409, b"<html>"), AskError::Failed(_)));
    }
}
