//! Reading an Ollama server as a fabric node.
//!
//! Ollama exposes no single health route, so readiness is assembled from three
//! documented endpoints:
//!
//! | endpoint | question it answers |
//! |---|---|
//! | `GET /api/version` | is this an Ollama server, and which one |
//! | `GET /api/tags` | which models are installed and could be served |
//! | `GET /api/ps` | which models are resident right now |
//!
//! What it cannot answer is as important as what it can. There is no queue
//! depth and no capacity anywhere in that API, so a node read here reports
//! **no load at all** rather than a zero. `/api/ps` is the closest thing to
//! Camelid's single `active_model_id`, and it is only reported as one when
//! exactly one model is resident — Ollama can hold several, and picking one of
//! them would be inventing a fact.

use std::time::Duration;

use serde::Deserialize;

use super::cancel::Cancel;
use super::divergence::Answer;
use super::engine::NodeEngine;
use super::http;
use super::node::{NodeReady, NodeSpec, NodeStatus};
use super::transport::NodeTransport;

/// Refuse a listing larger than this. A tag list is a few KiB per model; this
/// is generous for a very full library and still bounds a hostile answer.
const MAX_LISTING_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct VersionPayload {
    #[serde(default)]
    version: String,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    /// Ollama sends both `name` and `model`; they agree in practice, and `name`
    /// is the one its own documentation shows operators using.
    #[serde(default)]
    name: String,
}

#[derive(Debug, Default, Deserialize)]
struct ModelListing {
    #[serde(default)]
    models: Vec<ModelEntry>,
}

impl ModelListing {
    fn names(self) -> Vec<String> {
        let mut names: Vec<String> = self
            .models
            .into_iter()
            .map(|entry| entry.name)
            .filter(|name| !name.is_empty())
            .collect();
        names.sort();
        names.dedup();
        names
    }
}

fn get_json<T: for<'de> Deserialize<'de>>(
    spec: &NodeSpec,
    path: &str,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<T, String> {
    request_json(spec, "GET", path, None, timeout, transport)
}

fn request_json<T: for<'de> Deserialize<'de>>(
    spec: &NodeSpec,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<T, String> {
    // No bearer: an Ollama server has no place for a Camelid API key, and
    // presenting one would hand this fabric's credential to a foreign process.
    let response = http::request_with_transport(
        &spec.host,
        spec.port,
        method,
        path,
        body,
        None,
        timeout,
        MAX_LISTING_BYTES,
        &Cancel::never(),
        transport,
    )
    .map_err(|error| error.to_string())?;
    if response.status != 200 {
        return Err(format!("{path} answered HTTP {}", response.status));
    }
    serde_json::from_slice::<T>(&response.body)
        .map_err(|error| format!("{path} was not readable: {error}"))
}

/// Read one Ollama server.
///
/// `Unreachable` is reserved for "we could not ask": if `/api/version` answers,
/// the node exists and anything missing after that is a `NotReady` reason the
/// operator can act on.
pub(crate) fn probe(spec: &NodeSpec, timeout: Duration, transport: &NodeTransport) -> NodeStatus {
    let version = match get_json::<VersionPayload>(spec, "/api/version", timeout, transport) {
        Ok(payload) => payload.version,
        Err(reason) => return NodeStatus::Unreachable { reason },
    };

    let installed = match get_json::<ModelListing>(spec, "/api/tags", timeout, transport) {
        Ok(listing) => listing.names(),
        Err(reason) => return NodeStatus::NotReady { reason },
    };
    if installed.is_empty() {
        return NodeStatus::NotReady {
            reason: "no models installed; run `ollama pull <model>` on that machine".to_string(),
        };
    }

    // A failure here is not fatal: the node can still serve, we simply do not
    // learn which model is warm. Reporting none is the honest degradation.
    let mut resident = get_json::<ModelListing>(spec, "/api/ps", timeout, transport)
        .map(ModelListing::names)
        .unwrap_or_default();
    resident.retain(|name| installed.contains(name));

    NodeStatus::Ready(NodeReady {
        engine: NodeEngine::Ollama,
        // Only when there is exactly one; see the module note.
        active_model_id: match resident.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        },
        models: installed,
        // Ollama names no execution lane, and publishes no queue depth.
        backend: None,
        version: (!version.is_empty()).then_some(version),
        load: None,
    })
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatPayload {
    /// The model Ollama says produced the answer.
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    message: Option<ChatMessage>,
}

/// Ask this server one prompt.
///
/// `POST /api/chat` with `stream: false`, which its documentation specifies
/// returns a single object whose `message.content` is the whole answer.
/// Sampling goes in `options`, where Ollama documents `temperature`, `seed` and
/// `num_predict` — so a comparison against Ollama really is seeded and greedy,
/// unlike one against a backend with no seed parameter.
pub(crate) fn complete(
    spec: &NodeSpec,
    ask: &super::divergence::Ask<'_>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<Answer, String> {
    let mut options = serde_json::json!({
        "temperature": ask.temperature,
        "num_predict": ask.max_tokens,
    });
    if let Some(seed) = ask.seed {
        options["seed"] = serde_json::json!(seed);
    }
    let body = serde_json::json!({
        "model": ask.model,
        "messages": ask.messages(),
        "stream": false,
        "options": options,
    });
    let encoded = serde_json::to_vec(&body).map_err(|error| error.to_string())?;
    let payload: ChatPayload = request_json(
        spec,
        "POST",
        "/api/chat",
        Some(&encoded),
        timeout,
        transport,
    )?;
    let text = payload
        .message
        .map(|message| message.content)
        .ok_or_else(|| "/api/chat answered without a message".to_string())?;
    Ok(Answer {
        text,
        model: payload.model.filter(|model| !model.is_empty()),
        runtime: None,
    })
}

#[derive(Debug, Deserialize)]
struct ShowPayload {
    #[serde(default)]
    template: String,
    #[serde(default)]
    modelfile: String,
}

fn show(
    spec: &NodeSpec,
    model: &str,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<ShowPayload, String> {
    let body = serde_json::json!({ "model": model });
    let encoded = serde_json::to_vec(&body).map_err(|error| error.to_string())?;
    request_json(
        spec,
        "POST",
        "/api/show",
        Some(&encoded),
        timeout,
        transport,
    )
}

/// The SHA-256 of the weights this server serves for `model`.
///
/// Ollama stores every layer content-addressed, so the `FROM` line of the
/// modelfile `POST /api/show` returns names the model blob by its own digest:
/// the digest of the GGUF bytes this server stores and serves. Those are not
/// always the bytes it was given: measured on 0.33.2, `ollama create` from a
/// local GGUF re-serialized it (same size, metadata reordered), so the blob's
/// digest differed from the source file's. `/api/tags` also carries a `digest`, but that
/// one is of the *manifest*, which changes with a template or a parameter while
/// the weights stay put, and is never read as a weights digest.
pub(crate) fn weights_digest(
    spec: &NodeSpec,
    model: &str,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<String, String> {
    let shown = show(spec, model, timeout, transport)?;
    weights_blob_digest(&shown.modelfile)
}

/// The digest of the one content-addressed blob a modelfile's `FROM` names.
///
/// Anything short of exactly one is not published: none means the weights
/// are named some other way, and several means which of them holds the
/// weights is not something this build can tell.
fn weights_blob_digest(modelfile: &str) -> Result<String, String> {
    let mut blobs = Vec::new();
    for line in modelfile.lines() {
        let line = line.trim();
        let Some(path) = line
            .get(..5)
            .filter(|keyword| keyword.eq_ignore_ascii_case("FROM "))
            .map(|_| line[5..].trim())
        else {
            continue;
        };
        let mut components = path.rsplit(['/', '\\']);
        let (Some(name), Some(parent)) = (components.next(), components.next()) else {
            continue;
        };
        if parent != "blobs" {
            continue;
        }
        if let Some(digest) = name
            .strip_prefix("sha256-")
            .or_else(|| name.strip_prefix("sha256:"))
            .and_then(super::divergence::normalized_sha256)
        {
            blobs.push(digest);
        }
    }
    match blobs.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(
            "the modelfile from POST /api/show names no content-addressed model blob".to_string(),
        ),
        _ => Err(format!(
            "the modelfile from POST /api/show names {} blobs, so which holds the weights is not known",
            blobs.len()
        )),
    }
}

/// The prompt template this server would apply for a model.
///
/// `POST /api/show` documents a `template` field, which is the thing that
/// actually explains a divergence — so for Ollama the explanation is
/// obtainable rather than inferred.
pub(crate) fn template(
    spec: &NodeSpec,
    model: &str,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<String, String> {
    let payload = show(spec, model, timeout, transport)?;
    if payload.template.is_empty() {
        return Err("/api/show returned no template for this model".to_string());
    }
    Ok(payload.template)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listing(json: &str) -> Vec<String> {
        serde_json::from_str::<ModelListing>(json)
            .expect("listing parses")
            .names()
    }

    /// Shapes taken from Ollama's own API documentation.
    #[test]
    fn a_documented_tag_listing_reads_as_sorted_unique_names() {
        let names = listing(
            r#"{"models":[
                {"name":"llama3.2:latest","model":"llama3.2:latest","size":2019393189},
                {"name":"deepseek-r1:latest","model":"deepseek-r1:latest","size":4683075271}
            ]}"#,
        );
        assert_eq!(names, vec!["deepseek-r1:latest", "llama3.2:latest"]);
    }

    #[test]
    fn an_empty_or_absent_listing_is_empty_rather_than_an_error() {
        assert!(listing(r#"{"models":[]}"#).is_empty());
        assert!(listing("{}").is_empty());
    }

    #[test]
    fn an_entry_without_a_name_is_dropped_rather_than_listed_blank() {
        assert_eq!(
            listing(r#"{"models":[{"size":1},{"name":"a"}]}"#),
            vec!["a"]
        );
    }

    /// The shape `ollama show --modelfile` prints, from a model created from a
    /// local GGUF: comment lines naming the model, then `FROM` the blob.
    const MODELFILE: &str = "# Modelfile generated by \"ollama show\"\n\
        # To build a new Modelfile based on this, replace FROM with:\n\
        # FROM llama32-1b-q8-r1:latest\n\
        \n\
        FROM /srv/ollama/models/blobs/sha256-432F310A77F4650A88D0FD59ECDD7CEBED8D684BAFEA53CBFF0473542964F0C3\n\
        TEMPLATE \"\"\"{{ .Prompt }}\"\"\"\n\
        PARAMETER stop <|eot_id|>\n";

    #[test]
    fn the_weights_digest_is_the_model_blob_the_modelfile_names() {
        assert_eq!(
            weights_blob_digest(MODELFILE).as_deref(),
            Ok("432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3"),
            "the commented FROM names a tag, not bytes, and is skipped"
        );
        assert_eq!(
            weights_blob_digest(
                "from D:\\ollama\\models\\blobs\\sha256-36330585f362ee081a91cb45550c961dcdbff478f2d02e84e2ae5dc07505967d"
            )
            .as_deref(),
            Ok("36330585f362ee081a91cb45550c961dcdbff478f2d02e84e2ae5dc07505967d")
        );
    }

    #[test]
    fn a_modelfile_that_names_no_single_blob_publishes_no_digest() {
        for modelfile in [
            "",
            "# FROM llama32:latest\n",
            "FROM llama32:latest\n",
            "FROM /tmp/model.gguf\n",
            // A digest-shaped name outside the blob store is a user's file.
            "FROM /tmp/sha256-432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3\n",
            "FROM /x/blobs/sha256-tooshort\n",
        ] {
            assert!(weights_blob_digest(modelfile).is_err(), "{modelfile:?}");
        }
        let two = format!(
            "FROM /x/blobs/sha256-{}\nFROM /x/blobs/sha256-{}\n",
            "a".repeat(64),
            "b".repeat(64)
        );
        let reason = weights_blob_digest(&two).expect_err("ambiguous");
        assert!(reason.contains("2 blobs"), "{reason}");
    }

    #[test]
    fn unknown_fields_do_not_break_a_newer_ollama() {
        assert_eq!(
            listing(r#"{"models":[{"name":"a","brand_new_field":true}],"also_new":1}"#),
            vec!["a"]
        );
    }
}
