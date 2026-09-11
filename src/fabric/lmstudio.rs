//! Reading an LM Studio server as a fabric node.
//!
//! LM Studio has no health route either, but unlike Ollama it answers the whole
//! question in one call: `GET /api/v0/models` lists everything downloaded along
//! with each model's `state`, so what is installed and what is resident arrive
//! together.
//!
//! Two things it cannot tell us, both of which matter:
//!
//! * **No version.** Neither the OpenAI-compatible surface nor the native REST
//!   API documents an endpoint returning the application version. The one place
//!   a version appears is `runtime.version` on a *completion* response, which
//!   names the inference runtime rather than the app — and a probe must not run
//!   a completion to find out. So a node here reports no version, and every
//!   version-keyed capability stays *not probed*.
//! * **No load.** There is no queue-depth or capacity endpoint, so like Ollama
//!   it reports no load at all rather than a zero.

use std::time::Duration;

use serde::Deserialize;

use super::cancel::Cancel;
use super::engine::NodeEngine;
use super::http;
use super::node::{NodeReady, NodeSpec, NodeStatus};
use super::transport::NodeTransport;

/// A model listing is a few hundred bytes per model; this bounds a hostile
/// answer while staying generous for a very full library.
const MAX_LISTING_BYTES: usize = 4 * 1024 * 1024;

/// The value `GET /api/v0/models` uses for a model held in memory.
const STATE_LOADED: &str = "loaded";
/// ...and for one that is merely downloaded.
const STATE_NOT_LOADED: &str = "not-loaded";

#[derive(Debug, Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    state: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelListing {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

/// What one listing says, split into what it holds and what is warm.
struct Inventory {
    installed: Vec<String>,
    /// `None` when the listing used a `state` value this build does not know:
    /// which models are resident is then unknown, rather than none.
    resident: Option<Vec<String>>,
}

fn read_inventory(listing: ModelListing) -> Inventory {
    let mut installed = Vec::new();
    let mut resident = Vec::new();
    let mut understood_every_state = true;

    for entry in listing.data {
        if entry.id.is_empty() {
            continue;
        }
        match entry.state.as_deref() {
            Some(STATE_LOADED) => resident.push(entry.id.clone()),
            Some(STATE_NOT_LOADED) | None => {}
            // A state this build cannot name is not evidence of anything.
            Some(_) => understood_every_state = false,
        }
        installed.push(entry.id);
    }

    installed.sort();
    installed.dedup();
    resident.sort();
    resident.dedup();

    Inventory {
        installed,
        resident: understood_every_state.then_some(resident),
    }
}

/// Read one LM Studio server.
pub(crate) fn probe(spec: &NodeSpec, timeout: Duration, transport: &NodeTransport) -> NodeStatus {
    // No bearer: LM Studio has its own token scheme, and presenting a Camelid
    // credential to a foreign process is never the right move.
    let response = match http::request_with_transport(
        &spec.host,
        spec.port,
        "GET",
        "/api/v0/models",
        None,
        None,
        timeout,
        MAX_LISTING_BYTES,
        &Cancel::never(),
        transport,
    ) {
        Ok(response) => response,
        Err(error) => {
            return NodeStatus::Unreachable {
                reason: error.to_string(),
            }
        }
    };

    if response.status == 401 || response.status == 403 {
        return NodeStatus::NotReady {
            reason: format!(
                "LM Studio refused the listing (HTTP {}); it is configured to require an API token, \
                 which this fabric does not hold for a foreign engine",
                response.status
            ),
        };
    }
    if response.status != 200 {
        return NodeStatus::Unreachable {
            reason: format!("/api/v0/models answered HTTP {}", response.status),
        };
    }

    let listing = match serde_json::from_slice::<ModelListing>(&response.body) {
        Ok(listing) => listing,
        Err(error) => {
            return NodeStatus::Unreachable {
                reason: format!("/api/v0/models was not readable: {error}"),
            }
        }
    };

    let inventory = read_inventory(listing);
    if inventory.installed.is_empty() {
        return NodeStatus::NotReady {
            reason: "no models downloaded; add one in LM Studio on that machine".to_string(),
        };
    }

    NodeStatus::Ready(NodeReady {
        engine: NodeEngine::LmStudio,
        // Only when exactly one is resident, and only when every state in the
        // listing was one this build understands.
        active_model_id: match inventory.resident.as_deref() {
            Some([only]) => Some(only.clone()),
            _ => None,
        },
        models: inventory.installed,
        backend: None,
        // See the module note: there is no endpoint for it.
        version: None,
        load: None,
    })
}

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
struct Runtime {
    #[serde(default)]
    name: String,
    #[serde(default)]
    version: String,
}

#[derive(Debug, Deserialize)]
struct ChatPayload {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    runtime: Option<Runtime>,
}

/// One answer, plus the inference runtime LM Studio says produced it.
pub(crate) struct Completion {
    pub(crate) text: String,
    /// `runtime.name` and `runtime.version` from the response. This names the
    /// **inference runtime** (a llama.cpp build), not the LM Studio
    /// application, and is reported under that name so it is never mistaken
    /// for the engine version LM Studio does not publish.
    pub(crate) runtime: Option<String>,
}

/// Ask this server one prompt.
///
/// `POST /api/v0/chat/completions` documents `temperature`, `max_tokens` and
/// `stream`. It documents **no seed parameter**, so a comparison against LM
/// Studio is not a seeded one; the caller records that rather than pretending
/// otherwise.
pub(crate) fn complete(
    spec: &NodeSpec,
    ask: &super::divergence::Ask<'_>,
    timeout: Duration,
    transport: &NodeTransport,
) -> Result<Completion, String> {
    let body = serde_json::json!({
        "model": ask.model,
        "messages": [{ "role": "user", "content": ask.prompt }],
        "temperature": ask.temperature,
        "max_tokens": ask.max_tokens,
        "stream": false,
    });
    let encoded = serde_json::to_vec(&body).map_err(|error| error.to_string())?;

    let response = http::request_with_transport(
        &spec.host,
        spec.port,
        "POST",
        "/api/v0/chat/completions",
        Some(&encoded),
        None,
        timeout,
        MAX_LISTING_BYTES,
        &Cancel::never(),
        transport,
    )
    .map_err(|error| error.to_string())?;

    if response.status != 200 {
        return Err(format!(
            "/api/v0/chat/completions answered HTTP {}",
            response.status
        ));
    }
    let payload: ChatPayload = serde_json::from_slice(&response.body)
        .map_err(|error| format!("/api/v0/chat/completions was not readable: {error}"))?;

    let text = payload
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message)
        .map(|message| message.content)
        .ok_or_else(|| "/api/v0/chat/completions answered without a choice".to_string())?;

    let runtime = payload.runtime.and_then(|runtime| {
        match (runtime.name.is_empty(), runtime.version.is_empty()) {
            (true, true) => None,
            (false, false) => Some(format!("{} {}", runtime.name, runtime.version)),
            (false, true) => Some(runtime.name),
            (true, false) => Some(runtime.version),
        }
    });

    Ok(Completion { text, runtime })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inventory(json: &str) -> Inventory {
        read_inventory(serde_json::from_str::<ModelListing>(json).expect("listing parses"))
    }

    /// Shape taken from LM Studio's own REST API v0 documentation.
    #[test]
    fn a_documented_listing_separates_downloaded_from_resident() {
        let read = inventory(
            r#"{"object":"list","data":[
                {"id":"qwen2-vl-7b-instruct","object":"model","type":"vlm","arch":"qwen2_vl",
                 "compatibility_type":"mlx","quantization":"4bit","state":"not-loaded","max_context_length":32768},
                {"id":"meta-llama-3.1-8b-instruct","object":"model","type":"llm","arch":"llama",
                 "compatibility_type":"gguf","quantization":"Q4_K_M","state":"loaded","max_context_length":131072}
            ]}"#,
        );
        assert_eq!(
            read.installed,
            vec![
                "meta-llama-3.1-8b-instruct".to_string(),
                "qwen2-vl-7b-instruct".to_string()
            ],
            "everything downloaded is something it can serve"
        );
        assert_eq!(
            read.resident.as_deref(),
            Some(["meta-llama-3.1-8b-instruct".to_string()].as_slice())
        );
    }

    #[test]
    fn a_state_this_build_cannot_name_makes_residency_unknown_rather_than_empty() {
        // A future LM Studio adding a third state must not silently read as
        // "nothing is loaded".
        let read =
            inventory(r#"{"data":[{"id":"a","state":"loaded"},{"id":"b","state":"warming-up"}]}"#);
        assert_eq!(read.installed, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            read.resident, None,
            "one unrecognised state makes the whole residency answer unknown"
        );
    }

    #[test]
    fn a_listing_with_no_states_still_lists_what_is_downloaded() {
        let read = inventory(r#"{"data":[{"id":"a"},{"id":"b"}]}"#);
        assert_eq!(read.installed, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            read.resident.as_deref(),
            Some([].as_slice()),
            "no state field is not an unknown state; nothing is claimed resident"
        );
    }

    #[test]
    fn unknown_fields_do_not_break_a_newer_lm_studio() {
        let read = inventory(
            r#"{"object":"list","data":[{"id":"a","state":"loaded","brand_new":1}],"also_new":true}"#,
        );
        assert_eq!(read.installed, vec!["a".to_string()]);
    }

    #[test]
    fn an_entry_without_an_id_is_dropped_rather_than_listed_blank() {
        let read = inventory(r#"{"data":[{"state":"loaded"},{"id":"a","state":"loaded"}]}"#);
        assert_eq!(read.installed, vec!["a".to_string()]);
        assert_eq!(read.resident.as_deref(), Some(["a".to_string()].as_slice()));
    }
}
