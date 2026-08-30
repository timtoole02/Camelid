//! UI-agnostic chat session core shared by the inline REPL and the full-screen
//! TUI. It owns conversation state, sampling settings, the active model, and the
//! request shape — but never prints or draws. Each front end drives streaming via
//! [`Session::client`] + [`Session::build_request`] and renders the result.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::client::{Client, CompatRow, LoadOutcome, LoadedInfo};
use super::models::{self, PickerRow};

/// Set by the SIGINT handler while a stream is in flight so the read loop can
/// abort cleanly (Ctrl-C cancels the generation, not the session). Shared by both
/// front ends.
pub static CANCEL: AtomicBool = AtomicBool::new(false);

/// Live sampling controls, all adjustable in-session via `/set`.
#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub max_tokens: u32,
    pub seed: Option<u64>,
    pub stream: bool,
    /// Opt-in thinking mode: send `camelid_enable_thinking:true` so the model
    /// emits its own `<think>…</think>` reasoning. NOT parity-locked (the
    /// leading-trace lane only); off by default.
    #[serde(default)]
    pub enable_thinking: bool,
}

impl Settings {
    /// A compact one-line summary for the status bar / `/info`.
    pub fn summary(&self) -> String {
        let opt_f = |v: Option<f32>| v.map(|n| format!("{n:.2}")).unwrap_or_else(|| "off".into());
        let opt_u = |v: Option<u32>| v.map(|n| n.to_string()).unwrap_or_else(|| "off".into());
        let opt_s = |v: Option<u64>| v.map(|n| n.to_string()).unwrap_or_else(|| "off".into());
        format!(
            "temp {:.2} · top-p {} · top-k {} · max {} · seed {} · stream {}{}",
            self.temperature,
            opt_f(self.top_p),
            opt_u(self.top_k),
            self.max_tokens,
            opt_s(self.seed),
            if self.stream { "on" } else { "off" },
            if self.enable_thinking {
                " · thinking on (experimental)"
            } else {
                ""
            },
        )
    }
}

/// Who authored a turn.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn display(self) -> &'static str {
        match self {
            Role::User => "You",
            Role::Assistant => "Camelid",
        }
    }
    fn api(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One conversation turn.
#[derive(Clone, Serialize, Deserialize)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

/// Result of a load attempt — `Unsupported` carries the engine's typed message.
pub enum LoadResult {
    Loaded,
    Unsupported(String),
}

/// On-disk session format for `/save` and `/load`.
#[derive(Serialize, Deserialize)]
struct SavedSession {
    label: String,
    system: Option<String>,
    settings: Settings,
    history: Vec<Turn>,
}

pub struct Session {
    client: Client,
    models_dir: PathBuf,
    ledger: Vec<CompatRow>,
    pub settings: Settings,
    pub system: Option<String>,
    pub history: Vec<Turn>,
    pub active_id: Option<String>,
    pub active_label: String,
    pub active_posture: String,
    /// Exact identity reported by the server for the loaded artifact. These are
    /// separate from display/request ids because capability receipts attach to
    /// GGUF bytes, not a user-controlled alias.
    active_filename: Option<String>,
    active_gguf_sha256: Option<String>,
    /// Training context length of the active model (for the context gauge).
    pub active_ctx: Option<u32>,
    pub last_prompt_tokens: Option<u32>,
    pub last_completion_tokens: Option<u32>,
}

impl Session {
    pub fn new(
        client: Client,
        models_dir: PathBuf,
        settings: Settings,
        system: Option<String>,
    ) -> Self {
        let ledger = client.capabilities().unwrap_or_default();
        Self {
            client,
            models_dir,
            ledger,
            settings,
            system,
            history: Vec::new(),
            active_id: None,
            active_label: String::new(),
            active_posture: String::new(),
            active_filename: None,
            active_gguf_sha256: None,
            active_ctx: None,
            last_prompt_tokens: None,
            last_completion_tokens: None,
        }
    }

    /// Models currently loaded in the server (for the instant switcher).
    pub fn loaded_models(&self) -> Vec<LoadedInfo> {
        self.client.list_loaded()
    }

    /// Switch the active model to one already loaded in the server. This is
    /// instant — the next request's `model` field re-activates it server-side
    /// with no reload. History resets (different context window).
    pub fn switch_to_loaded(&mut self, info: &LoadedInfo) {
        self.active_ctx = info.context_length();
        let posture = self.posture_for(&info.id);
        self.set_active(info.id.clone(), info.id.clone(), posture);
        self.active_filename = info.filename.clone();
        self.active_gguf_sha256 = (!info.gguf_sha256.is_empty()).then(|| info.gguf_sha256.clone());
    }

    pub fn client(&self) -> Client {
        self.client.clone()
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// The supported rows for the picker (ledger-derived, joined to the catalog).
    pub fn supported_rows(&self) -> Vec<PickerRow> {
        models::supported_rows(&self.ledger, &self.models_dir)
    }

    pub fn has_model(&self) -> bool {
        self.active_id.is_some()
    }

    pub fn active_gguf_sha256(&self) -> Option<&str> {
        self.active_gguf_sha256.as_deref()
    }

    pub fn generation_ready(&self) -> bool {
        self.client
            .health()
            .map(|h| h.generation_ready)
            .unwrap_or(false)
    }

    /// The ledger `family` of the active model (drives tool-call parsing), or "".
    pub fn active_family(&self) -> String {
        if let Some(row_id) = self.active_artifact_row() {
            if let Some(row) = self.ledger.iter().find(|row| row.id == row_id) {
                return row.family.clone();
            }
        }
        self.ledger
            .iter()
            .find(|row| {
                row.id == self.active_label || Some(row.id.as_str()) == self.active_id.as_deref()
            })
            .map(|row| row.family.clone())
            .unwrap_or_default()
    }

    /// True only when the active model matches a ledger row verified for
    /// tool-calling *and* the server reports an artifact filename/digest that
    /// resolves to that same row. An alias or filename alone cannot make
    /// different bytes inherit another artifact's agent receipt.
    pub fn active_tool_capable(&self) -> bool {
        let Some(artifact_row) = self.active_artifact_row() else {
            return false;
        };
        self.ledger
            .iter()
            .any(|row| row.tool_capable && row.id == artifact_row)
    }

    fn active_artifact_row(&self) -> Option<&'static str> {
        let filename = self.active_filename.as_deref()?;
        let sha256 = self.active_gguf_sha256.as_deref()?;
        crate::api::workspace::tool_capable_row_for_loaded_artifact(filename, sha256)
            .map(|(row, _)| row)
    }

    /// Every ledger row promoted for agent mode — for refusal messages that
    /// tell the user what WOULD work instead of only what does not.
    pub fn tool_capable_rows(&self) -> Vec<String> {
        self.ledger
            .iter()
            .filter(|r| {
                r.tool_capable
                    && crate::api::workspace::tool_capable_row_has_certified_artifact(&r.id)
            })
            .map(|r| r.id.clone())
            .collect()
    }

    /// Support posture for a model id, read from the ledger ("supported" or the
    /// raw status; "loaded" when the id is not a ledger row).
    pub fn posture_for(&self, id: &str) -> String {
        self.ledger
            .iter()
            .find(|row| row.id == id)
            .map(|row| {
                if row.status.starts_with("supported") {
                    "supported".to_string()
                } else {
                    row.status.clone()
                }
            })
            .unwrap_or_else(|| "loaded".to_string())
    }

    /// Load a GGUF and, on success, make it active. `label`/`posture` override the
    /// display (the picker passes the ledger row id + "supported"); when `None`
    /// the server-assigned id and a ledger lookup are used. Returns
    /// `LoadResult::Unsupported` (without changing the active model) for the typed
    /// gate or a non-generation-ready load.
    pub fn load_model_file(
        &mut self,
        path: &Path,
        label: Option<&str>,
        posture: Option<&str>,
    ) -> anyhow::Result<LoadResult> {
        let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        match self.client.load_model(&abs.to_string_lossy(), None)? {
            LoadOutcome::Unsupported { message } => Ok(LoadResult::Unsupported(message)),
            LoadOutcome::Loaded { id } => {
                if !self.generation_ready() {
                    return Ok(LoadResult::Unsupported(format!(
                        "loaded model '{id}' is not generation-ready on this Camelid build"
                    )));
                }
                let posture = posture
                    .map(str::to_string)
                    .unwrap_or_else(|| self.posture_for(&id));
                let label = label.map(str::to_string).unwrap_or_else(|| id.clone());
                let loaded = self.loaded_models().into_iter().find(|m| m.id == id);
                self.active_ctx = loaded.as_ref().and_then(LoadedInfo::context_length);
                self.set_active(id, label, posture);
                if let Some(info) = loaded {
                    self.active_filename = info.filename;
                    self.active_gguf_sha256 =
                        (!info.gguf_sha256.is_empty()).then_some(info.gguf_sha256);
                }
                Ok(LoadResult::Loaded)
            }
        }
    }

    /// Make a model active and reset history (a different model is a different
    /// context window). The system prompt is session-level and is preserved.
    pub fn set_active(&mut self, request_id: String, label: String, posture: String) {
        self.history.clear();
        self.active_id = Some(request_id);
        self.active_label = label;
        self.active_posture = posture;
        self.active_filename = None;
        self.active_gguf_sha256 = None;
    }

    pub fn reset_history(&mut self) {
        self.history.clear();
    }

    pub fn push_user(&mut self, content: String) {
        self.history.push(Turn {
            role: Role::User,
            content,
        });
    }

    pub fn push_assistant(&mut self, content: String) {
        self.history.push(Turn {
            role: Role::Assistant,
            content,
        });
    }

    /// Drop the last turn (used to discard a cancelled/failed in-flight turn).
    pub fn pop_last(&mut self) {
        self.history.pop();
    }

    /// The text of the most recent user turn, if any (for `/retry`).
    pub fn last_user_message(&self) -> Option<String> {
        self.history
            .iter()
            .rev()
            .find(|t| t.role == Role::User)
            .map(|t| t.content.clone())
    }

    /// Build an OpenAI-style chat request from the system prompt, history, and
    /// live sampling settings.
    pub fn build_request(&self, stream: bool) -> Value {
        let mut messages: Vec<Value> = Vec::new();
        if let Some(system) = &self.system {
            messages.push(json!({ "role": "system", "content": system }));
        }
        for turn in &self.history {
            messages.push(json!({ "role": turn.role.api(), "content": turn.content }));
        }
        let mut request = json!({
            "model": self.active_id,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
        });
        if let Some(top_p) = self.settings.top_p {
            request["top_p"] = json!(top_p);
        }
        if let Some(top_k) = self.settings.top_k {
            request["top_k"] = json!(top_k);
        }
        if let Some(seed) = self.settings.seed {
            request["seed"] = json!(seed);
        }
        if self.settings.enable_thinking {
            // Opt-in, NOT parity-locked: leading-trace lane only. Silent by
            // default so the parity-locked thinking-DISABLED rendering stands.
            request["camelid_enable_thinking"] = json!(true);
        }
        request
    }

    /// Apply a `/set <name> <value>` change. Returns a confirmation string, or an
    /// error string describing the problem.
    pub fn set_param(&mut self, name: &str, value: &str) -> Result<String, String> {
        let off = matches!(
            value.to_ascii_lowercase().as_str(),
            "off" | "none" | "default"
        );
        match name.to_ascii_lowercase().replace('-', "_").as_str() {
            "temperature" | "temp" => {
                let v: f32 = value.parse().map_err(|_| "temperature must be a number".to_string())?;
                if v < 0.0 {
                    return Err("temperature must be >= 0".into());
                }
                self.settings.temperature = v;
                Ok(format!("temperature = {v}"))
            }
            "top_p" => {
                self.settings.top_p = if off {
                    None
                } else {
                    Some(value.parse().map_err(|_| "top_p must be a number".to_string())?)
                };
                Ok(format!("top_p = {}", value_or_off(off, value)))
            }
            "top_k" => {
                self.settings.top_k = if off {
                    None
                } else {
                    Some(value.parse().map_err(|_| "top_k must be an integer".to_string())?)
                };
                Ok(format!("top_k = {}", value_or_off(off, value)))
            }
            "max_tokens" | "max" => {
                let v: u32 = value.parse().map_err(|_| "max_tokens must be an integer".to_string())?;
                if v == 0 {
                    return Err("max_tokens must be >= 1".into());
                }
                self.settings.max_tokens = v;
                Ok(format!("max_tokens = {v}"))
            }
            "seed" => {
                self.settings.seed = if off {
                    None
                } else {
                    Some(value.parse().map_err(|_| "seed must be an integer".to_string())?)
                };
                Ok(format!("seed = {}", value_or_off(off, value)))
            }
            "stream" => {
                self.settings.stream = match value.to_ascii_lowercase().as_str() {
                    "on" | "true" | "1" | "yes" => true,
                    "off" | "false" | "0" | "no" => false,
                    _ => return Err("stream must be on/off".into()),
                };
                Ok(format!("stream = {}", self.settings.stream))
            }
            "thinking" | "enable_thinking" => {
                // Opt-in, NOT parity-locked: the leading-trace lane only.
                self.settings.enable_thinking = match value.to_ascii_lowercase().as_str() {
                    "on" | "true" | "1" | "yes" => true,
                    "off" | "false" | "0" | "no" => false,
                    _ => return Err("thinking must be on/off".into()),
                };
                Ok(format!(
                    "thinking = {} (experimental — not parity-locked)",
                    if self.settings.enable_thinking { "on" } else { "off" }
                ))
            }
            other => Err(format!(
                "unknown setting '{other}' — try temperature, top_p, top_k, max_tokens, seed, stream, thinking"
            )),
        }
    }

    /// Save the session (settings + system + transcript) as JSON.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let saved = SavedSession {
            label: self.active_label.clone(),
            system: self.system.clone(),
            settings: self.settings.clone(),
            history: self.history.clone(),
        };
        let mut json = serde_json::to_string_pretty(&saved)?;
        json.push('\n');
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a saved session's settings + system + transcript (keeps the active
    /// model — the transcript continues against whatever is loaded now).
    pub fn load(&mut self, path: &Path) -> anyhow::Result<()> {
        let raw = std::fs::read_to_string(path)?;
        let saved: SavedSession = serde_json::from_str(&raw)?;
        self.settings = saved.settings;
        self.system = saved.system;
        self.history = saved.history;
        Ok(())
    }

    /// Key/value rows describing the active model, for `/info`.
    pub fn model_info(&self) -> Vec<(String, String)> {
        let mut rows = Vec::new();
        rows.push(("model".into(), self.active_label.clone()));
        rows.push(("posture".into(), self.active_posture.clone()));
        if let Some(value) = self.client.current_model() {
            if let Some(arch) = value
                .pointer("/gguf/metadata/general.architecture")
                .and_then(Value::as_str)
            {
                rows.push(("architecture".into(), arch.to_string()));
            }
            if let Some(ctx) = value
                .pointer("/llama_config/context_length")
                .and_then(Value::as_u64)
            {
                rows.push(("context_length".into(), ctx.to_string()));
            }
            if let Some(path) = value.get("path").and_then(Value::as_str) {
                rows.push(("path".into(), path.to_string()));
            }
        }
        rows.push(("settings".into(), self.settings.summary()));
        rows
    }
}

fn value_or_off(off: bool, value: &str) -> String {
    if off {
        "off".to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn session() -> Session {
        let client = Client::new("127.0.0.1:1".parse::<SocketAddr>().unwrap());
        Session {
            client,
            models_dir: PathBuf::from("models"),
            ledger: Vec::new(),
            settings: Settings {
                temperature: 0.0,
                top_p: None,
                top_k: None,
                max_tokens: 512,
                seed: None,
                stream: true,
                enable_thinking: false,
            },
            system: None,
            history: Vec::new(),
            active_id: Some("m".into()),
            active_label: "m".into(),
            active_posture: "loaded".into(),
            active_filename: None,
            active_gguf_sha256: None,
            active_ctx: None,
            last_prompt_tokens: None,
            last_completion_tokens: None,
        }
    }

    #[test]
    fn set_param_updates_and_validates() {
        let mut s = session();
        assert!(s.set_param("temperature", "0.7").is_ok());
        assert_eq!(s.settings.temperature, 0.7);
        assert!(s.set_param("top_p", "0.9").is_ok());
        assert_eq!(s.settings.top_p, Some(0.9));
        assert!(s.set_param("top_k", "off").is_ok());
        assert_eq!(s.settings.top_k, None);
        assert!(s.set_param("stream", "off").is_ok());
        assert!(!s.settings.stream);
        assert!(s.set_param("temperature", "-1").is_err());
        assert!(s.set_param("bogus", "1").is_err());
    }

    #[test]
    fn agent_gate_requires_the_certified_loaded_artifact_identity() {
        let mut s = session();
        s.active_id = Some("llama32_3b_instruct_q8_0".into());
        s.active_label = "llama32_3b_instruct_q8_0".into();
        s.ledger = vec![CompatRow {
            id: "llama32_3b_instruct_q8_0".into(),
            family: "llama_bpe_decoder".into(),
            quantization: "Q8_0".into(),
            status: "supported_exact_row_smoke".into(),
            tool_capable: true,
        }];
        s.active_filename = Some("Llama-3.2-3B-Instruct-Q8_0.gguf".into());
        s.active_gguf_sha256 =
            Some("f34112a11b7dad74ab517dedf6dcf00d624c9adac2dc0c72c719ca0478554ef2".into());
        assert!(s.active_tool_capable());

        // The server id may be a startup hash or another local alias. Exact
        // bytes, not that alias, are what earned the receipt.
        s.active_id = Some("sha256:local-runtime-id".into());
        s.active_label = "local-runtime-id".into();
        assert!(s.active_tool_capable());
        assert_eq!(s.active_family(), "llama_bpe_decoder");

        s.active_gguf_sha256 = Some("00".repeat(32));
        assert!(!s.active_tool_capable(), "same name with different bytes");

        s.active_gguf_sha256 = None;
        assert!(
            !s.active_tool_capable(),
            "missing identity must fail closed"
        );
    }

    #[test]
    fn agent_refusal_list_omits_unpinned_tool_rows() {
        let mut s = session();
        s.ledger = vec![
            CompatRow {
                id: "qwen3_4b_instruct_q8_0".into(),
                family: "qwen".into(),
                quantization: "Q8_0".into(),
                status: "supported_exact_row_smoke".into(),
                tool_capable: true,
            },
            CompatRow {
                id: "qwen3_4b_q4_k_m".into(),
                family: "qwen".into(),
                quantization: "Q4_K_M".into(),
                status: "supported_exact_row_smoke".into(),
                tool_capable: true,
            },
        ];
        assert_eq!(s.tool_capable_rows(), vec!["qwen3_4b_q4_k_m"]);
    }

    #[test]
    fn build_request_includes_set_sampling_params() {
        let mut s = session();
        s.set_param("top_p", "0.8").unwrap();
        s.set_param("seed", "42").unwrap();
        s.push_user("hi".into());
        let req = s.build_request(true);
        assert_eq!(req["stream"], true);
        // top_p was set as f32, so compare with a tolerance after the f64 widen.
        assert!((req["top_p"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(req["seed"], 42);
        assert_eq!(req["messages"][0]["role"], "user");
        assert_eq!(req["messages"][0]["content"], "hi");
    }

    #[test]
    fn thinking_flag_only_sent_when_opted_in() {
        let mut s = session();
        s.push_user("hi".into());
        // Default OFF: the request stays silent so the server keeps the
        // parity-locked thinking-DISABLED rendering.
        assert!(s
            .build_request(true)
            .get("camelid_enable_thinking")
            .is_none());
        // /set thinking on flips it; the request then carries the opt-in flag.
        let msg = s.set_param("thinking", "on").unwrap();
        assert!(msg.contains("not parity-locked"));
        assert!(s.settings.enable_thinking);
        assert_eq!(s.build_request(true)["camelid_enable_thinking"], true);
        // /set thinking off returns to silence.
        s.set_param("thinking", "off").unwrap();
        assert!(s
            .build_request(true)
            .get("camelid_enable_thinking")
            .is_none());
    }

    #[test]
    fn save_then_load_round_trips_transcript_and_settings() {
        let dir = std::env::temp_dir();
        let path = dir.join("camelid-chat-session-test.json");
        let mut s = session();
        s.system = Some("be brief".into());
        s.set_param("temperature", "0.5").unwrap();
        s.push_user("q".into());
        s.push_assistant("a".into());
        s.save(&path).unwrap();

        let mut s2 = session();
        s2.load(&path).unwrap();
        assert_eq!(s2.system.as_deref(), Some("be brief"));
        assert_eq!(s2.settings.temperature, 0.5);
        assert_eq!(s2.history.len(), 2);
        assert_eq!(s2.history[1].content, "a");
        let _ = std::fs::remove_file(&path);
    }
}
