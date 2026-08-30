use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use super::{
    api_error, capabilities_response_with_plan, curated_catalog,
    supported_artifact_expected_sha256, AppState, CatalogItemView, LoadedModel,
    NON_CATALOG_SUPPORTED_ARTIFACTS,
};
use crate::chat::agent::LoopEnd;
use crate::chat::workspace_bridge::{
    bridge, run_live, WorkspaceBridgeControl, WorkspaceBridgeWorker, WorkspaceDecisionKind,
    WorkspaceEvent, WorkspaceRunConfig,
};
use crate::chat::workspace_memory::{
    default_store_path, EvidenceInput, StoredThread, WorkspaceMemoryStore,
};

const EVENT_BACKLOG: usize = 128;
const EVENT_STREAM_BUFFER: usize = 128;
const DEFAULT_MAX_STEPS: usize = 12;
const MAX_STEPS: usize = 32;
const DEFAULT_MAX_TOKENS: u32 = 512;
const MAX_TOKENS: u32 = 1024;
const MAX_GOAL_BYTES: usize = 4 * 1024;
const EVENT_CLAIM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const AUTO_COMPACT_TRIGGER_PERCENT: u32 = 75;
const AUTO_COMPACT_MIN_TURNS: u32 = 4;
const WORKSPACE_MODEL_STEP_TIMEOUT_ENV: &str = "CAMELID_WORKSPACE_MODEL_STEP_TIMEOUT_SECS";
const DEFAULT_WORKSPACE_MODEL_STEP_TIMEOUT_SECS: u64 = 15 * 60;
const MAX_WORKSPACE_MODEL_STEP_TIMEOUT_SECS: u64 = 60 * 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkspaceRuntimeBudget {
    context_tokens: u32,
    reply_tokens: u32,
}

fn workspace_runtime_budget(
    model_context_tokens: u32,
    server_prompt_ceiling: usize,
    server_generation_ceiling: u32,
    requested_reply_tokens: u32,
) -> WorkspaceRuntimeBudget {
    let reply_tokens = requested_reply_tokens.min(server_generation_ceiling);
    let prompt_ceiling = u32::try_from(server_prompt_ceiling).unwrap_or(u32::MAX);
    let context_tokens = model_context_tokens
        .min(crate::chat::agent::AGENT_VALIDATED_CTX)
        .min(prompt_ceiling.saturating_add(reply_tokens));
    WorkspaceRuntimeBudget {
        context_tokens,
        reply_tokens,
    }
}

fn parse_workspace_model_step_timeout(raw: Option<&str>) -> Duration {
    let seconds = raw
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| (1..=MAX_WORKSPACE_MODEL_STEP_TIMEOUT_SECS).contains(seconds))
        .unwrap_or(DEFAULT_WORKSPACE_MODEL_STEP_TIMEOUT_SECS);
    Duration::from_secs(seconds)
}

fn workspace_model_step_timeout() -> Duration {
    let configured = std::env::var(WORKSPACE_MODEL_STEP_TIMEOUT_ENV).ok();
    parse_workspace_model_step_timeout(configured.as_deref())
}

fn workspace_model_identity_matches(
    expected_id: &str,
    expected_sha256: &str,
    active_id: &str,
    active_sha256: &str,
) -> bool {
    expected_id == active_id && expected_sha256.eq_ignore_ascii_case(active_sha256)
}

async fn run_workspace_blocking<T, F>(operation: F) -> Result<T, Response>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "workspace_blocking_task_failed",
                format!("Workspace background operation failed: {error}"),
                None,
            )
        })
}

fn should_auto_compact(
    turn_count: u32,
    prompt_tokens: u32,
    generation_tokens: u32,
    budget_total: u32,
) -> bool {
    if turn_count < AUTO_COMPACT_MIN_TURNS || budget_total == 0 {
        return false;
    }
    u64::from(prompt_tokens.saturating_add(generation_tokens)) * 100
        >= u64::from(budget_total) * u64::from(AUTO_COMPACT_TRIGGER_PERCENT)
}

/// Forwarding runs on a dedicated blocking thread. A full bounded queue is
/// backpressure, not a client disconnect; only receiver closure is terminal.
/// Any compaction generated while settling a turn must precede the terminal
/// event because the SSE consumer intentionally stops after that terminal.
fn forward_workspace_event(
    sender: &tokio::sync::mpsc::Sender<WorkspaceEvent>,
    event: WorkspaceEvent,
    before_terminal: Option<WorkspaceEvent>,
) -> Result<(), ()> {
    if let Some(prefix) = before_terminal {
        sender.blocking_send(prefix).map_err(|_| ())?;
    }
    sender.blocking_send(event).map_err(|_| ())
}

#[derive(Clone, Default)]
pub(super) struct WorkspaceSessionManager {
    active: Arc<Mutex<Option<Arc<ActiveWorkspaceSession>>>>,
}

struct ActiveWorkspaceSession {
    id: String,
    workspace: PathBuf,
    model_id: String,
    model_sha256: String,
    max_steps: usize,
    max_tokens: u32,
    temperature: f32,
    context_budget_tokens: u32,
    model_step_timeout: Duration,
    allow_writes: bool,
    semantic_retriever: Option<Arc<crate::chat::semantic_search::WorkspaceSemanticRetriever>>,
    memory: WorkspaceMemoryStore,
    state: StdMutex<WorkspaceSessionState>,
    events: StdMutex<Option<std::sync::mpsc::Receiver<WorkspaceEvent>>>,
    worker: StdMutex<Option<WorkspaceBridgeWorker>>,
    run_config: StdMutex<Option<WorkspaceRunConfig>>,
    control: StdMutex<Option<WorkspaceBridgeControl>>,
    current_turn: StdMutex<Option<(String, u32)>>,
}

enum InstallTurn {
    Installed,
    Duplicate(u32),
}

#[derive(Clone, Copy)]
enum TurnCompletion {
    Idle,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceSessionState {
    WaitingForEvents,
    Running,
    Idle,
    Cancelling,
    Cancelled,
    Failed,
}

impl WorkspaceSessionState {
    fn as_str(self) -> &'static str {
        match self {
            Self::WaitingForEvents => "waiting_for_events",
            Self::Running => "running",
            Self::Idle => "idle",
            Self::Cancelling => "cancelling",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }

    fn blocks_model_transition(self) -> bool {
        matches!(
            self,
            Self::WaitingForEvents | Self::Running | Self::Cancelling
        )
    }

    fn accepts_new_turn(self) -> bool {
        matches!(self, Self::Idle | Self::Cancelled | Self::Failed)
    }

    fn after_cancel_request(self) -> Self {
        match self {
            Self::WaitingForEvents => Self::Cancelled,
            Self::Running => Self::Cancelling,
            Self::Idle => Self::Cancelled,
            other => other,
        }
    }

    fn after_events_claimed(self) -> Self {
        match self {
            Self::WaitingForEvents => Self::Running,
            other => other,
        }
    }
}

impl WorkspaceSessionManager {
    pub(super) async fn blocks_model_transition(&self) -> bool {
        let active = self.active.lock().await.clone();
        active.is_some_and(|session| {
            session
                .state
                .lock()
                .map(|state| state.blocks_model_transition())
                .unwrap_or(true)
        })
    }

    fn active_state(active: &Option<Arc<ActiveWorkspaceSession>>) -> Option<WorkspaceSessionState> {
        active.as_ref().map(|session| {
            session
                .state
                .lock()
                .map(|state| *state)
                .unwrap_or(WorkspaceSessionState::Running)
        })
    }
}

impl ActiveWorkspaceSession {
    fn expire_unclaimed_turn(&self, message_id: &str) -> anyhow::Result<bool> {
        let (Ok(mut status), Ok(mut current_turn)) = (self.state.lock(), self.current_turn.lock())
        else {
            anyhow::bail!("Workspace turn state is unavailable");
        };
        if *status != WorkspaceSessionState::WaitingForEvents
            || current_turn
                .as_ref()
                .is_none_or(|(current_id, _)| current_id != message_id)
        {
            return Ok(false);
        }
        let config = self
            .run_config
            .lock()
            .map_err(|_| anyhow::anyhow!("Workspace turn configuration is unavailable"))?
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Workspace turn configuration is missing"))?;
        self.memory.append_terminal_turn(
            &self.id,
            &config.client_message_id,
            &config.goal,
            "",
            "aborted",
            &[],
        )?;
        if let Some(control) = self.control.lock().ok().and_then(|control| control.clone()) {
            control.cancel();
        }
        *status = WorkspaceSessionState::Cancelled;
        *current_turn = None;
        Ok(true)
    }

    fn pending_message(&self, client_message_id: &str) -> Option<u32> {
        self.current_turn
            .lock()
            .ok()
            .and_then(|turn| turn.as_ref().cloned())
            .filter(|(message_id, _)| message_id == client_message_id)
            .map(|(_, turn_index)| turn_index)
    }

    fn install_turn(
        &self,
        events: std::sync::mpsc::Receiver<WorkspaceEvent>,
        worker: WorkspaceBridgeWorker,
        run_config: WorkspaceRunConfig,
        control: WorkspaceBridgeControl,
    ) -> Result<InstallTurn, &'static str> {
        let mut status = self
            .state
            .lock()
            .map_err(|_| "thread state is unavailable")?;
        let mut current_turn = self
            .current_turn
            .lock()
            .map_err(|_| "turn identity is unavailable")?;
        if !status.accepts_new_turn() {
            if let Some((message_id, turn_index)) = current_turn.as_ref() {
                if message_id == &run_config.client_message_id {
                    return Ok(InstallTurn::Duplicate(*turn_index));
                }
            }
            return Err("a turn is already active");
        }
        let mut event_slot = self
            .events
            .lock()
            .map_err(|_| "event slot is unavailable")?;
        let mut worker_slot = self
            .worker
            .lock()
            .map_err(|_| "worker slot is unavailable")?;
        let mut config_slot = self
            .run_config
            .lock()
            .map_err(|_| "turn configuration is unavailable")?;
        let mut control_slot = self
            .control
            .lock()
            .map_err(|_| "turn control is unavailable")?;
        *current_turn = Some((run_config.client_message_id.clone(), run_config.turn_index));
        *event_slot = Some(events);
        *worker_slot = Some(worker);
        *config_slot = Some(run_config);
        *control_slot = Some(control);
        *status = WorkspaceSessionState::WaitingForEvents;
        Ok(InstallTurn::Installed)
    }

    fn finish_turn_if_current(&self, message_id: &str, completion: TurnCompletion) -> bool {
        let (Ok(mut status), Ok(mut current_turn)) = (self.state.lock(), self.current_turn.lock())
        else {
            return false;
        };
        let owns_turn = current_turn
            .as_ref()
            .is_some_and(|(current_id, _)| current_id == message_id);
        if !owns_turn {
            return false;
        }
        *status = if matches!(
            *status,
            WorkspaceSessionState::Cancelling | WorkspaceSessionState::Cancelled
        ) {
            WorkspaceSessionState::Cancelled
        } else {
            match completion {
                TurnCompletion::Idle => WorkspaceSessionState::Idle,
                TurnCompletion::Failed => WorkspaceSessionState::Failed,
            }
        };
        *current_turn = None;
        true
    }

    fn persist_aborted_turn_and_finish(
        &self,
        run_config: &WorkspaceRunConfig,
        evidence: &[EvidenceInput],
    ) -> anyhow::Result<bool> {
        let persisted = self.memory.append_terminal_turn(
            &self.id,
            &run_config.client_message_id,
            &run_config.goal,
            "",
            "aborted",
            evidence,
        );
        let finished = self.finish_turn_if_current(
            &run_config.client_message_id,
            if persisted.is_ok() {
                TurnCompletion::Idle
            } else {
                TurnCompletion::Failed
            },
        );
        persisted?;
        Ok(finished)
    }
}

fn arm_event_claim_deadline(session: &Arc<ActiveWorkspaceSession>, message_id: String) {
    let session = Arc::downgrade(session);
    tokio::spawn(async move {
        tokio::time::sleep(EVENT_CLAIM_TIMEOUT).await;
        let Some(session) = session.upgrade() else {
            return;
        };
        let expiry_session = Arc::clone(&session);
        let expiry =
            tokio::task::spawn_blocking(move || expiry_session.expire_unclaimed_turn(&message_id))
                .await;
        if let Err(error) = expiry
            .map_err(anyhow::Error::from)
            .and_then(|result| result)
        {
            eprintln!("Workspace event-claim timeout could not persist the turn: {error}");
            if let Ok(mut status) = session.state.lock() {
                *status = WorkspaceSessionState::Failed;
            }
        }
    });
}

#[derive(Debug, Deserialize)]
pub(super) struct CreateWorkspaceSessionRequest {
    workspace: PathBuf,
    goal: String,
    #[serde(default)]
    thread_id: Option<String>,
    #[serde(default)]
    max_steps: Option<usize>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    allow_writes: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(super) struct WorkspaceMessageRequest {
    text: String,
    client_message_id: String,
}

#[derive(Debug, Serialize)]
struct WorkspaceSessionResponse {
    id: String,
    workspace: String,
    model_id: String,
    state: &'static str,
    max_steps: usize,
    max_tokens: u32,
    allow_writes: bool,
    semantic_retrieval: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_model_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct WorkspaceSessionStatusResponse {
    id: String,
    workspace: String,
    model_id: String,
    state: &'static str,
    context_budget_tokens: u32,
    resident_cuda: Option<crate::inference::ResidentCudaStatus>,
    allow_writes: bool,
    semantic_retrieval: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_model_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct WorkspaceMessageResponse {
    session_id: String,
    turn_index: u32,
    state: &'static str,
    duplicate: bool,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct WorkspaceModelOption {
    row_id: &'static str,
    name: String,
    filename: &'static str,
    quantization: &'static str,
    installed: bool,
    catalog_id: Option<&'static str>,
    fit: crate::fit::FitVerdict,
    fit_confidence: &'static str,
}

#[derive(Debug, Serialize)]
struct WorkspaceModelsResponse {
    models: Vec<WorkspaceModelOption>,
}

#[derive(Debug, Deserialize)]
pub(super) struct WorkspaceThreadsQuery {
    workspace: PathBuf,
}

#[derive(Debug, Serialize)]
struct WorkspaceThreadsResponse {
    threads: Vec<StoredThread>,
}

#[derive(Debug, Serialize)]
struct WorkspaceThreadResponse {
    thread: StoredThread,
    turns: Vec<crate::chat::workspace_memory::StoredTurn>,
}

#[derive(Debug, Deserialize)]
pub(super) struct WorkspaceDecisionRequest {
    approval_id: String,
    decision: WorkspaceDecisionKind,
}

#[derive(Debug, Serialize)]
struct WorkspaceEventEnvelope {
    sequence: u64,
    session_id: String,
    #[serde(flatten)]
    event: WorkspaceEvent,
}

struct CancelStreamOnDrop(WorkspaceBridgeControl);

impl Drop for CancelStreamOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub(super) async fn compatible_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }

    let models_dir = state.models_dir.clone();
    match run_workspace_blocking(move || workspace_model_options(&models_dir)).await {
        Ok(models) => Json(WorkspaceModelsResponse { models }).into_response(),
        Err(response) => response,
    }
}

pub(super) async fn list_threads(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<WorkspaceThreadsQuery>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let requested_workspace = query.workspace;
    let workspace = match run_workspace_blocking(move || {
        std::fs::canonicalize(requested_workspace)
            .ok()
            .filter(|path| path.is_dir())
            .map(|path| simplify_path(&path))
    })
    .await
    {
        Ok(Some(workspace)) => workspace,
        Ok(None) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "workspace_root_not_accessible",
                "workspace must name an accessible local directory".to_string(),
                Some("workspace"),
            )
        }
        Err(response) => return response,
    };
    let (model, _) = match active_tool_capable_model(&state).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let model_id = model.id.clone();
    let model_sha256 = model.lane.gguf_sha256.to_string();
    let result = match run_workspace_blocking(move || -> anyhow::Result<_> {
        let store = WorkspaceMemoryStore::open(default_store_path())?;
        let threads = store
            .threads_for_root(&workspace, 20)?
            .into_iter()
            .filter(|thread| thread.model_id == model_id && thread.model_sha256 == model_sha256)
            .collect();
        Ok(threads)
    })
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    match result {
        Ok(threads) => Json(WorkspaceThreadsResponse { threads }).into_response(),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_memory_unavailable",
            format!("Workspace threads could not be listed: {error}"),
            None,
        ),
    }
}

pub(super) async fn get_thread(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<WorkspaceThreadsQuery>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let workspace = query.workspace;
    let result = match run_workspace_blocking(move || -> anyhow::Result<_> {
        let workspace = match std::fs::canonicalize(workspace) {
            Ok(path) if path.is_dir() => simplify_path(&path),
            _ => return Ok(None),
        };
        let store = WorkspaceMemoryStore::open(default_store_path())?;
        let Some(thread) = store
            .thread(&id)?
            .filter(|thread| thread.canonical_root == workspace)
        else {
            return Ok(Some(None));
        };
        let turns = store.recent_turns(&id, 200)?;
        Ok(Some(Some(WorkspaceThreadResponse { thread, turns })))
    })
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    match result {
        Ok(Some(Some(thread))) => Json(thread).into_response(),
        Ok(Some(None)) => workspace_not_found(),
        Ok(None) => api_error(
            StatusCode::BAD_REQUEST,
            "workspace_root_not_accessible",
            "workspace must name an accessible local directory".to_string(),
            Some("workspace"),
        ),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_memory_unavailable",
            format!("Workspace transcript could not be loaded: {error}"),
            None,
        ),
    }
}

pub(super) async fn delete_thread(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<WorkspaceThreadsQuery>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    if let Some(session) = state.workspace_sessions.active.lock().await.as_ref() {
        if session.id == id {
            let terminal = session
                .state
                .lock()
                .map(|state| {
                    matches!(
                        *state,
                        WorkspaceSessionState::Cancelled | WorkspaceSessionState::Failed
                    )
                })
                .unwrap_or(false);
            if !terminal {
                return api_error(
                    StatusCode::CONFLICT,
                    "workspace_thread_active",
                    "clear the active Workspace thread before deleting its saved memory"
                        .to_string(),
                    None,
                );
            }
        }
    }
    let workspace = query.workspace;
    let result = match run_workspace_blocking(move || -> anyhow::Result<_> {
        let workspace = match std::fs::canonicalize(workspace) {
            Ok(path) if path.is_dir() => simplify_path(&path),
            _ => return Ok(None),
        };
        let store = WorkspaceMemoryStore::open(default_store_path())?;
        if store
            .thread(&id)?
            .is_none_or(|thread| thread.canonical_root != workspace)
        {
            return Ok(Some(false));
        }
        Ok(Some(store.delete_thread(&id)?))
    })
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    match result {
        Ok(Some(true)) => StatusCode::NO_CONTENT.into_response(),
        Ok(Some(false)) => workspace_not_found(),
        Ok(None) => api_error(
            StatusCode::BAD_REQUEST,
            "workspace_root_not_accessible",
            "workspace must name an accessible local directory".to_string(),
            Some("workspace"),
        ),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_memory_unavailable",
            format!("Workspace memory could not delete this thread: {error}"),
            None,
        ),
    }
}

pub(super) async fn compact_thread(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<WorkspaceThreadsQuery>,
) -> Response {
    compact_thread_operation(state, headers, id, query, false).await
}

pub(super) async fn undo_thread_compaction(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<WorkspaceThreadsQuery>,
) -> Response {
    compact_thread_operation(state, headers, id, query, true).await
}

async fn compact_thread_operation(
    state: AppState,
    headers: HeaderMap,
    id: String,
    query: WorkspaceThreadsQuery,
    undo: bool,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    if let Some(session) = state.workspace_sessions.active.lock().await.as_ref() {
        if session.id == id {
            // Stopped or failed sessions hold no in-flight turn; the same states
            // that accept a follow-up turn must also allow manual compact/undo,
            // or a stopped session's memory controls stay dead until Reset.
            let available = session
                .state
                .lock()
                .map(|state| state.accepts_new_turn())
                .unwrap_or(false);
            if !available {
                return api_error(
                    StatusCode::CONFLICT,
                    "workspace_turn_active",
                    "wait for the active turn to finish before compacting this conversation"
                        .to_string(),
                    None,
                );
            }
        }
    }
    let workspace = query.workspace;
    let result = match run_workspace_blocking(move || -> anyhow::Result<_> {
        let workspace = match std::fs::canonicalize(workspace) {
            Ok(path) if path.is_dir() => simplify_path(&path),
            _ => return Ok(None),
        };
        let store = WorkspaceMemoryStore::open(default_store_path())?;
        if store
            .thread(&id)?
            .is_none_or(|thread| thread.canonical_root != workspace)
        {
            return Ok(Some(None));
        }
        let compaction = if undo {
            store.undo_compaction(&id)?
        } else {
            store.compact_thread(&id)?
        };
        Ok(Some(Some(compaction)))
    })
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    match result {
        Ok(Some(Some(compaction))) => Json(compaction).into_response(),
        Ok(Some(None)) => workspace_not_found(),
        Ok(None) => api_error(
            StatusCode::BAD_REQUEST,
            "workspace_root_not_accessible",
            "workspace must name an accessible local directory".to_string(),
            Some("workspace"),
        ),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_memory_unavailable",
            format!("Workspace memory compaction failed: {error}"),
            None,
        ),
    }
}

fn workspace_model_options(models_dir: &std::path::Path) -> Vec<WorkspaceModelOption> {
    let catalog = curated_catalog();
    let rows = tool_capable_compatibility_rows();
    let hardware = crate::capability::HardwareProfile::cached();
    let mut models = Vec::new();

    for item in &catalog {
        if supported_artifact_expected_sha256(item.filename).is_none() {
            continue;
        }
        if let Some(row) = rows.iter().find(|row| row.id == item.catalog_id) {
            let view = CatalogItemView::from_curated(item, hardware);
            models.push(WorkspaceModelOption {
                row_id: row.id,
                name: item.name.to_string(),
                filename: item.filename,
                quantization: row.quantization,
                installed: models_dir.join(item.filename).is_file(),
                catalog_id: Some(item.catalog_id),
                fit: view.fit,
                fit_confidence: view.fit_confidence,
            });
        }
    }

    for (filename, row_id, _) in NON_CATALOG_SUPPORTED_ARTIFACTS {
        if supported_artifact_expected_sha256(filename).is_none() {
            continue;
        }
        let Some(row) = rows.iter().find(|row| row.id == *row_id) else {
            continue;
        };
        if models.iter().any(|model| model.filename == *filename) {
            continue;
        }
        models.push(WorkspaceModelOption {
            row_id: row.id,
            name: filename.trim_end_matches(".gguf").to_string(),
            filename,
            quantization: row.quantization,
            installed: models_dir.join(filename).is_file(),
            catalog_id: None,
            fit: crate::fit::FitVerdict::Unknown,
            fit_confidence: "unknown",
        });
    }

    models.sort_by_key(|model| {
        (
            !model.installed,
            workspace_fit_rank(model.fit),
            model.catalog_id.is_none(),
            model.name.clone(),
        )
    });
    models
}

fn workspace_fit_rank(fit: crate::fit::FitVerdict) -> u8 {
    match fit {
        crate::fit::FitVerdict::FitsResident => 0,
        crate::fit::FitVerdict::FitsWithOffload | crate::fit::FitVerdict::CpuOnlyOk => 1,
        crate::fit::FitVerdict::Unknown => 2,
        // Both refusals sort last. `InsufficientFreeMemory` is transient and could
        // become runnable by freeing memory, but ranking it above a proven fit
        // would put a currently-unloadable model ahead of one the user can start
        // right now.
        crate::fit::FitVerdict::InsufficientFreeMemory | crate::fit::FitVerdict::WontFit => 3,
    }
}

fn tool_capable_compatibility_rows() -> Vec<super::ModelCompatibilityTarget> {
    capabilities_response_with_plan(None)
        .model_compatibility
        .into_iter()
        .filter(|row| row.tool_capable && row.status.starts_with("supported"))
        .collect()
}

/// Cap on directories returned per browse call so a folder with an enormous
/// number of children cannot produce an unbounded response body.
const MAX_BROWSE_ENTRIES: usize = 4096;

#[derive(Debug, Deserialize)]
pub(super) struct WorkspaceBrowseQuery {
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Serialize)]
struct WorkspaceBrowseEntry {
    name: String,
    path: String,
}

#[derive(Debug, Serialize)]
struct WorkspaceBrowseResponse {
    /// Canonical folder being listed, or `None` for the Windows drive-roots
    /// view. The UI shows an "up" affordance only when this is `Some`.
    path: Option<String>,
    /// Parent folder within the filesystem, or `None` at a drive/root.
    parent: Option<String>,
    /// True on platforms that have a roots listing (Windows drives), so the UI
    /// can offer "up to drives" when `parent` is `None`.
    has_roots: bool,
    /// Native path separator, so the UI can render paths without guessing.
    separator: String,
    /// Immediate child directories, name-sorted, directories only.
    entries: Vec<WorkspaceBrowseEntry>,
    /// True when `entries` was capped at `MAX_BROWSE_ENTRIES`.
    truncated: bool,
}

/// Read-only directory browsing that backs the Workspace folder picker. This is
/// a setup helper, not an agent tool: it never reads file contents, lists only
/// directories, and does not widen the agent sandbox (the chosen root is still
/// canonicalized and confined when a session starts).
pub(super) async fn browse(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<WorkspaceBrowseQuery>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let result = match run_workspace_blocking(move || browse_directory(query.path.as_deref())).await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    match result {
        Ok(body) => Json(body).into_response(),
        Err((code, id, message)) => api_error(code, id, message, Some("path")),
    }
}

fn browse_directory(
    requested: Option<&str>,
) -> Result<WorkspaceBrowseResponse, (StatusCode, &'static str, String)> {
    let has_roots = cfg!(windows);
    let separator = std::path::MAIN_SEPARATOR.to_string();
    let requested = requested.map(str::trim).filter(|value| !value.is_empty());

    // Windows with no folder selected shows the available drive letters.
    if requested.is_none() && has_roots {
        return Ok(WorkspaceBrowseResponse {
            path: None,
            parent: None,
            has_roots,
            separator,
            entries: windows_drive_roots(),
            truncated: false,
        });
    }

    let target = match requested {
        Some(value) => PathBuf::from(value),
        // Unix with no folder selected starts at the filesystem root.
        None => PathBuf::from(std::path::MAIN_SEPARATOR.to_string()),
    };

    let canonical = std::fs::canonicalize(&target).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            "workspace_browse_path_invalid",
            "that folder is not accessible".to_string(),
        )
    })?;
    if !canonical.is_dir() {
        return Err((
            StatusCode::BAD_REQUEST,
            "workspace_browse_path_invalid",
            "that path is not a folder".to_string(),
        ));
    }

    let (entries, truncated) = list_child_directories(&canonical);
    Ok(WorkspaceBrowseResponse {
        path: Some(simplify_path(&canonical)),
        parent: canonical.parent().map(simplify_path),
        has_roots,
        separator,
        entries,
        truncated,
    })
}

fn windows_drive_roots() -> Vec<WorkspaceBrowseEntry> {
    let mut roots = Vec::new();
    for letter in b'A'..=b'Z' {
        let root = format!("{}:\\", letter as char);
        if std::path::Path::new(&root).is_dir() {
            roots.push(WorkspaceBrowseEntry {
                name: root.clone(),
                path: root,
            });
        }
    }
    roots
}

fn list_child_directories(dir: &std::path::Path) -> (Vec<WorkspaceBrowseEntry>, bool) {
    let mut entries = std::collections::BinaryHeap::new();
    let mut truncated = false;
    if let Ok(read_dir) = std::fs::read_dir(dir) {
        for entry in read_dir.flatten() {
            // Directories only, and don't follow symlinks (`file_type` reports
            // the link itself). Unreadable entries are skipped, not fatal.
            if !entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            // Hide dot-directories to keep the picker readable.
            if name.starts_with('.') {
                continue;
            }
            entries.push((name.to_lowercase(), name, entry.path()));
            if entries.len() > MAX_BROWSE_ENTRIES {
                entries.pop();
                truncated = true;
            }
        }
    }
    let mut entries = entries
        .into_iter()
        .map(|(_, name, path)| WorkspaceBrowseEntry {
            path: simplify_path(&path),
            name,
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.name.to_lowercase());
    (entries, truncated)
}

/// `std::fs::canonicalize` yields Windows extended-length (`\\?\C:\...`) paths.
/// Strip that verbatim prefix so the picker shows and round-trips ordinary
/// `C:\...` paths; selecting one canonicalizes again on the server anyway.
fn simplify_path(path: &std::path::Path) -> String {
    let text = path.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        if let Some(stripped) = text.strip_prefix(r"\\?\") {
            if let Some(unc) = stripped.strip_prefix("UNC\\") {
                return format!(r"\\{unc}");
            }
            return stripped.to_string();
        }
    }
    text
}

pub(super) async fn create_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateWorkspaceSessionRequest>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }

    let goal = request.goal.trim().to_string();
    if goal.is_empty() || goal.len() > MAX_GOAL_BYTES {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_goal",
            format!("goal must contain 1 to {MAX_GOAL_BYTES} UTF-8 bytes"),
            Some("goal"),
        );
    }
    let max_steps = request.max_steps.unwrap_or(DEFAULT_MAX_STEPS);
    if !(1..=MAX_STEPS).contains(&max_steps) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_limits",
            format!("max_steps must be between 1 and {MAX_STEPS}"),
            Some("max_steps"),
        );
    }
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    if !(1..=MAX_TOKENS).contains(&max_tokens) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_limits",
            format!("max_tokens must be between 1 and {MAX_TOKENS}"),
            Some("max_tokens"),
        );
    }
    let temperature = request.temperature.unwrap_or(0.0);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_limits",
            "temperature must be finite and between 0 and 2".to_string(),
            Some("temperature"),
        );
    }
    if request.allow_writes.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "workspace_read_only",
            "Workspace is read-only; write and edit tools are not available".to_string(),
            Some("allow_writes"),
        );
    }
    let allow_writes = false;

    let requested_workspace = request.workspace;
    let workspace =
        match run_workspace_blocking(move || match std::fs::canonicalize(requested_workspace) {
            Ok(path) if path.is_dir() => Ok(path),
            Ok(_) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "workspace root is not a directory",
            )),
            Err(error) => Err(error),
        })
        .await
        {
            Ok(Ok(path)) => path,
            Ok(Err(_)) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "workspace_root_not_accessible",
                    "workspace must name an accessible local directory".to_string(),
                    Some("workspace"),
                )
            }
            Err(response) => return response,
        };

    // Publish the session while holding the same transition mutex as model
    // load/unload. This closes the check-then-act window where a model could be
    // replaced after identity validation but before the session became active.
    let _model_transition = state.model_transition.lock().await;
    let (model, family) = match active_tool_capable_model(&state).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let Some(model_context_tokens) = model
        .llama_config
        .as_ref()
        .map(|config| config.context_length)
    else {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_model_context_unavailable",
            "the active model does not expose an effective runtime context length".to_string(),
            None,
        );
    };
    let runtime_budget = workspace_runtime_budget(
        model_context_tokens,
        state.server_limits.max_prompt_tokens,
        state.server_limits.max_generation_tokens,
        max_tokens,
    );
    let context_budget_tokens = runtime_budget.context_tokens;
    // Report and retain the effective reply allowance, so the initial turn and
    // every follow-up use a value the active server can actually honor.
    let max_tokens = runtime_budget.reply_tokens;
    if context_budget_tokens <= max_tokens {
        return api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "workspace_context_too_small",
            format!(
                "the active model and server limits leave no prompt room after reserving {max_tokens} reply tokens"
            ),
            Some("max_tokens"),
        );
    }
    let model_step_timeout = workspace_model_step_timeout();
    let semantic_retriever = workspace_semantic_retriever(&state, &workspace).await;
    let embedding_model_id = semantic_retriever
        .as_ref()
        .map(|retriever| retriever.model_id().to_string());

    let mut active = state.workspace_sessions.active.lock().await;
    if let Some(existing_state) = WorkspaceSessionManager::active_state(&active) {
        let blocks_replacement = existing_state.blocks_model_transition();
        if blocks_replacement {
            return api_error(
                StatusCode::CONFLICT,
                "workspace_session_already_active",
                "finish or cancel the current Workspace session before starting another"
                    .to_string(),
                None,
            );
        }
        *active = None;
    }

    let canonical_root = simplify_path(&workspace);
    let resume_id = request
        .thread_id
        .as_deref()
        .map(str::trim)
        .filter(|thread_id| !thread_id.is_empty())
        .map(str::to_string);
    let memory_root = canonical_root.clone();
    let memory_model_id = model.id.clone();
    let memory_model_sha256 = model.lane.gguf_sha256.to_string();
    let memory_goal = goal.clone();
    let prepared = match run_workspace_blocking(move || -> anyhow::Result<_> {
        let memory = WorkspaceMemoryStore::open(default_store_path())?;
        let prepared = if let Some(thread_id) = resume_id {
            let Some(stored) = memory.thread(&thread_id)? else {
                return Ok(Err("not_found"));
            };
            if stored.canonical_root != memory_root
                || stored.model_id != memory_model_id
                || stored.model_sha256 != memory_model_sha256
            {
                return Ok(Err("identity_mismatch"));
            }
            let context = memory.context_for(&thread_id, &memory_goal, 2 * 1024)?;
            (memory, thread_id, context, stored.turn_count)
        } else {
            let id = format!("workspace-{}", uuid::Uuid::new_v4());
            memory.create_thread_for_model(
                &id,
                &memory_root,
                &memory_model_id,
                &memory_model_sha256,
                &memory_goal,
            )?;
            (memory, id, Default::default(), 0)
        };
        Ok(Ok(prepared))
    })
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    let (memory, id, context, turn_index) = match prepared {
        Ok(Ok(prepared)) => prepared,
        Ok(Err("not_found")) => return workspace_not_found(),
        Ok(Err("identity_mismatch")) => {
            return api_error(
                StatusCode::CONFLICT,
                "workspace_thread_identity_mismatch",
                "the saved thread does not belong to this canonical folder and active model"
                    .to_string(),
                None,
            )
        }
        Ok(Err(_)) => unreachable!("fixed Workspace preparation error"),
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "workspace_memory_unavailable",
                format!("Workspace memory could not prepare this thread: {error}"),
                None,
            )
        }
    };
    let (worker, client) = bridge(EVENT_BACKLOG);
    let (events, control) = client.into_parts();
    let client_message_id = format!("initial-{}", uuid::Uuid::new_v4());
    let run_config = WorkspaceRunConfig {
        addr: state.serve_addr,
        workspace: workspace.clone(),
        goal: goal.to_string(),
        client_message_id: client_message_id.clone(),
        turn_index,
        memory: context,
        model_id: model.id.clone(),
        model_sha256: model.lane.gguf_sha256.clone(),
        family,
        max_steps,
        max_tokens,
        temperature,
        context_budget_tokens,
        model_step_timeout,
        semantic_retriever: semantic_retriever.clone(),
    };
    let session = Arc::new(ActiveWorkspaceSession {
        id: id.clone(),
        workspace: workspace.clone(),
        model_id: model.id.clone(),
        model_sha256: model.lane.gguf_sha256.clone(),
        max_steps,
        max_tokens,
        temperature,
        context_budget_tokens,
        model_step_timeout,
        allow_writes,
        semantic_retriever,
        memory,
        state: StdMutex::new(WorkspaceSessionState::WaitingForEvents),
        events: StdMutex::new(Some(events)),
        worker: StdMutex::new(Some(worker)),
        run_config: StdMutex::new(Some(run_config)),
        control: StdMutex::new(Some(control)),
        current_turn: StdMutex::new(Some((client_message_id.clone(), turn_index))),
    });
    arm_event_claim_deadline(&session, client_message_id);
    *active = Some(session);

    (
        StatusCode::CREATED,
        Json(WorkspaceSessionResponse {
            id,
            workspace: simplify_path(&workspace),
            model_id: model.id,
            state: WorkspaceSessionState::WaitingForEvents.as_str(),
            max_steps,
            max_tokens,
            allow_writes,
            semantic_retrieval: embedding_model_id.is_some(),
            embedding_model_id,
        }),
    )
        .into_response()
}

pub(super) async fn session_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let session = match find_session(&state, &id).await {
        Ok(session) => session,
        Err(response) => return response,
    };
    let Ok(mut status) = session.state.lock() else {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_state_unavailable",
            "Workspace session state is unavailable".to_string(),
            None,
        );
    };
    if *status != WorkspaceSessionState::WaitingForEvents {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_event_stream_unavailable",
            "this Workspace turn is no longer waiting for an event consumer".to_string(),
            None,
        );
    }
    let events = session
        .events
        .lock()
        .ok()
        .and_then(|mut events| events.take());
    let worker = session
        .worker
        .lock()
        .ok()
        .and_then(|mut worker| worker.take());
    let run_config = session
        .run_config
        .lock()
        .ok()
        .and_then(|mut config| config.take());
    let control = session
        .control
        .lock()
        .ok()
        .and_then(|control| control.clone());
    let (Some(events), Some(worker), Some(run_config), Some(control)) =
        (events, worker, run_config, control)
    else {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_event_stream_already_claimed",
            "this Workspace session already has an event consumer".to_string(),
            None,
        );
    };
    let persisted_turn = run_config.clone();
    *status = status.after_events_claimed();
    drop(status);

    let worker_session = Arc::clone(&session);
    let worker_turn_id = run_config.client_message_id.clone();
    let delivery_failed = Arc::clone(&worker.delivery_failed);
    std::thread::Builder::new()
        .name("camelid-workspace-agent".to_string())
        .spawn(move || {
            let result = run_live(run_config, worker);
            if result.is_err() || delivery_failed.load(std::sync::atomic::Ordering::Acquire) {
                let completion = if matches!(result, Ok(LoopEnd::DriverError) | Err(_)) {
                    TurnCompletion::Failed
                } else {
                    TurnCompletion::Idle
                };
                worker_session.finish_turn_if_current(&worker_turn_id, completion);
            }
        })
        .expect("spawn Workspace agent thread");

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(EVENT_STREAM_BUFFER);
    let forward_control = control.clone();
    let persist_session = Arc::clone(&session);
    std::thread::Builder::new()
        .name("camelid-workspace-events".to_string())
        .spawn(move || {
            let mut pending_call = None;
            let mut evidence = Vec::new();
            let mut last_context_usage = None;
            let mut assistant_answer = None;
            let mut persistence_attempted = false;
            while let Ok(event) = events.recv() {
                if let WorkspaceEvent::MemoryUpdated {
                    prompt_tokens,
                    generation_tokens,
                    budget_total,
                    ..
                } = &event
                {
                    last_context_usage =
                        Some((*prompt_tokens, *generation_tokens, *budget_total));
                }
                if let WorkspaceEvent::ToolCall { detail } = &event {
                    pending_call = Some(detail.clone());
                }
                if let WorkspaceEvent::ToolResult { tool, content, .. } = &event {
                    evidence.push(EvidenceInput {
                        tool: tool.clone(),
                        detail: pending_call.take().unwrap_or_default(),
                        observation: content.clone(),
                    });
                }
                let mut automatic_compaction = None;
                if let WorkspaceEvent::ModelAnswer { content } = &event {
                    assistant_answer = Some(content.clone());
                }
                if let WorkspaceEvent::Finished { outcome } = &event {
                    persistence_attempted = true;
                    if let Err(error) = persist_session.memory.append_terminal_turn(
                        &persist_session.id,
                        &persisted_turn.client_message_id,
                        &persisted_turn.goal,
                        assistant_answer.as_deref().unwrap_or_default(),
                        outcome,
                        &evidence,
                    ) {
                        let _ = event_tx.blocking_send(WorkspaceEvent::Error {
                            message: format!("Workspace memory could not save this turn: {error}"),
                        });
                        persist_session.finish_turn_if_current(
                            &persisted_turn.client_message_id,
                            TurnCompletion::Failed,
                        );
                        forward_control.cancel();
                        break;
                    }
                    if *outcome == "answered" {
                        if let Some((prompt_tokens, generation_tokens, budget_total)) =
                            last_context_usage
                        {
                            let thread = persist_session.memory.thread(&persist_session.id);
                            if let Ok(Some(thread)) = thread {
                                if should_auto_compact(
                                    thread.turn_count,
                                    prompt_tokens,
                                    generation_tokens,
                                    budget_total,
                                ) {
                                    match persist_session.memory.compact_thread(&persist_session.id)
                                    {
                                        Ok(result) if result.archived_turns > 0 => {
                                            automatic_compaction =
                                                Some(WorkspaceEvent::MemoryCompacted {
                                                    compacted_through_turn: result
                                                        .compacted_through_turn,
                                                    archived_turns: result.archived_turns,
                                                    compaction_count: result.compaction_count,
                                                    trigger_tokens: prompt_tokens
                                                        .saturating_add(generation_tokens),
                                                    budget_total,
                                                });
                                        }
                                        Ok(_) => {}
                                        Err(error) => {
                                            automatic_compaction = Some(WorkspaceEvent::Notice {
                                                content: format!(
                                                    "Automatic conversation compaction was skipped: {error}"
                                                ),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let completion = if *outcome == "driver_error" {
                        TurnCompletion::Failed
                    } else {
                        TurnCompletion::Idle
                    };
                    persist_session
                        .finish_turn_if_current(&persisted_turn.client_message_id, completion);
                }
                if forward_workspace_event(&event_tx, event, automatic_compaction).is_err() {
                    forward_control.cancel();
                    break;
                }
            }
            if !persistence_attempted {
                if let Err(error) =
                    persist_session.persist_aborted_turn_and_finish(&persisted_turn, &evidence)
                {
                    eprintln!("Workspace memory could not save an interrupted turn: {error}");
                }
            }
        })
        .expect("spawn Workspace event forwarder");

    let session_id = session.id.clone();
    let disconnect_guard = CancelStreamOnDrop(control);
    let stream = async_stream::stream! {
        let _disconnect_guard = disconnect_guard;
        let mut sequence = 0_u64;
        while let Some(event) = event_rx.recv().await {
            sequence += 1;
            let terminal = matches!(event, WorkspaceEvent::Finished { .. } | WorkspaceEvent::Error { .. });
            let envelope = WorkspaceEventEnvelope {
                sequence,
                session_id: session_id.clone(),
                event,
            };
            match serde_json::to_string(&envelope) {
                Ok(json) => yield Ok::<Event, std::convert::Infallible>(
                    Event::default().event("workspace").id(sequence.to_string()).data(json)
                ),
                Err(_) => continue,
            }
            if terminal {
                break;
            }
        }
    };
    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(10))
                .text("ping"),
        )
        .into_response()
}

pub(super) async fn decide(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<WorkspaceDecisionRequest>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let session = match find_session(&state, &id).await {
        Ok(session) => session,
        Err(response) => return response,
    };
    let control = session
        .control
        .lock()
        .ok()
        .and_then(|control| control.clone());
    let Some(control) = control else {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_approval_not_pending",
            "this Workspace thread has no active turn".to_string(),
            Some("approval_id"),
        );
    };
    match control.try_decide(request.approval_id, request.decision) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(message) => api_error(
            StatusCode::CONFLICT,
            "workspace_approval_not_pending",
            message.to_string(),
            Some("approval_id"),
        ),
    }
}

pub(super) async fn session_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let session = match find_session(&state, &id).await {
        Ok(session) => session,
        Err(response) => return response,
    };
    let status = session
        .state
        .lock()
        .map(|state| state.as_str())
        .unwrap_or("error");
    Json(WorkspaceSessionStatusResponse {
        id: session.id.clone(),
        workspace: simplify_path(&session.workspace),
        model_id: session.model_id.clone(),
        state: status,
        context_budget_tokens: session.context_budget_tokens,
        resident_cuda: crate::inference::resident_cuda_status(super::model_resident_cache_key(
            &session.model_id,
        )),
        allow_writes: session.allow_writes,
        semantic_retrieval: session.semantic_retriever.is_some(),
        embedding_model_id: session
            .semantic_retriever
            .as_ref()
            .map(|retriever| retriever.model_id().to_string()),
    })
    .into_response()
}

pub(super) async fn send_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<WorkspaceMessageRequest>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let text = request.text.trim().to_string();
    let client_message_id = request.client_message_id.trim().to_string();
    if text.is_empty() || text.len() > MAX_GOAL_BYTES {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_message",
            format!("text must contain 1 to {MAX_GOAL_BYTES} UTF-8 bytes"),
            Some("text"),
        );
    }
    if client_message_id.is_empty() || client_message_id.len() > 128 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_workspace_message_id",
            "client_message_id must contain 1 to 128 characters".to_string(),
            Some("client_message_id"),
        );
    }
    let session = match find_session(&state, &id).await {
        Ok(session) => session,
        Err(response) => return response,
    };
    if let Some(turn_index) = session.pending_message(&client_message_id) {
        let state = session
            .state
            .lock()
            .map(|state| state.as_str())
            .unwrap_or("error");
        return (
            StatusCode::OK,
            Json(WorkspaceMessageResponse {
                session_id: session.id.clone(),
                turn_index,
                state,
                duplicate: true,
            }),
        )
            .into_response();
    }
    let duplicate_memory = session.memory.clone();
    let duplicate_session_id = session.id.clone();
    let duplicate_message_id = client_message_id.clone();
    let duplicate = match run_workspace_blocking(move || {
        duplicate_memory.turn_by_client_message(&duplicate_session_id, &duplicate_message_id)
    })
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    match duplicate {
        Ok(Some(turn)) => {
            return (
                StatusCode::OK,
                Json(WorkspaceMessageResponse {
                    session_id: session.id.clone(),
                    turn_index: turn.turn_index,
                    state: WorkspaceSessionState::Idle.as_str(),
                    duplicate: true,
                }),
            )
                .into_response()
        }
        Ok(None) => {}
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "workspace_memory_unavailable",
                format!("Workspace memory could not check this message: {error}"),
                None,
            )
        }
    }
    // Pair identity validation + turn installation atomically with model
    // transitions. A transition that wins the lock first is observed below; a
    // follow-up that wins first becomes blocking before the lock is released.
    let _model_transition = state.model_transition.lock().await;
    let (model, family) = match active_tool_capable_model(&state).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    if !workspace_model_identity_matches(
        &session.model_id,
        &session.model_sha256,
        &model.id,
        &model.lane.gguf_sha256,
    ) {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_model_changed",
            "resume this thread with the same model bytes that created it".to_string(),
            None,
        );
    }
    let context_memory = session.memory.clone();
    let context_session_id = session.id.clone();
    let context_query = text.clone();
    let context = match run_workspace_blocking(move || -> anyhow::Result<_> {
        let memory = context_memory.context_for(&context_session_id, &context_query, 2 * 1024)?;
        let turn_index = context_memory
            .thread(&context_session_id)?
            .map(|thread| thread.turn_count);
        Ok(turn_index.map(|turn_index| (memory, turn_index)))
    })
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    let (memory, turn_index) = match context {
        Ok(Some(context)) => context,
        Ok(None) => return workspace_not_found(),
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "workspace_memory_unavailable",
                format!("Workspace memory could not retrieve prior turns: {error}"),
                None,
            )
        }
    };
    let (worker, client) = bridge(EVENT_BACKLOG);
    let (events, control) = client.into_parts();
    let run_config = WorkspaceRunConfig {
        addr: state.serve_addr,
        workspace: session.workspace.clone(),
        goal: text,
        client_message_id: client_message_id.clone(),
        turn_index,
        memory,
        model_id: session.model_id.clone(),
        model_sha256: session.model_sha256.clone(),
        family,
        max_steps: session.max_steps,
        max_tokens: session.max_tokens,
        temperature: session.temperature,
        context_budget_tokens: session.context_budget_tokens,
        model_step_timeout: session.model_step_timeout,
        semantic_retriever: session.semantic_retriever.clone(),
    };
    match session.install_turn(events, worker, run_config, control) {
        Ok(InstallTurn::Installed) => {}
        Ok(InstallTurn::Duplicate(existing_index)) => {
            return (
                StatusCode::OK,
                Json(WorkspaceMessageResponse {
                    session_id: session.id.clone(),
                    turn_index: existing_index,
                    state: session
                        .state
                        .lock()
                        .map(|state| state.as_str())
                        .unwrap_or("error"),
                    duplicate: true,
                }),
            )
                .into_response()
        }
        Err(message) => {
            return api_error(
                StatusCode::CONFLICT,
                "workspace_turn_already_active",
                message.to_string(),
                None,
            )
        }
    }
    arm_event_claim_deadline(&session, client_message_id);
    (
        StatusCode::ACCEPTED,
        Json(WorkspaceMessageResponse {
            session_id: session.id.clone(),
            turn_index,
            state: WorkspaceSessionState::WaitingForEvents.as_str(),
            duplicate: false,
        }),
    )
        .into_response()
}

pub(super) async fn cancel_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let active = state.workspace_sessions.active.lock().await;
    let Some(session) = active.as_ref().cloned() else {
        return workspace_not_found();
    };
    if session.id != id {
        return workspace_not_found();
    }
    drop(active);
    if let Some(control) = session
        .control
        .lock()
        .ok()
        .and_then(|control| control.clone())
    {
        control.cancel();
    }
    let was_waiting = session
        .state
        .lock()
        .map(|status| *status == WorkspaceSessionState::WaitingForEvents)
        .unwrap_or(false);
    let unclaimed_turn = if was_waiting {
        session
            .run_config
            .lock()
            .ok()
            .and_then(|config| config.clone())
    } else {
        None
    };
    if let Ok(mut status) = session.state.lock() {
        *status = status.after_cancel_request();
    }
    if let Some(turn) = unclaimed_turn {
        let cancel_memory = session.memory.clone();
        let cancel_session_id = session.id.clone();
        let cancel_turn = turn.clone();
        let persisted = match run_workspace_blocking(move || {
            cancel_memory.append_terminal_turn(
                &cancel_session_id,
                &cancel_turn.client_message_id,
                &cancel_turn.goal,
                "",
                "aborted",
                &[],
            )
        })
        .await
        {
            Ok(result) => result,
            Err(response) => return response,
        };
        if let Err(error) = persisted {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "workspace_memory_unavailable",
                format!("Workspace memory could not save the cancelled turn: {error}"),
                None,
            );
        }
        session.finish_turn_if_current(&turn.client_message_id, TurnCompletion::Idle);
    }
    StatusCode::NO_CONTENT.into_response()
}

async fn find_session(state: &AppState, id: &str) -> Result<Arc<ActiveWorkspaceSession>, Response> {
    let active = state.workspace_sessions.active.lock().await;
    active
        .as_ref()
        .filter(|session| session.id == id)
        .cloned()
        .ok_or_else(workspace_not_found)
}

fn workspace_not_found() -> Response {
    api_error(
        StatusCode::NOT_FOUND,
        "workspace_session_not_found",
        "Workspace session was not found".to_string(),
        None,
    )
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    if state.serve_addr.ip().is_loopback()
        && workspace_request_allowed(headers, state.workspace_cli_token.as_deref())
    {
        return None;
    }
    Some(api_error(
        StatusCode::FORBIDDEN,
        "local_management_forbidden",
        "Workspace requires Camelid's same-origin web UI or same-user CLI credential".to_string(),
        None,
    ))
}

fn workspace_request_allowed(headers: &HeaderMap, cli_token: Option<&str>) -> bool {
    let loopback_host = headers
        .get("host")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<axum::http::uri::Authority>().ok())
        .is_some_and(|authority| loopback_host(authority.host()));
    if !loopback_host {
        return false;
    }
    if local_management_request_allowed(headers) {
        return true;
    }
    let provided = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    matches!((cli_token, provided), (Some(expected), Some(provided)) if crate::workspace_auth::token_matches(expected, provided))
}

fn local_management_request_allowed(headers: &HeaderMap) -> bool {
    let authority = headers
        .get("host")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<axum::http::uri::Authority>().ok());
    let Some(authority) = authority else {
        return false;
    };
    if !loopback_host(authority.host()) {
        return false;
    }
    let origin = headers
        .get("origin")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<axum::http::Uri>().ok());
    if let Some(origin) = &origin {
        if !matches!(origin.scheme_str(), Some("http") | Some("https"))
            || origin.authority() != Some(&authority)
        {
            return false;
        }
    }
    headers
        .get("sec-fetch-site")
        .and_then(|value| value.to_str().ok())
        == Some("same-origin")
        || origin.is_some()
}

fn loopback_host(host: &str) -> bool {
    let host = host.trim_matches(['[', ']']);
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

async fn workspace_semantic_retriever(
    state: &AppState,
    workspace: &std::path::Path,
) -> Option<Arc<crate::chat::semantic_search::WorkspaceSemanticRetriever>> {
    let mut candidates = state
        .loaded_models
        .read()
        .await
        .values()
        .filter(|model| super::is_embedding_model(&model.gguf))
        .map(|model| model.id.clone())
        .collect::<Vec<_>>();
    candidates.sort();
    if candidates.is_empty() {
        return None;
    }
    let cached = state.embedding_runtimes.read().await;
    candidates.sort_by_key(|id| !cached.contains_key(id));
    drop(cached);
    let model_id = candidates.into_iter().next()?;
    match super::resolve_embedding_runtime(state, Some(&model_id)).await {
        Ok((model_id, runtime)) => Some(Arc::new(
            crate::chat::semantic_search::WorkspaceSemanticRetriever::new(
                workspace.to_path_buf(),
                model_id,
                runtime,
            ),
        )),
        Err(_) => None,
    }
}

async fn active_tool_capable_model(state: &AppState) -> Result<(LoadedModel, String), Response> {
    let active_id = state.active_model_id.read().await.clone().ok_or_else(|| {
        api_error(
            StatusCode::CONFLICT,
            "model_not_loaded",
            "load a tool-capable model before starting Workspace".to_string(),
            None,
        )
    })?;
    let model = state
        .loaded_models
        .read()
        .await
        .get(&active_id)
        .cloned()
        .ok_or_else(|| {
            api_error(
                StatusCode::CONFLICT,
                "model_not_loaded",
                "the active model is no longer loaded".to_string(),
                None,
            )
        })?;
    let filename = model
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let row = tool_capable_row_for_loaded_artifact(filename, &model.lane.gguf_sha256);
    match row {
        Some((_, family)) => Ok((model, family.to_string())),
        None => Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "model_not_tool_capable",
            "the active model is not an exact tool-capable artifact with a recorded certified digest"
                .to_string(),
            None,
        )),
    }
}

/// Resolve only rows with both earned tool capability and a recorded certified
/// digest. The UI must not promise an unpinned row that the loaded-artifact
/// authorization gate will necessarily refuse.
fn tool_capable_row_for_filename(filename: &str) -> Option<(&'static str, &'static str)> {
    supported_artifact_expected_sha256(filename)?;
    earned_tool_capable_row_for_filename(filename)
}

fn earned_tool_capable_row_for_filename(filename: &str) -> Option<(&'static str, &'static str)> {
    let row_id = curated_catalog()
        .iter()
        .find(|item| item.filename == filename)
        .map(|item| item.catalog_id)
        .or_else(|| {
            NON_CATALOG_SUPPORTED_ARTIFACTS
                .iter()
                .find(|(artifact, _, _)| *artifact == filename)
                .map(|(_, row_id, _)| *row_id)
        });
    row_id.and_then(|row_id| {
        tool_capable_compatibility_rows()
            .into_iter()
            .find(|row| row.id == row_id)
            .map(|row| (row.id, row.family))
    })
}

/// Whether a ledger row has at least one exact filename whose certified digest
/// is recorded. Used by CLI refusal copy so it lists only artifacts an operator
/// can actually authorize, not name-only rows that fail closed at runtime.
pub(crate) fn tool_capable_row_has_certified_artifact(row_id: &str) -> bool {
    curated_catalog().iter().any(|item| {
        item.catalog_id == row_id && tool_capable_row_for_filename(item.filename).is_some()
    }) || NON_CATALOG_SUPPORTED_ARTIFACTS
        .iter()
        .any(|(filename, candidate, _)| {
            *candidate == row_id && tool_capable_row_for_filename(filename).is_some()
        })
}

/// Tool capability for a LOADED artifact. `tool_capable` is earned per exact row
/// by a committed agent-eval receipt against specific bytes (the Ornith Q4_K_M
/// row is the live example), so a hash-pinned artifact must present its
/// certified digest before Workspace will drive it — otherwise a same-named
/// replacement inherits an agent battery it never passed.
pub(crate) fn tool_capable_row_for_loaded_artifact(
    filename: &str,
    gguf_sha256: &str,
) -> Option<(&'static str, &'static str)> {
    let expected = supported_artifact_expected_sha256(filename)?;
    if !gguf_sha256.eq_ignore_ascii_case(expected) {
        return None;
    }
    tool_capable_row_for_filename(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browse_lists_only_child_directories_sorted_and_excludes_files() {
        let root = tempfile::tempdir().expect("browse root");
        std::fs::create_dir(root.path().join("zeta")).unwrap();
        std::fs::create_dir(root.path().join("Alpha")).unwrap();
        std::fs::write(root.path().join("note.txt"), b"x").unwrap();

        let response = browse_directory(Some(root.path().to_str().unwrap())).expect("browse ok");
        let names: Vec<_> = response
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(names, vec!["Alpha", "zeta"]);
        assert!(response.path.is_some());
        assert!(!response.truncated);
        for entry in &response.entries {
            assert!(std::path::Path::new(&entry.path).is_dir());
        }
    }

    #[test]
    fn browse_reports_parent_of_a_subdirectory() {
        let root = tempfile::tempdir().expect("browse root");
        let child = root.path().join("child");
        std::fs::create_dir(&child).unwrap();

        let response = browse_directory(Some(child.to_str().unwrap())).expect("browse ok");
        let parent = response.parent.expect("child has a parent");
        assert_eq!(
            std::fs::canonicalize(parent).unwrap(),
            std::fs::canonicalize(root.path()).unwrap()
        );
    }

    #[test]
    fn browse_rejects_missing_and_non_directory_paths() {
        let root = tempfile::tempdir().expect("browse root");
        let missing = root.path().join("does-not-exist");
        assert!(browse_directory(Some(missing.to_str().unwrap())).is_err());

        let file = root.path().join("file.txt");
        std::fs::write(&file, b"x").unwrap();
        assert!(browse_directory(Some(file.to_str().unwrap())).is_err());
    }

    #[test]
    fn capped_folder_browse_retains_a_deterministic_sorted_subset() {
        let root = tempfile::tempdir().expect("browse root");
        for index in 0..=MAX_BROWSE_ENTRIES {
            std::fs::create_dir(root.path().join(format!("entry-{index:04}"))).unwrap();
        }

        let response = browse_directory(Some(root.path().to_str().unwrap())).expect("browse ok");
        assert!(response.truncated);
        assert_eq!(response.entries.len(), MAX_BROWSE_ENTRIES);
        assert_eq!(response.entries.first().unwrap().name, "entry-0000");
        assert_eq!(response.entries.last().unwrap().name, "entry-4095");
        assert!(!response
            .entries
            .iter()
            .any(|entry| entry.name == "entry-4096"));
    }

    #[test]
    fn compatible_models_expose_only_exact_earned_artifacts_and_installed_state() {
        let models_dir = tempfile::tempdir().expect("models dir");
        std::fs::write(models_dir.path().join("Qwen3-4B-Q4_K_M.gguf"), b"stub")
            .expect("write installed model");

        let options = workspace_model_options(models_dir.path());
        let installed = options
            .iter()
            .find(|model| model.filename == "Qwen3-4B-Q4_K_M.gguf")
            .expect("earned Qwen row");
        assert!(installed.installed);
        assert_eq!(installed.row_id, "qwen3_4b_q4_k_m");
        assert!(options
            .iter()
            .all(|model| tool_capable_row_for_filename(model.filename).is_some()));
        assert!(!options
            .iter()
            .any(|model| model.filename == "ornith-1.0-9b-Q3_K_M.gguf"));
        assert!(!options
            .iter()
            .any(|model| model.filename == "Qwen3-4B-Q8_0.gguf"));
    }

    #[test]
    fn tool_capability_requires_the_certified_bytes_not_just_the_name() {
        // `tool_capable` is earned per exact row by a committed agent-eval receipt
        // against specific bytes. The Ornith Q4_K_M name is shared by the certified
        // in-house requant and a different public HuggingFace imatrix quant, so the
        // digest — not the filename — has to authorize Workspace.
        const CERTIFIED: &str = "2711bf1ef034fa39eb899f793fe63bbb0aac21ebdacbcbe09406b5600ad5188f";
        const HF_IMATRIX_SAME_NAME: &str =
            "5720d1f671b4996481274fffe01868c3c36e87c135cc8538471cc7bd6087b106";
        let filename = "ornith-1.0-9b-Q4_K_M.gguf";
        assert!(
            tool_capable_row_for_filename(filename).is_some(),
            "precondition: this row is tool-capable by name"
        );
        assert!(tool_capable_row_for_loaded_artifact(filename, CERTIFIED).is_some());
        assert!(
            tool_capable_row_for_loaded_artifact(filename, HF_IMATRIX_SAME_NAME).is_none(),
            "uncertified bytes must not inherit the agent battery this row passed"
        );
        // A tool-capable compatibility row with no certified digest is useful
        // ledger metadata, but it cannot authorize executable agent tools.
        // Refuse it until evidence pins exact bytes.
        let unpinned = curated_catalog()
            .into_iter()
            .map(|item| item.filename)
            .find(|name| {
                supported_artifact_expected_sha256(name).is_none()
                    && earned_tool_capable_row_for_filename(name).is_some()
            })
            .expect("some tool-capable curated row still has no recorded digest");
        assert!(earned_tool_capable_row_for_filename(unpinned).is_some());
        assert!(tool_capable_row_for_filename(unpinned).is_none());
        assert!(
            tool_capable_row_for_loaded_artifact(unpinned, &"00".repeat(32)).is_none(),
            "{unpinned} must stay demoted until a certified digest is recorded"
        );
    }

    #[test]
    fn management_authorization_is_loopback_and_browser_scoped() {
        let mut headers = HeaderMap::new();
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        headers.insert("sec-fetch-site", "same-origin".parse().unwrap());
        assert!(local_management_request_allowed(&headers));

        headers.insert("origin", "https://attacker.example".parse().unwrap());
        assert!(!local_management_request_allowed(&headers));

        headers.insert("origin", "http://localhost:4173".parse().unwrap());
        assert!(!local_management_request_allowed(&headers));

        headers.insert("origin", "http://127.0.0.1:8181".parse().unwrap());
        assert!(local_management_request_allowed(&headers));

        headers.remove("origin");
        headers.remove("sec-fetch-site");
        assert!(!local_management_request_allowed(&headers));
    }

    #[test]
    fn cli_authorization_requires_the_exact_token_and_loopback_host() {
        let token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let mut headers = HeaderMap::new();
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        assert!(!workspace_request_allowed(&headers, Some(token)));

        headers.insert("authorization", format!("Bearer {token}").parse().unwrap());
        assert!(workspace_request_allowed(&headers, Some(token)));
        assert!(!workspace_request_allowed(&headers, Some(&"0".repeat(64))));
        assert!(!workspace_request_allowed(&headers, None));

        headers.insert("host", "example.com:8181".parse().unwrap());
        assert!(!workspace_request_allowed(&headers, Some(token)));
    }

    #[test]
    fn valid_cli_token_does_not_enable_a_non_loopback_listener() {
        let token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let state = AppState {
            serve_addr: "192.0.2.1:8181".parse().unwrap(),
            workspace_cli_token: Some(Arc::from(token)),
            ..AppState::default()
        };
        let mut headers = HeaderMap::new();
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        headers.insert("authorization", format!("Bearer {token}").parse().unwrap());
        assert!(authorize(&state, &headers).is_some());
    }

    #[test]
    fn session_state_blocks_model_transitions_only_while_active() {
        assert!(WorkspaceSessionState::WaitingForEvents.blocks_model_transition());
        assert!(WorkspaceSessionState::Running.blocks_model_transition());
        assert!(WorkspaceSessionState::Cancelling.blocks_model_transition());
        assert!(!WorkspaceSessionState::Idle.blocks_model_transition());
        assert!(!WorkspaceSessionState::Cancelled.blocks_model_transition());
        assert!(!WorkspaceSessionState::Failed.blocks_model_transition());
    }

    #[test]
    fn cancellation_stays_blocking_until_a_running_worker_exits() {
        let requested = WorkspaceSessionState::Running.after_cancel_request();
        assert_eq!(requested, WorkspaceSessionState::Cancelling);
        assert!(requested.blocks_model_transition());
        assert_eq!(
            WorkspaceSessionState::WaitingForEvents.after_cancel_request(),
            WorkspaceSessionState::Cancelled
        );
        assert_eq!(
            WorkspaceSessionState::Cancelled.after_events_claimed(),
            WorkspaceSessionState::Cancelled
        );
    }

    #[test]
    fn automatic_compaction_uses_exact_context_threshold_and_minimum_turns() {
        assert!(!should_auto_compact(3, 2_800, 512, 4_096));
        assert!(!should_auto_compact(4, 2_559, 512, 4_096));
        assert!(should_auto_compact(4, 2_560, 512, 4_096));
        assert!(!should_auto_compact(100, 4_096, 0, 0));
    }

    #[test]
    fn workspace_context_budget_reserves_reply_under_every_runtime_ceiling() {
        assert_eq!(
            workspace_runtime_budget(4_096, 16_384, 8_192, 512),
            WorkspaceRuntimeBudget {
                context_tokens: 4_096,
                reply_tokens: 512,
            }
        );
        assert_eq!(
            workspace_runtime_budget(32_768, 2_048, 96, 512),
            WorkspaceRuntimeBudget {
                context_tokens: 2_144,
                reply_tokens: 96,
            },
            "the generation ceiling must cap both generation and its reserve"
        );
        assert_eq!(
            workspace_runtime_budget(32_768, 131_072, 8_192, 512).context_tokens,
            crate::chat::agent::AGENT_VALIDATED_CTX
        );
        assert_eq!(
            workspace_runtime_budget(32_768, usize::MAX, u32::MAX, u32::MAX).context_tokens,
            crate::chat::agent::AGENT_VALIDATED_CTX
        );
    }

    #[test]
    fn workspace_step_timeout_is_configurable_but_bounded() {
        assert_eq!(
            parse_workspace_model_step_timeout(None),
            Duration::from_secs(DEFAULT_WORKSPACE_MODEL_STEP_TIMEOUT_SECS)
        );
        assert_eq!(
            parse_workspace_model_step_timeout(Some(" 1200 ")),
            Duration::from_secs(1_200)
        );
        for invalid in ["", "0", "not-a-number", "3601"] {
            assert_eq!(
                parse_workspace_model_step_timeout(Some(invalid)),
                Duration::from_secs(DEFAULT_WORKSPACE_MODEL_STEP_TIMEOUT_SECS)
            );
        }
    }

    #[test]
    fn workspace_follow_ups_require_the_same_model_id_and_bytes() {
        let sha = "ab".repeat(32);
        assert!(workspace_model_identity_matches(
            "model",
            &sha,
            "model",
            &sha.to_uppercase()
        ));
        assert!(!workspace_model_identity_matches(
            "model", &sha, "other", &sha
        ));
        assert!(!workspace_model_identity_matches(
            "model",
            &sha,
            "model",
            &"cd".repeat(32)
        ));
    }

    #[test]
    fn event_forwarding_backpressures_and_places_compaction_before_terminal() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        let join = std::thread::spawn(move || {
            forward_workspace_event(
                &sender,
                WorkspaceEvent::Finished {
                    outcome: "answered",
                },
                Some(WorkspaceEvent::MemoryCompacted {
                    compacted_through_turn: Some(3),
                    archived_turns: 3,
                    compaction_count: 1,
                    trigger_tokens: 6_144,
                    budget_total: 8_192,
                }),
            )
        });

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while receiver.is_empty() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(receiver.len(), 1, "prefix was not queued");
        assert!(
            !join.is_finished(),
            "a full event queue must apply backpressure"
        );
        assert!(matches!(
            receiver.blocking_recv(),
            Some(WorkspaceEvent::MemoryCompacted { .. })
        ));
        assert_eq!(join.join().unwrap(), Ok(()));
        assert!(matches!(
            receiver.blocking_recv(),
            Some(WorkspaceEvent::Finished {
                outcome: "answered"
            })
        ));
    }

    #[test]
    fn manager_reads_terminal_and_active_session_states_without_guessing() {
        let make_session = |state| {
            let (worker, client) = bridge(1);
            let (events, control) = client.into_parts();
            Arc::new(ActiveWorkspaceSession {
                id: "session-test".to_string(),
                workspace: PathBuf::from("."),
                model_id: "model-test".to_string(),
                model_sha256: "aa".repeat(32),
                max_steps: 1,
                max_tokens: 1,
                temperature: 0.0,
                context_budget_tokens: 8_192,
                model_step_timeout: Duration::from_secs(1),
                allow_writes: true,
                semantic_retriever: None,
                memory: WorkspaceMemoryStore::open(std::env::temp_dir().join(format!(
                    "camelid-workspace-state-test-{}.sqlite3",
                    uuid::Uuid::new_v4()
                )))
                .unwrap(),
                state: StdMutex::new(state),
                events: StdMutex::new(Some(events)),
                worker: StdMutex::new(Some(worker)),
                run_config: StdMutex::new(None),
                control: StdMutex::new(Some(control)),
                current_turn: StdMutex::new(None),
            })
        };

        let running = Some(make_session(WorkspaceSessionState::Running));
        assert_eq!(
            WorkspaceSessionManager::active_state(&running),
            Some(WorkspaceSessionState::Running)
        );
        assert!(WorkspaceSessionManager::active_state(&running)
            .unwrap()
            .blocks_model_transition());

        let finished = Some(make_session(WorkspaceSessionState::Idle));
        assert_eq!(
            WorkspaceSessionManager::active_state(&finished),
            Some(WorkspaceSessionState::Idle)
        );
        assert!(!WorkspaceSessionManager::active_state(&finished)
            .unwrap()
            .blocks_model_transition());
    }

    #[test]
    fn duplicate_pending_message_resolves_to_the_installed_turn() {
        let dir = tempfile::tempdir().unwrap();
        let memory = WorkspaceMemoryStore::open(dir.path().join("memory.sqlite3")).unwrap();
        let (initial_worker, initial_client) = bridge(1);
        let (_initial_events, initial_control) = initial_client.into_parts();
        let session = ActiveWorkspaceSession {
            id: "thread".into(),
            workspace: dir.path().to_path_buf(),
            model_id: "model".into(),
            model_sha256: "aa".repeat(32),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            context_budget_tokens: 8_192,
            model_step_timeout: Duration::from_secs(1),
            allow_writes: true,
            semantic_retriever: None,
            memory,
            state: StdMutex::new(WorkspaceSessionState::Idle),
            events: StdMutex::new(None),
            worker: StdMutex::new(Some(initial_worker)),
            run_config: StdMutex::new(None),
            control: StdMutex::new(Some(initial_control)),
            current_turn: StdMutex::new(None),
        };
        let config = WorkspaceRunConfig {
            addr: "127.0.0.1:8181".parse().unwrap(),
            workspace: dir.path().to_path_buf(),
            goal: "question".into(),
            client_message_id: "message-1".into(),
            turn_index: 3,
            memory: Default::default(),
            model_id: "model".into(),
            model_sha256: "aa".repeat(32),
            family: "qwen3".into(),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            context_budget_tokens: 8_192,
            model_step_timeout: Duration::from_secs(1),
            semantic_retriever: None,
        };
        let (worker, client) = bridge(1);
        let (events, control) = client.into_parts();
        assert!(matches!(
            session.install_turn(events, worker, config.clone(), control),
            Ok(InstallTurn::Installed)
        ));
        assert!(!session.finish_turn_if_current("stale-message", TurnCompletion::Idle));
        assert_eq!(
            session.state.lock().map(|state| *state).unwrap(),
            WorkspaceSessionState::WaitingForEvents
        );
        assert_eq!(session.pending_message("message-1"), Some(3));
        let (duplicate_worker, duplicate_client) = bridge(1);
        let (duplicate_events, duplicate_control) = duplicate_client.into_parts();
        assert!(matches!(
            session.install_turn(
                duplicate_events,
                duplicate_worker,
                config,
                duplicate_control
            ),
            Ok(InstallTurn::Duplicate(3))
        ));
        assert!(session.finish_turn_if_current("message-1", TurnCompletion::Idle));
        assert_eq!(
            session.state.lock().map(|state| *state).unwrap(),
            WorkspaceSessionState::Idle
        );
    }

    #[test]
    fn cancelled_turn_completion_remains_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let memory = WorkspaceMemoryStore::open(dir.path().join("memory.sqlite3")).unwrap();
        let session = ActiveWorkspaceSession {
            id: "thread".into(),
            workspace: dir.path().to_path_buf(),
            model_id: "model".into(),
            model_sha256: "aa".repeat(32),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            context_budget_tokens: 8_192,
            model_step_timeout: Duration::from_secs(1),
            allow_writes: false,
            semantic_retriever: None,
            memory,
            state: StdMutex::new(WorkspaceSessionState::Cancelled),
            events: StdMutex::new(None),
            worker: StdMutex::new(None),
            run_config: StdMutex::new(None),
            control: StdMutex::new(None),
            current_turn: StdMutex::new(Some(("message-1".into(), 0))),
        };
        assert!(session.finish_turn_if_current("message-1", TurnCompletion::Idle));
        assert_eq!(
            session.state.lock().map(|state| *state).unwrap(),
            WorkspaceSessionState::Cancelled
        );
    }

    #[test]
    fn event_forwarder_fallback_persists_and_finishes_cancelled_turn() {
        let dir = tempfile::tempdir().unwrap();
        let memory = WorkspaceMemoryStore::open(dir.path().join("memory.sqlite3")).unwrap();
        memory.create_thread("thread", "root", "model").unwrap();
        let session = ActiveWorkspaceSession {
            id: "thread".into(),
            workspace: dir.path().to_path_buf(),
            model_id: "model".into(),
            model_sha256: "aa".repeat(32),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            context_budget_tokens: 8_192,
            model_step_timeout: Duration::from_secs(1),
            allow_writes: false,
            semantic_retriever: None,
            memory,
            state: StdMutex::new(WorkspaceSessionState::Cancelling),
            events: StdMutex::new(None),
            worker: StdMutex::new(None),
            run_config: StdMutex::new(None),
            control: StdMutex::new(None),
            current_turn: StdMutex::new(Some(("message-1".into(), 0))),
        };
        let run_config = WorkspaceRunConfig {
            addr: "127.0.0.1:8181".parse().unwrap(),
            workspace: dir.path().to_path_buf(),
            goal: "question".into(),
            client_message_id: "message-1".into(),
            turn_index: 0,
            memory: Default::default(),
            model_id: "model".into(),
            model_sha256: "aa".repeat(32),
            family: "qwen3".into(),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            context_budget_tokens: 8_192,
            model_step_timeout: Duration::from_secs(1),
            semantic_retriever: None,
        };

        assert!(session
            .persist_aborted_turn_and_finish(&run_config, &[])
            .unwrap());
        assert_eq!(
            session.state.lock().map(|state| *state).unwrap(),
            WorkspaceSessionState::Cancelled
        );
        assert_eq!(session.pending_message("message-1"), None);
        let turn = session
            .memory
            .turn_by_client_message("thread", "message-1")
            .unwrap()
            .unwrap();
        assert_eq!(turn.terminal_outcome, "aborted");
    }

    #[test]
    fn terminal_turn_states_accept_a_follow_up() {
        for terminal_state in [
            WorkspaceSessionState::Cancelled,
            WorkspaceSessionState::Failed,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let memory = WorkspaceMemoryStore::open(dir.path().join("memory.sqlite3")).unwrap();
            let (stale_worker, stale_client) = bridge(1);
            let (stale_events, stale_control) = stale_client.into_parts();
            let session = ActiveWorkspaceSession {
                id: "thread".into(),
                workspace: dir.path().to_path_buf(),
                model_id: "model".into(),
                model_sha256: "aa".repeat(32),
                max_steps: 1,
                max_tokens: 1,
                temperature: 0.0,
                context_budget_tokens: 8_192,
                model_step_timeout: Duration::from_secs(1),
                allow_writes: false,
                semantic_retriever: None,
                memory,
                state: StdMutex::new(terminal_state),
                events: StdMutex::new(Some(stale_events)),
                worker: StdMutex::new(Some(stale_worker)),
                run_config: StdMutex::new(None),
                control: StdMutex::new(Some(stale_control)),
                current_turn: StdMutex::new(Some(("old-message".into(), 0))),
            };
            let config = WorkspaceRunConfig {
                addr: "127.0.0.1:8181".parse().unwrap(),
                workspace: dir.path().to_path_buf(),
                goal: "follow up".into(),
                client_message_id: "new-message".into(),
                turn_index: 1,
                memory: Default::default(),
                model_id: "model".into(),
                model_sha256: "aa".repeat(32),
                family: "qwen3".into(),
                max_steps: 1,
                max_tokens: 1,
                temperature: 0.0,
                context_budget_tokens: 8_192,
                model_step_timeout: Duration::from_secs(1),
                semantic_retriever: None,
            };
            let (worker, client) = bridge(1);
            let (events, control) = client.into_parts();

            assert!(matches!(
                session.install_turn(events, worker, config, control),
                Ok(InstallTurn::Installed)
            ));
            assert_eq!(session.pending_message("old-message"), None);
            assert_eq!(session.pending_message("new-message"), Some(1));
            assert_eq!(
                session.state.lock().map(|state| *state).unwrap(),
                WorkspaceSessionState::WaitingForEvents
            );
        }
    }

    #[test]
    fn unclaimed_turn_expiry_persists_and_unblocks_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let memory = WorkspaceMemoryStore::open(dir.path().join("memory.sqlite3")).unwrap();
        memory.create_thread("thread", "root", "model").unwrap();
        let (worker, client) = bridge(1);
        let (events, control) = client.into_parts();
        let session = ActiveWorkspaceSession {
            id: "thread".into(),
            workspace: dir.path().to_path_buf(),
            model_id: "model".into(),
            model_sha256: "aa".repeat(32),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            context_budget_tokens: 8_192,
            model_step_timeout: Duration::from_secs(1),
            allow_writes: false,
            semantic_retriever: None,
            memory,
            state: StdMutex::new(WorkspaceSessionState::WaitingForEvents),
            events: StdMutex::new(Some(events)),
            worker: StdMutex::new(Some(worker)),
            run_config: StdMutex::new(Some(WorkspaceRunConfig {
                addr: "127.0.0.1:8181".parse().unwrap(),
                workspace: dir.path().to_path_buf(),
                goal: "question".into(),
                client_message_id: "message-1".into(),
                turn_index: 0,
                memory: Default::default(),
                model_id: "model".into(),
                model_sha256: "aa".repeat(32),
                family: "qwen3".into(),
                max_steps: 1,
                max_tokens: 1,
                temperature: 0.0,
                context_budget_tokens: 8_192,
                model_step_timeout: Duration::from_secs(1),
                semantic_retriever: None,
            })),
            control: StdMutex::new(Some(control)),
            current_turn: StdMutex::new(Some(("message-1".into(), 0))),
        };

        assert!(session.expire_unclaimed_turn("message-1").unwrap());
        assert_eq!(
            session.state.lock().map(|state| *state).unwrap(),
            WorkspaceSessionState::Cancelled
        );
        assert!(!session
            .state
            .lock()
            .map(|state| state.blocks_model_transition())
            .unwrap());
        assert_eq!(session.pending_message("message-1"), None);
        let turn = session
            .memory
            .turn_by_client_message("thread", "message-1")
            .unwrap()
            .unwrap();
        assert_eq!(turn.user_text, "question");
        assert_eq!(turn.terminal_outcome, "aborted");
        assert!(!session.expire_unclaimed_turn("message-1").unwrap());
    }

    #[test]
    fn every_certified_tool_capable_artifact_resolves_by_exact_filename() {
        let expected = [
            ("ornith-1.0-9b-Q4_K_M.gguf", "ornith_1_0_9b_q4_k_m"),
            ("ornith-1.0-9b-Q8_0.gguf", "Ornith 1.0 9B"),
            (
                "Llama-3.2-3B-Instruct-Q8_0.gguf",
                "llama32_3b_instruct_q8_0",
            ),
            ("Qwen3-4B-Q4_K_M.gguf", "qwen3_4b_q4_k_m"),
        ];
        for (filename, row_id) in expected {
            assert_eq!(
                tool_capable_row_for_filename(filename).map(|row| row.0),
                Some(row_id),
                "missing exact tool-capable mapping for {filename}"
            );
        }
        assert_eq!(
            tool_capable_row_for_filename("neighboring-model.gguf"),
            None
        );
        assert!(
            earned_tool_capable_row_for_filename("Qwen3-4B-Q8_0.gguf").is_some(),
            "precondition: the ledger still records the historical tool result"
        );
        assert_eq!(
            tool_capable_row_for_filename("Qwen3-4B-Q8_0.gguf"),
            None,
            "an unpinned row must not be advertised as currently authorizable"
        );
    }
}
