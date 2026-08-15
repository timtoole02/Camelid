use std::collections::VecDeque;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

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
use crate::chat::context_window::{
    configured_agent_context_max, select_context_window, ContextWindowInputs,
    ContextWindowSelection,
};
use crate::chat::workspace_bridge::{
    bridge, run_live, WorkspaceApprovalMode, WorkspaceBridgeControl, WorkspaceBridgeWorker,
    WorkspaceDecisionKind, WorkspaceEvent, WorkspaceRunConfig, WorkspaceRunMode,
};
use crate::chat::workspace_memory::{
    default_store_path, EvidenceInput, StoredThread, WorkspaceMemoryStore,
};

// Every generated token is one `model.delta` through this bounded channel, and
// the send BLOCKS. It used to be the browser's render loop that could throttle
// decode through it; the forwarder on the other end now drains into a retained
// session feed and never waits on a socket, so this is the worker's run-ahead
// against a busy forwarder and nothing more. A browser MUST NOT be able to
// backpressure decode on a turn it no longer owns.
//
// Kept deep anyway, because the worst-case memory it can pin is bounded — 1024 x
// the per-observation ceiling (`WEB_CODE_OBSERVATION_LIMIT`, tools.rs:292)
// rather than 1024 x "whatever the workspace printed". Both halves are
// load-bearing: a count-based bound is only a real bound once the item size has
// one too.
const EVENT_BACKLOG: usize = 1024;
/// Retained transcript depth, split by what a returning reader actually needs.
///
/// Structural entries — tool calls, results, approvals, notices, agent updates,
/// answers — are the transcript that EXPLAINS a run, so they are retained whole.
/// The browser renders at most `MAX_WORKSPACE_ACTIVITY_EVENTS` (240) of them, so
/// this is already deeper than any client will show, and the worst case is the
/// same bounded product as `EVENT_BACKLOG` above.
const EVENT_HISTORY_STRUCTURAL: usize = 1024;
/// Streamed model text is kept only deep enough for a reader that is LAGGING,
/// not one that was away: the UI shows a 2000-character tail of the current step
/// (`LIVE_TAIL_CHARS`) and nothing of earlier ones, and the finished text
/// arrives again as `model.answer`. Evicting these is therefore not a hole in
/// the transcript, which is why they get their own budget instead of pushing
/// tool calls out of history one token at a time.
const EVENT_HISTORY_DELTAS: usize = 512;
const DEFAULT_MAX_STEPS: usize = 12;
const MAX_STEPS: usize = 32;
// A coding step routinely carries a whole file in a `write_file` argument. At
// the old 512/1024 the call was cut off mid-JSON, parsed as no call at all, and
// landed in the transcript as a mangled "answer" with the write silently lost.
const DEFAULT_MAX_TOKENS: u32 = 2048;
const MAX_TOKENS: u32 = 8192;
// A written-out task spec — module layout, constraints, acceptance criteria — is
// the normal shape of a good coding goal, and 4 KiB rejected one outright AFTER
// the user had typed the whole thing. This is an abuse bound, not a prompt
// budget: what a goal can actually afford is the model's context window, which
// the context budget and auto-compaction already enforce downstream with real
// numbers. Keep the hard cap generous and let that machinery do the sizing.
const MAX_GOAL_BYTES: usize = 64 * 1024;
/// How long a turn may run before ANY `/events` response has ever attached to
/// it. This is the old `EVENT_CLAIM_TIMEOUT`, kept at its old value and its old
/// meaning: a POST whose GET never arrived is almost always a request that was
/// abandoned in flight, and 30 seconds is long enough to tell that apart from a
/// slow page load. It matters more than it used to, because the turn now starts
/// at POST rather than at GET, so those 30 seconds are real GPU.
const FIRST_ATTACH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// How long a turn may run with nobody actually watching it.
///
/// This is the entire replacement for cancel-on-disconnect, so it is sized
/// against what it protects rather than against the network. A refresh
/// re-attaches in about a second (bundle parse, mount, one 1s activity poll);
/// a browser relaunch takes tens of seconds; a resumed laptop does not spend
/// this window at all, because it is measured on a monotonic clock that macOS
/// stops during sleep. What it bounds is the tab that closed for good: at most
/// this much exclusive decode on a single-worker host, after which the turn is
/// cancelled exactly the way Stop cancels it.
///
/// The asymmetry sets the number. Ending a turn whose browser is two seconds
/// from coming back throws away minutes of GPU; waiting for one that never
/// comes back costs 90 seconds. Deliberate departures do not arrive here at
/// all — in-app navigation, Stop and Reset each cancel explicitly through
/// DELETE (CodeWorkspace.jsx:1149, :1355, :1410).
const ABANDON_GRACE: std::time::Duration = std::time::Duration::from_secs(90);
/// The unconditional ceiling on one turn, watched or not.
///
/// It exists because nothing else bounds a Code turn in wall-clock terms:
/// `workspace_max_steps` returns 0 for Code (:86-89), 0 means no step cap
/// (agent.rs:326, :979), and Code gets `set_stream_cancel` with no model-step
/// deadline (workspace_bridge.rs:818-822). Without this, a degenerate
/// repeating generate with a browser glued to it holds the machine's only
/// decode slot forever. Generous on purpose — a legitimate coding turn runs
/// 5-20 minutes, so this is 6-24x the real workload and is a runaway backstop,
/// not a policy.
const TURN_WALL_CLOCK_CEILING: std::time::Duration = std::time::Duration::from_secs(2 * 60 * 60);
/// Supervisor granularity. Deliberately coarse: the deadlines are minutes and
/// one tick costs a clock read plus four atomic loads.
const SUPERVISOR_TICK: std::time::Duration = std::time::Duration::from_secs(5);
const AUTO_COMPACT_TRIGGER_PERCENT: u32 = 75;

/// Reject an empty or over-long goal/message with the numbers, not just the rule.
/// "must contain 1 to 65536 UTF-8 bytes" leaves the author guessing how far over
/// they are and whether trimming a paragraph would be enough.
fn oversize_text_message(field: &str, value: &str) -> String {
    if value.is_empty() {
        return format!("{field} cannot be empty");
    }
    format!(
        "{field} is {} bytes, over the {} byte limit — trim about {} bytes and resend",
        value.len(),
        MAX_GOAL_BYTES,
        value.len().saturating_sub(MAX_GOAL_BYTES)
    )
}

/// Whether a stored thread belongs to `mode`, by its id prefix.
///
/// The prefix is the whole cross-mode boundary: a read-only Workspace thread
/// resumed as Code would inherit write tools it was never approved for, and a
/// Code thread resumed read-only would silently lose them. One function so the
/// resume gate and both thread listings cannot drift apart.
fn thread_id_belongs_to_mode(thread_id: &str, mode: WorkspaceRunMode) -> bool {
    let expected = if mode.is_code() {
        "code-"
    } else {
        "workspace-"
    };
    thread_id.starts_with(expected)
}

fn workspace_max_steps(mode: WorkspaceRunMode, requested: Option<usize>) -> Result<usize, ()> {
    if mode.is_code() {
        return Ok(0);
    }
    let max_steps = requested.unwrap_or(DEFAULT_MAX_STEPS);
    (1..=MAX_STEPS)
        .contains(&max_steps)
        .then_some(max_steps)
        .ok_or(())
}
const AUTO_COMPACT_MIN_TURNS: u32 = 4;

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

#[derive(Clone, Default)]
pub(super) struct WorkspaceSessionManager {
    active: Arc<Mutex<Option<Arc<ActiveWorkspaceSession>>>>,
}

struct ActiveWorkspaceSession {
    id: String,
    workspace: PathBuf,
    model_id: String,
    /// Digest of the artifact that opened the session. An id alone does not
    /// identify a model: a re-pulled or replaced GGUF keeps its filename, and
    /// an idle session survives an unload/reload, so follow-up turns check this
    /// too — the same exactness `create_session`'s resume path applies.
    model_sha256: String,
    /// The memory/model-aware context envelope resolved when this session was
    /// created. It stays fixed for the session so follow-up turns and spawned
    /// children share one predictable prompt/cache contract.
    context_window: ContextWindowSelection,
    max_steps: usize,
    max_tokens: u32,
    temperature: f32,
    allow_writes: bool,
    approval_mode: WorkspaceApprovalMode,
    allow_network: bool,
    mode: WorkspaceRunMode,
    semantic_retriever: Option<Arc<crate::chat::semantic_search::WorkspaceSemanticRetriever>>,
    memory: WorkspaceMemoryStore,
    state: StdMutex<WorkspaceSessionState>,
    events: StdMutex<Option<std::sync::mpsc::Receiver<WorkspaceEvent>>>,
    worker: StdMutex<Option<WorkspaceBridgeWorker>>,
    run_config: StdMutex<Option<WorkspaceRunConfig>>,
    control: StdMutex<Option<WorkspaceBridgeControl>>,
    current_turn: StdMutex<Option<(String, u32)>>,
    activity: StdMutex<WorkspaceActivitySnapshot>,
    feed: SessionFeed,
    watch: TurnWatch,
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
    Running,
    Idle,
    Cancelling,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct WorkspaceAgentActivity {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_id: Option<String>,
    label: String,
    status: String,
    task: String,
    detail: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct WorkspaceActivitySnapshot {
    started_at_ms: u64,
    updated_at_ms: u64,
    phase: String,
    stage: String,
    detail: String,
    task: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    current_tool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_model_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reused_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefilled_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_decision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    common_prefix_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    divergent_suffix_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_block_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    matched_cache_blocks: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    terminal_outcome: Option<String>,
    agents: Vec<WorkspaceAgentActivity>,
}

impl WorkspaceActivitySnapshot {
    fn new(task: &str) -> Self {
        let now = now_epoch_millis();
        Self {
            started_at_ms: now,
            updated_at_ms: now,
            phase: "starting".to_string(),
            stage: "starting".to_string(),
            detail: "Starting coding session".to_string(),
            task: task.to_string(),
            current_tool: None,
            output_tokens: None,
            total_model_ms: None,
            ttft_ms: None,
            prefill_ms: None,
            prompt_cache_hit: None,
            reused_tokens: None,
            prefilled_tokens: None,
            prompt_cache_decision: None,
            common_prefix_tokens: None,
            divergent_suffix_tokens: None,
            candidate_tokens: None,
            cache_block_tokens: None,
            matched_cache_blocks: None,
            terminal_outcome: None,
            agents: vec![WorkspaceAgentActivity {
                id: "main".to_string(),
                parent_id: None,
                label: "Camelid".to_string(),
                status: "starting".to_string(),
                task: task.to_string(),
                detail: "Preparing the coding session".to_string(),
            }],
        }
    }

    fn main_agent_mut(&mut self) -> &mut WorkspaceAgentActivity {
        let index = self
            .agents
            .iter()
            .position(|agent| agent.id == "main")
            .unwrap_or_else(|| {
                self.agents.insert(
                    0,
                    WorkspaceAgentActivity {
                        id: "main".to_string(),
                        parent_id: None,
                        label: "Camelid".to_string(),
                        status: "running".to_string(),
                        task: self.task.clone(),
                        detail: self.detail.clone(),
                    },
                );
                0
            });
        &mut self.agents[index]
    }

    fn sync_main(&mut self, status: Option<&str>) {
        let detail = self.detail.clone();
        let main = self.main_agent_mut();
        if let Some(status) = status {
            main.status = status.to_string();
        }
        main.detail = detail;
    }

    fn upsert_agent(&mut self, incoming: WorkspaceAgentActivity) {
        if let Some(existing) = self.agents.iter_mut().find(|agent| {
            agent.id == incoming.id
                || (agent.parent_id.is_some()
                    && incoming.parent_id.is_some()
                    && agent.label == incoming.label)
        }) {
            existing.id = incoming.id;
            existing.parent_id = incoming.parent_id;
            existing.label = incoming.label;
            existing.status = incoming.status;
            if !incoming.task.is_empty() {
                existing.task = incoming.task;
            }
            existing.detail = incoming.detail;
        } else {
            self.agents.push(incoming);
        }
    }

    fn apply(&mut self, event: &WorkspaceEvent) {
        self.updated_at_ms = now_epoch_millis();
        match event {
            WorkspaceEvent::Started { .. } => {
                self.phase = "running".to_string();
                self.stage = "starting".to_string();
                self.detail = "Preparing the model and workspace tools".to_string();
                self.sync_main(Some("running"));
            }
            WorkspaceEvent::TurnStarted { .. } => {
                self.phase = "running".to_string();
                self.stage = "context".to_string();
                self.detail = "Building the model context".to_string();
                self.sync_main(None);
            }
            WorkspaceEvent::MemoryUpdated {
                prompt_tokens,
                budget_total,
                ..
            } => {
                self.phase = "running".to_string();
                self.stage = "context".to_string();
                self.detail = format!(
                    "Prepared {prompt_tokens} prompt tokens inside a {budget_total}-token context"
                );
                self.sync_main(None);
            }
            WorkspaceEvent::ModelDelta { .. } => {
                self.phase = "running".to_string();
                self.stage = "generating".to_string();
                self.detail = "The model is generating its next action".to_string();
                self.current_tool = None;
                self.sync_main(Some("running"));
            }
            WorkspaceEvent::ModelTiming {
                total_ms,
                ttft_ms,
                output_tokens,
                prefill_ms,
                prompt_cache_hit,
                reused_tokens,
                prefilled_tokens,
                prompt_cache_decision,
                common_prefix_tokens,
                divergent_suffix_tokens,
                candidate_tokens,
                cache_block_tokens,
                matched_cache_blocks,
                ..
            } => {
                self.output_tokens = *output_tokens;
                self.total_model_ms = Some(*total_ms);
                self.ttft_ms = *ttft_ms;
                self.prefill_ms = *prefill_ms;
                self.prompt_cache_hit = *prompt_cache_hit;
                self.reused_tokens = *reused_tokens;
                self.prefilled_tokens = *prefilled_tokens;
                self.prompt_cache_decision = prompt_cache_decision.clone();
                self.common_prefix_tokens = *common_prefix_tokens;
                self.divergent_suffix_tokens = *divergent_suffix_tokens;
                self.candidate_tokens = *candidate_tokens;
                self.cache_block_tokens = *cache_block_tokens;
                self.matched_cache_blocks = *matched_cache_blocks;
                let mut detail = output_tokens.map_or_else(
                    || "The model finished a generation step".to_string(),
                    |tokens| format!("The model finished a {tokens}-token generation step"),
                );
                if let Some(hit) = prompt_cache_hit {
                    detail.push_str(if *hit {
                        " with a prompt-cache hit"
                    } else {
                        " with a prompt-cache miss"
                    });
                }
                if let Some(tokens) = reused_tokens.filter(|tokens| *tokens > 0) {
                    detail.push_str(&format!(" ({tokens} prompt tokens reused)"));
                }
                self.detail = detail;
                self.sync_main(None);
            }
            WorkspaceEvent::ModelAnswer { .. } => {
                self.phase = "running".to_string();
                self.stage = "finishing".to_string();
                self.detail = "Reviewing and saving the final answer".to_string();
                self.current_tool = None;
                self.sync_main(None);
            }
            WorkspaceEvent::ToolCall { detail } => {
                self.phase = "running".to_string();
                self.stage = "tool".to_string();
                self.detail = detail.clone();
                self.current_tool = Some(detail.clone());
                self.sync_main(Some("running"));
            }
            WorkspaceEvent::ToolResult { tool, outcome, .. } => {
                self.phase = "running".to_string();
                self.stage = "tool_result".to_string();
                self.detail = format!(
                    "{} {}",
                    tool.replace('_', " "),
                    if *outcome == "ok" {
                        "completed"
                    } else {
                        "failed"
                    }
                );
                self.current_tool = None;
                self.sync_main(Some("running"));
            }
            WorkspaceEvent::ApprovalRequired { tool, .. } => {
                self.phase = "awaiting_approval".to_string();
                self.stage = "approval".to_string();
                self.detail = format!("Waiting for approval to run {}", tool.replace('_', " "));
                self.sync_main(Some("waiting"));
            }
            WorkspaceEvent::Notice { content } => {
                self.detail = content.clone();
                self.sync_main(None);
            }
            WorkspaceEvent::AgentUpdated {
                agent_id,
                parent_id,
                label,
                status,
                task,
                detail,
            } => self.upsert_agent(WorkspaceAgentActivity {
                id: agent_id.clone(),
                parent_id: parent_id.clone(),
                label: label.clone(),
                status: status.clone(),
                task: task.clone(),
                detail: detail.clone(),
            }),
            WorkspaceEvent::Finished { outcome } => {
                self.phase = (*outcome).to_string();
                self.stage = "finished".to_string();
                self.current_tool = None;
                self.terminal_outcome = Some((*outcome).to_string());
                self.detail = terminal_activity_detail(outcome).to_string();
                let main_status = if *outcome == "answered" {
                    "completed"
                } else if matches!(*outcome, "driver_error" | "failed") {
                    "failed"
                } else {
                    "stopped"
                };
                for agent in &mut self.agents {
                    if agent.id == "main" {
                        agent.status = main_status.to_string();
                        agent.detail = self.detail.clone();
                    } else if matches!(agent.status.as_str(), "starting" | "running" | "waiting") {
                        agent.status = "stopped".to_string();
                        agent.detail = "The parent turn ended".to_string();
                    }
                }
            }
            WorkspaceEvent::Error { message } => {
                self.phase = "error".to_string();
                self.stage = "finished".to_string();
                self.detail = message.clone();
                self.current_tool = None;
                self.terminal_outcome = Some("error".to_string());
                let main = self.main_agent_mut();
                main.status = "failed".to_string();
                main.detail = message.clone();
            }
            WorkspaceEvent::MemoryCompacted { .. } => {}
        }
    }
}
/// Ordered replay history for one session, plus the wake every attached stream
/// waits on.
///
/// This is what makes `/events` an observer instead of an owner. The forwarder
/// writes here once; zero or more responses read from it at their own pace.
/// Nothing a reader does is visible to the writer, so no reader can end a turn.
/// That property IS the fix — the guard that used to sit on the response was
/// only ever a proxy for "is anyone still interested", and it answered a
/// question about a socket with a decision about a run.
struct SessionFeed {
    entries: StdMutex<FeedEntries>,
    /// Level-triggered wake carrying the newest sequence. `watch` and not
    /// `Notify`: it cannot lose a wakeup between a reader's drain and its next
    /// await, and it wakes every subscriber rather than one of them.
    tip: tokio::sync::watch::Sender<u64>,
}

impl Default for SessionFeed {
    fn default() -> Self {
        Self {
            entries: StdMutex::new(FeedEntries {
                first_structural: u64::MAX,
                ..FeedEntries::default()
            }),
            // The initial receiver is dropped on purpose: subscribers come and
            // go, and `send_replace` publishes whether or not any exist.
            tip: tokio::sync::watch::channel(0).0,
        }
    }
}

#[derive(Default)]
struct FeedEntries {
    /// Newest sequence handed out. Session-scoped and monotonic ACROSS turns, so
    /// a cursor a browser kept over a refresh still means the same thing
    /// afterwards. The old counter was per stream and restarted at 0 on every
    /// claim, which is why the client had to key its rendering on an arrival
    /// counter instead (CodeWorkspace.jsx:176-180).
    last: u64,
    /// Sequence the CURRENT turn started at. A reader that arrives without a
    /// cursor resumes from here, not from 0 — replaying a previous turn's
    /// `session.finished` into a live page makes the UI report a running turn
    /// as complete and then close the stream it just opened.
    turn_start: u64,
    /// The transcript: everything except streamed model text.
    structural: VecDeque<(u64, WorkspaceEvent)>,
    /// Streamed model text, on its own eviction budget.
    deltas: VecDeque<(u64, WorkspaceEvent)>,
    /// Oldest structural sequence still retained. Below this a reader has a real
    /// gap and is TOLD so; a transcript with a silent hole is worse than an
    /// admittedly short one.
    first_structural: u64,
}

impl FeedEntries {
    /// Mark where a newly installed turn begins. The sequence and the retained
    /// entries deliberately survive: a follow-up turn continues the same feed,
    /// so a client that kept its cursor across the boundary resumes without
    /// re-rendering what it already showed.
    fn begin_turn(&mut self) {
        self.turn_start = self.last;
    }

    fn record(&mut self, event: &WorkspaceEvent) -> u64 {
        self.last += 1;
        if matches!(event, WorkspaceEvent::ModelDelta { .. }) {
            self.deltas.push_back((self.last, event.clone()));
            while self.deltas.len() > EVENT_HISTORY_DELTAS {
                self.deltas.pop_front();
            }
        } else {
            self.structural.push_back((self.last, event.clone()));
            while self.structural.len() > EVENT_HISTORY_STRUCTURAL {
                self.structural.pop_front();
            }
            self.first_structural = self
                .structural
                .front()
                .map_or(u64::MAX, |(sequence, _)| *sequence);
        }
        self.last
    }

    /// Everything after `after`, in sequence order, and whether that range is
    /// actually complete. Two sorted deques merged by sequence — the split is an
    /// eviction policy, never an ordering one.
    fn since(&self, after: u64) -> (Vec<(u64, WorkspaceEvent)>, bool) {
        let complete =
            self.structural.is_empty() || after.saturating_add(1) >= self.first_structural;
        let structural = self
            .structural
            .partition_point(|(sequence, _)| *sequence <= after);
        let deltas = self
            .deltas
            .partition_point(|(sequence, _)| *sequence <= after);
        let mut merged: Vec<(u64, WorkspaceEvent)> = self
            .structural
            .iter()
            .skip(structural)
            .chain(self.deltas.iter().skip(deltas))
            .cloned()
            .collect();
        merged.sort_by_key(|(sequence, _)| *sequence);
        (merged, complete)
    }
}

/// Everything the supervisor reads to decide whether a turn is still wanted.
///
/// Atomics rather than a mutex on purpose. This is read on a 5-second tick and
/// written on every delivered event, and — more to the point — a reaper whose
/// inputs can be poisoned by an unrelated panic is a reaper that stops reaping.
/// Nothing here is ever left at its `Default`; `begin` stamps every clock.
#[derive(Default)]
struct TurnWatch {
    /// How many `/events` responses are attached right now.
    observers: AtomicUsize,
    /// Whether any `/events` response has ever attached to the current turn.
    ever_observed: AtomicBool,
    /// Monotonic ms at which `observers` last fell to zero.
    unobserved_since: AtomicU64,
    /// Highest feed sequence any attached response has actually written out.
    ///
    /// `observers > 0` proves a subscription EXISTS; only this proves one is
    /// consuming. It is the port of the `try_send`-on-`Full` bound that used to
    /// end a turn whose client stopped draining — a case a half-open TCP peer,
    /// a suspended renderer or a stale tab reconnect-looping across an upgrade
    /// can sustain indefinitely while the refcount stays at one.
    delivered: AtomicU64,
    /// Monotonic ms at which `delivered` last advanced.
    delivered_at: AtomicU64,
    /// Monotonic ms at which the current turn was installed.
    turn_started: AtomicU64,
}

impl TurnWatch {
    fn begin(&self) {
        let now = monotonic_millis();
        self.observers.store(0, Ordering::Release);
        self.ever_observed.store(false, Ordering::Release);
        self.unobserved_since.store(now, Ordering::Release);
        self.delivered.store(0, Ordering::Release);
        self.delivered_at.store(now, Ordering::Release);
        self.turn_started.store(now, Ordering::Release);
    }

    fn note_delivered(&self, sequence: u64) {
        if self.delivered.fetch_max(sequence, Ordering::AcqRel) < sequence {
            self.delivered_at
                .store(monotonic_millis(), Ordering::Release);
        }
    }
}

fn terminal_activity_detail(outcome: &str) -> &'static str {
    match outcome {
        "answered" => "The coding turn completed",
        "repeated" => "Stopped because the model repeated actions without making progress",
        "step_capped" => "Stopped after reaching the step limit",
        "aborted" | "cancelled" => "The coding turn was stopped",
        "driver_error" | "failed" => "The model or runtime failed",
        _ => "The coding turn ended",
    }
}

fn now_epoch_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}
/// Milliseconds since the first call, on a clock that only moves forward.
///
/// Every reaper deadline is measured with this and never with
/// `now_epoch_millis`. `SystemTime` is subject to NTP steps and manual clock
/// changes: a backwards correction pins every elapsed-time subtraction at zero,
/// which would silently disable the only thing that ends an abandoned turn, and
/// a forward jump would kill healthy ones. On macOS this also excludes system
/// suspend, so a closed lid pauses the grace window rather than spending it.
fn monotonic_millis() -> u64 {
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    ORIGIN
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

impl WorkspaceSessionState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Idle => "idle",
            Self::Cancelling => "cancelling",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }

    fn blocks_model_transition(self) -> bool {
        matches!(self, Self::Running | Self::Cancelling)
    }

    fn accepts_new_turn(self) -> bool {
        matches!(self, Self::Idle | Self::Cancelled | Self::Failed)
    }

    /// `WaitingForEvents` is gone because nothing waits for events any more: a
    /// turn is running before the POST response is written. `Cancelling` keeps
    /// `blocks_model_transition` true until the worker has actually exited,
    /// which is what makes an unload or replace during teardown fail closed.
    fn after_cancel_request(self) -> Self {
        match self {
            Self::Running => Self::Cancelling,
            Self::Idle => Self::Cancelled,
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
        let mut activity = self
            .activity
            .lock()
            .map_err(|_| "activity state is unavailable")?;
        *activity = WorkspaceActivitySnapshot::new(&run_config.goal);
        if let Ok(mut entries) = self.feed.entries.lock() {
            entries.begin_turn();
        }
        // Every clock the supervisor reads restarts here. A follow-up sent from
        // a page whose stream is closed gets its own full grace rather than
        // inheriting however long the session had already sat unwatched.
        self.watch.begin();
        *current_turn = Some((run_config.client_message_id.clone(), run_config.turn_index));
        *event_slot = Some(events);
        *worker_slot = Some(worker);
        *config_slot = Some(run_config);
        *control_slot = Some(control);
        *status = WorkspaceSessionState::Running;
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
        // Published BEFORE the turn is settled, and through the feed rather than
        // straight into the activity snapshot. Both halves matter: a reader
        // breaks out of its loop once `current_turn` is None, so a terminal
        // published afterwards is never drained; and a terminal that only ever
        // reaches the snapshot leaves every attached response parked forever on
        // a `tip` that will never change again.
        self.publish(&WorkspaceEvent::Finished { outcome: "aborted" });
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

    fn record_activity(&self, event: &WorkspaceEvent) {
        if let Ok(mut activity) = self.activity.lock() {
            activity.apply(event);
        }
    }
    /// The one place an event becomes visible: the pollable snapshot, the replay
    /// history, and the wake for every attached stream, in that order.
    fn publish(&self, event: &WorkspaceEvent) {
        self.record_activity(event);
        let sequence = match self.feed.entries.lock() {
            Ok(mut entries) => entries.record(event),
            // The ring has no invariant a panic could half-break, and an event
            // that never reaches the feed is an event no reader can ever see.
            Err(poisoned) => poisoned.into_inner().record(event),
        };
        self.feed.tip.send_replace(sequence);
    }

    /// Fails CLOSED: an unreadable slot is not evidence that this turn is
    /// someone else's problem, and the supervisor's exit condition is the only
    /// thing standing between a bug and a pinned GPU.
    fn owns_turn(&self, message_id: &str) -> bool {
        match self.current_turn.lock() {
            Ok(turn) => turn
                .as_ref()
                .is_some_and(|(current_id, _)| current_id == message_id),
            Err(_) => true,
        }
    }

    /// Pull the same lever Stop pulls, and mark the session as stopping.
    ///
    /// The cancel flag is the ONLY lever that reaps delegated child processes:
    /// they are torn down by `WorkspaceSubagentTurnGuard::drop`
    /// (workspace_bridge.rs:373-377), which runs on the worker thread when
    /// `run_live` returns, and the registry is thread-local so no other thread
    /// can reach them. Anything that ends a turn without setting this flag
    /// leaves subagents writing into the workspace.
    fn request_cancel(&self) {
        let control = match self.control.lock() {
            Ok(control) => control.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        if let Some(control) = control {
            control.cancel();
        }
        if let Ok(mut status) = self.state.lock() {
            *status = status.after_cancel_request();
        }
    }

    /// Why this turn should be stopped without anyone asking, if it should.
    ///
    /// Three deadlines, all monotonic, all failing closed on unreadable state.
    /// The middle one is the interesting one: "watched" is not "a response
    /// object exists", it is "somebody is keeping up". A reader that stops
    /// consuming while the feed runs ahead is indistinguishable from a reader
    /// that vanished, and both used to be handled — by killing the turn on the
    /// spot. Now they buy the same bounded grace as everything else.
    fn abandonment_reason(&self) -> Option<String> {
        self.abandonment_reason_at(monotonic_millis())
    }

    /// `now` is a parameter so the deadlines are testable without sleeping
    /// through them. `monotonic_millis` counts from the first call in the
    /// process, so in a test it is a handful of milliseconds and backdating a
    /// clock saturates at zero — no amount of arithmetic on the stored stamps
    /// can reach a 30-second deadline. Passing the instant in makes every
    /// branch below reachable and deterministic.
    fn abandonment_reason_at(&self, now: u64) -> Option<String> {
        let ran_for = now.saturating_sub(self.watch.turn_started.load(Ordering::Acquire));
        if ran_for >= TURN_WALL_CLOCK_CEILING.as_millis() as u64 {
            return Some(format!(
                "This turn ran for {} hours without finishing, so Camelid stopped it.",
                TURN_WALL_CLOCK_CEILING.as_secs() / 3600
            ));
        }
        if !self.watch.ever_observed.load(Ordering::Acquire) {
            return (ran_for >= FIRST_ATTACH_TIMEOUT.as_millis() as u64).then(|| {
                "No browser ever attached to this turn, so Camelid stopped it.".to_string()
            });
        }
        let unattended_since = if self.watch.observers.load(Ordering::Acquire) == 0 {
            self.watch.unobserved_since.load(Ordering::Acquire)
        } else {
            let published = match self.feed.entries.lock() {
                Ok(entries) => entries.last,
                // Cannot prove anyone is keeping up.
                Err(_) => u64::MAX,
            };
            if published <= self.watch.delivered.load(Ordering::Acquire) {
                // Nothing to keep up WITH. A silent turn is not the reader's
                // fault; that case belongs to the ceiling above.
                return None;
            }
            self.watch.delivered_at.load(Ordering::Acquire)
        };
        (now.saturating_sub(unattended_since) >= ABANDON_GRACE.as_millis() as u64).then(|| {
            format!(
                "No browser watched this turn for {} seconds, so Camelid stopped it.",
                ABANDON_GRACE.as_secs()
            )
        })
    }
}

/// Start the worker and the event forwarder for the turn just installed.
///
/// Called by whoever INSTALLS a turn, never by whoever watches one. That single
/// move is the decoupling: `/events` no longer starts anything, so there is no
/// first consumer to lose, no claim to expire, and nothing about a socket
/// anywhere in a turn's lifetime.
fn start_turn(session: &Arc<ActiveWorkspaceSession>, message_id: String) {
    // Armed FIRST, before anything below can bail. The supervisor is the only
    // thing that can end an unattended turn, and the arm below — a slot that is
    // unexpectedly empty — is exactly the path where nothing else ever will.
    supervise_turn(session, message_id.clone());

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
        // `install_turn` fills all four under the state lock and this runs once
        // per install, so this is a broken invariant rather than a caller's
        // mistake — most plausibly a poisoned slot. Fail the turn loudly instead
        // of returning: a `Running` session with no worker blocks every model
        // load, unload and new session for the life of the process.
        debug_assert!(
            false,
            "install_turn fills every turn slot under the state lock"
        );
        eprintln!("Workspace turn {message_id} was installed without a worker; failing it");
        session.publish(&WorkspaceEvent::Error {
            message: "Camelid could not start this coding turn.".to_string(),
        });
        session.finish_turn_if_current(&message_id, TurnCompletion::Failed);
        return;
    };

    let persisted_turn = run_config.clone();
    std::thread::Builder::new()
        .name("camelid-workspace-agent".to_string())
        .spawn(move || run_workspace_agent(run_config, worker))
        .expect("spawn Workspace agent thread");

    let forward_control = control;
    let persist_session = Arc::clone(session);
    std::thread::Builder::new()
        .name("camelid-workspace-events".to_string())
        .spawn(move || {
            forward_workspace_events(persist_session, events, persisted_turn, forward_control)
        })
        .expect("spawn Workspace event forwarder");
}

/// Run the model-side half of a Workspace turn without settling the session.
///
/// `run_live` publishes its terminal events before it returns. Those events may
/// still be sitting in the bounded bridge when this function returns, so only
/// `forward_workspace_events` may persist them, publish them, and clear the
/// current turn. Settling here races the forwarder and lets event readers stop
/// before the queued fallback answer and terminal event become visible.
fn run_workspace_agent(run_config: WorkspaceRunConfig, worker: WorkspaceBridgeWorker) {
    let delivery_failed = Arc::clone(&worker.delivery_failed);
    let _ = run_live(run_config, worker);
    if delivery_failed.load(Ordering::Acquire) {
        // Diagnostic only. The receiver closes after the forwarder has either
        // settled its fallback path or failed; it is never safe to race that
        // owner by clearing the turn from this thread.
        eprintln!("Workspace agent event delivery ended before the worker returned");
    }
}

/// The only thing that ends an unattended turn.
///
/// Socket liveness used to be this rule, which is exactly why a refresh killed a
/// run. Its replacement has to be bounded and self-healing WITHOUT a socket, so
/// it is a slow tick over `abandonment_reason`'s three deadlines. Shaped like
/// the claim deadline it replaces — a detached task holding a `Weak`, re-checking
/// the turn identity every tick — so it cannot outlive its turn, cannot keep the
/// session alive, and cannot act on a later one.
///
/// It does NOT return after asking once. Cancellation is cooperative: a
/// `write_file` already dispatched completes, a `run_shell` can sit up to
/// WEB_CODE_SHELL_TIMEOUT, and `request_cancel` is idempotent. The only exit is
/// the turn actually going away.
fn supervise_turn(session: &Arc<ActiveWorkspaceSession>, message_id: String) {
    let session = Arc::downgrade(session);
    tokio::spawn(async move {
        let mut announced = false;
        let mut ticks_since_ask = 0_u32;
        loop {
            tokio::time::sleep(SUPERVISOR_TICK).await;
            let Some(session) = session.upgrade() else {
                return;
            };
            if !session.owns_turn(&message_id) {
                return;
            }
            let Some(reason) = session.abandonment_reason() else {
                continue;
            };
            if !announced {
                announced = true;
                // Published before the cancel, so a reader still attached — or
                // one that re-attaches later and replays — is told WHY the turn
                // ended instead of watching it stop for no stated reason.
                session.publish(&WorkspaceEvent::Notice { content: reason });
            } else {
                ticks_since_ask = ticks_since_ask.saturating_add(1);
                if ticks_since_ask.is_multiple_of(12) {
                    eprintln!(
                        "Workspace turn {message_id} has not stopped {}s after it was reaped",
                        u64::from(ticks_since_ask) * SUPERVISOR_TICK.as_secs()
                    );
                }
            }
            session.request_cancel();
        }
    });
}

fn forward_workspace_events(
    persist_session: Arc<ActiveWorkspaceSession>,
    events: std::sync::mpsc::Receiver<WorkspaceEvent>,
    persisted_turn: WorkspaceRunConfig,
    forward_control: WorkspaceBridgeControl,
) {
    let mut pending_call = None;
    let mut evidence = Vec::new();
    let mut last_context_usage = None;
    let mut assistant_answer = None;
    let mut persistence_attempted = false;
    while let Ok(event) = events.recv() {
        // EDIT 1 (was `record_activity` at :1943): one publish path, and the
        // TERMINAL is deferred. Publishing `Finished{answered}` before
        // `append_terminal_turn` has returned would tell an attached browser the
        // turn succeeded and let it close the stream, so a failed memory write
        // would render as a success.
        let terminal = matches!(event, WorkspaceEvent::Finished { .. });
        if !terminal {
            persist_session.publish(&event);
        }
        if let WorkspaceEvent::MemoryUpdated {
            prompt_tokens,
            generation_tokens,
            budget_total,
            ..
        } = &event
        {
            last_context_usage = Some((*prompt_tokens, *generation_tokens, *budget_total));
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
                // EDIT 2 (was :1978-1988): publish, THEN settle. A reader breaks
                // out of its loop once the turn is settled, so a terminal
                // published afterwards is never drained. This cancel stays
                // immediate — nothing inside `run_loop` reads `delivery_failed`,
                // so without it a broken forwarder leaves the agent running
                // invisibly, still writing files.
                persist_session.publish(&WorkspaceEvent::Error {
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
                if let Some((prompt_tokens, generation_tokens, budget_total)) = last_context_usage {
                    let thread = persist_session.memory.thread(&persist_session.id);
                    if let Ok(Some(thread)) = thread {
                        if should_auto_compact(
                            thread.turn_count,
                            prompt_tokens,
                            generation_tokens,
                            budget_total,
                        ) {
                            match persist_session.memory.compact_thread(&persist_session.id) {
                                Ok(result) if result.archived_turns > 0 => {
                                    automatic_compaction = Some(WorkspaceEvent::MemoryCompacted {
                                        compacted_through_turn: result.compacted_through_turn,
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
            // EDIT 3 (was :2041-2047): compaction is published BEFORE the
            // terminal rather than after it, so `Finished` is unambiguously the
            // last event of a turn. Under the old channel it was sent after a
            // reader had already broken on the terminal, i.e. never delivered.
            if let Some(compaction) = automatic_compaction {
                persist_session.publish(&compaction);
            }
            persist_session.publish(&event);
            let completion = if *outcome == "driver_error" {
                TurnCompletion::Failed
            } else {
                TurnCompletion::Idle
            };
            persist_session.finish_turn_if_current(&persisted_turn.client_message_id, completion);
        }
        // EDIT 4: the forwarder->SSE channel is gone, so both `try_send` cancels
        // (:2037-2040, :2043-2046) go with it. A reader that is absent or slow is
        // no longer an event at all, let alone a reason to end a run.
    }
    if !persistence_attempted {
        if let Err(error) =
            persist_session.persist_aborted_turn_and_finish(&persisted_turn, &evidence)
        {
            eprintln!("Workspace memory could not save an interrupted turn: {error}");
        }
    }
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
    #[serde(default)]
    mode: WorkspaceRunMode,
    #[serde(default)]
    approval_mode: WorkspaceApprovalMode,
    #[serde(default)]
    allow_network: bool,
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
    context_window: ContextWindowSelection,
    allow_writes: bool,
    approval_mode: WorkspaceApprovalMode,
    allow_network: bool,
    mode: WorkspaceRunMode,
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
    context_window: ContextWindowSelection,
    resident_cuda: Option<crate::inference::ResidentCudaStatus>,
    allow_writes: bool,
    approval_mode: WorkspaceApprovalMode,
    allow_network: bool,
    mode: WorkspaceRunMode,
    semantic_retrieval: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_model_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct WorkspaceActivityResponse {
    id: String,
    workspace: String,
    model_id: String,
    state: &'static str,
    context_window: ContextWindowSelection,
    approval_mode: WorkspaceApprovalMode,
    allow_network: bool,
    mode: WorkspaceRunMode,
    #[serde(flatten)]
    activity: WorkspaceActivitySnapshot,
}

#[derive(Debug, Serialize)]
struct WorkspaceActivityEnvelope {
    #[serde(skip_serializing_if = "Option::is_none")]
    activity: Option<WorkspaceActivityResponse>,
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
    #[serde(default)]
    mode: WorkspaceRunMode,
}

#[derive(Debug, Deserialize)]
pub(super) struct WorkspaceRecentThreadsQuery {
    #[serde(default)]
    mode: WorkspaceRunMode,
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
    /// Set on the first entry of an incomplete replay: this reader's cursor was
    /// older than the retained history, so earlier steps of the turn are missing
    /// from this feed. Carried on the envelope rather than emitted as a
    /// synthetic event so sequences stay monotonic and a client can keep
    /// deduplicating on them alone.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    replay_gap: bool,
    #[serde(flatten)]
    event: WorkspaceEvent,
}

/// Attach/detach bookkeeping for one `/events` response.
///
/// The replacement for `CancelStreamOnDrop`, and the replacement IS the fix: a
/// dropped stream now records that nobody is watching, which the supervisor may
/// act on ninety seconds later, instead of ending the turn on the spot.
/// Dropping a socket is no longer a decision about a run.
struct ObserverGuard(Arc<ActiveWorkspaceSession>);

impl ObserverGuard {
    fn attach(session: &Arc<ActiveWorkspaceSession>) -> Self {
        session.watch.ever_observed.store(true, Ordering::Release);
        session.watch.observers.fetch_add(1, Ordering::AcqRel);
        Self(Arc::clone(session))
    }
}

impl Drop for ObserverGuard {
    fn drop(&mut self) {
        if self.0.watch.observers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0
                .watch
                .unobserved_since
                .store(monotonic_millis(), Ordering::Release);
        }
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
    let mode = query.mode;
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
            .filter(|thread| {
                let mode_matches = thread_id_belongs_to_mode(&thread.id, mode);
                mode_matches && thread.model_id == model_id && thread.model_sha256 == model_sha256
            })
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

#[derive(Debug, Serialize)]
struct WorkspaceChangesResponse {
    summary: String,
    diff: String,
    files: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct WorkspaceUndoRequest {
    #[serde(default)]
    force: bool,
}

#[derive(Debug, Serialize)]
struct WorkspaceUndoResponse {
    result: String,
    summary: String,
    diff: String,
    files: Vec<String>,
}

pub(super) async fn list_recent_threads(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<WorkspaceRecentThreadsQuery>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let mode = query.mode;
    let result = match run_workspace_blocking(move || -> anyhow::Result<_> {
        let store = WorkspaceMemoryStore::open(default_store_path())?;
        Ok(store
            .recent_threads(100)?
            .into_iter()
            .filter(|thread| thread_id_belongs_to_mode(&thread.id, mode))
            .take(40)
            .collect::<Vec<_>>())
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
            oversize_text_message("goal", &goal),
            Some("goal"),
        );
    }
    let mode = request.mode;
    // Code mode is user-cancellable and has a result-aware no-progress guard,
    // so it does not impose an arbitrary number of model/tool turns. A zero
    // internal value means unlimited; read-only Workspace keeps its bounded
    // request contract.
    let max_steps = match workspace_max_steps(mode, request.max_steps) {
        Ok(max_steps) => max_steps,
        Err(()) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_workspace_limits",
                format!("max_steps must be between 1 and {MAX_STEPS}"),
                Some("max_steps"),
            )
        }
    };
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
    let approval_mode = request.approval_mode;
    let allow_network = request.allow_network;
    if request.allow_writes.unwrap_or(false) && !mode.is_code() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "workspace_read_only",
            "Workspace is read-only; write and edit tools are not available".to_string(),
            Some("allow_writes"),
        );
    }
    if approval_mode.is_full_auto() && !mode.is_code() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "workspace_read_only",
            "Full-auto approval mode is available only in Code mode".to_string(),
            Some("approval_mode"),
        );
    }
    if allow_network && !mode.is_code() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "workspace_read_only",
            "Network and web-search tools are available only in Code mode".to_string(),
            Some("allow_network"),
        );
    }
    if approval_mode.is_full_auto() && crate::chat::agent::is_production() {
        return api_error(
            StatusCode::FORBIDDEN,
            "workspace_full_auto_refused",
            "Full auto is disabled while CAMELID_PRODUCTION is set".to_string(),
            Some("approval_mode"),
        );
    }
    let allow_writes = mode.is_code();

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

    let (model, family) = match active_tool_capable_model(&state).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    // Dense model registration parses GGUF metadata and tensor bindings but
    // intentionally defers the multi-gigabyte weight materialization until the
    // first generation. Sampling "available RAM" before that allocation makes
    // the adaptive KV budget count the same memory twice. Warm the exact weights
    // first (runnable-only architectures already initialize their runtime at
    // model load), then take the live memory snapshot used for this session.
    if let Some(binding) = model.llama_tensors.as_ref() {
        if let Err(response) = super::load_weights_lru(&state, &model, binding).await {
            return response;
        }
    }
    let paging_config = mode
        .is_code()
        .then(crate::chat::context_paging::ContextPagingConfig::from_env);
    let enabled_paging_config = paging_config.as_ref().filter(|config| config.enabled);
    let (context_window, enable_single_kv_owner_mode) =
        select_workspace_context_window(&state, &model, max_tokens, enabled_paging_config);
    if let Some((memory_safe, required)) = workspace_context_memory_shortfall(&context_window) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "workspace_context_memory_insufficient",
            format!(
                "the active model has memory for about {memory_safe} context tokens, below the required {required}-token Workspace envelope; close other memory-heavy work or choose a smaller model"
            ),
            None,
        );
    }
    eprintln!(
        "[workspace-context] model={} selected={} validated={} native={} recommended={} memory_safe={} kv_owners={} limited_by={:?} available_ram_mib={} kv_bytes_per_token={} resident_capacity={} paged_target={} paged_working_set={}",
        model.id,
        context_window.effective_tokens,
        context_window.validated_max_tokens,
        context_window.model_max_tokens,
        context_window.recommended_max_tokens,
        context_window.memory_safe_max_tokens.unwrap_or(0),
        context_window.kv_owner_slots,
        context_window.limiting_factor,
        context_window
            .available_memory_bytes
            .map(|bytes| bytes / (1024 * 1024))
            .unwrap_or(0),
        context_window.kv_bytes_per_token.unwrap_or(0),
        context_window.resident_capacity_tokens.unwrap_or(0),
        context_window.paged_target_tokens.unwrap_or(0),
        context_window.paged_working_set_tokens.unwrap_or(0),
    );
    // Semantic retrieval is a read-only Workspace feature: the session-scoped
    // index is built once and never invalidated, which is only sound while the
    // workspace cannot change under it. Code mode writes files, so its turns
    // must not be fed pre-edit excerpts labeled as live workspace content.
    let semantic_retriever = if mode.is_code() {
        None
    } else {
        workspace_semantic_retriever(&state, &workspace).await
    };
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
        // The finished session is NOT evicted here. Everything below can still
        // fail — wrong mode, wrong artifact, unknown thread, memory store — and
        // a refused request must not destroy the session it refused to replace:
        // that session still owns the Changes view and the guarded undo for work
        // already applied to the user's files, both of which resolve through the
        // active slot. It is replaced wholesale on success (`*active = Some(..)`),
        // and this lock is held throughout, so nothing observes a stale slot.
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
            if !thread_id_belongs_to_mode(&thread_id, mode) {
                return Ok(Err("mode_mismatch"));
            }
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
            let prefix = if mode.is_code() { "code" } else { "workspace" };
            let id = format!("{prefix}-{}", uuid::Uuid::new_v4());
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
        Ok(Err("mode_mismatch")) => {
            return api_error(
                StatusCode::CONFLICT,
                "workspace_thread_mode_mismatch",
                "the saved thread belongs to a different Workspace mode".to_string(),
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
        family,
        max_steps,
        max_tokens,
        context_budget_tokens: context_window.effective_tokens,
        temperature,
        mode,
        approval_mode,
        allow_network,
        semantic_retriever: semantic_retriever.clone(),
    };
    if mode.is_code() {
        // Clears the workspace journal too, so this session's first undo cannot
        // walk back into a previous session's (or its subagents') changes.
        crate::chat::checkpoint::clear_for_workspace(&workspace);
    }
    let session = Arc::new(ActiveWorkspaceSession {
        id: id.clone(),
        workspace: workspace.clone(),
        model_id: model.id.clone(),
        model_sha256: model.lane.gguf_sha256.to_string(),
        context_window,
        max_steps,
        max_tokens,
        temperature,
        allow_writes,
        approval_mode,
        allow_network,
        mode,
        semantic_retriever,
        memory,
        state: StdMutex::new(WorkspaceSessionState::Running),
        events: StdMutex::new(Some(events)),
        worker: StdMutex::new(Some(worker)),
        run_config: StdMutex::new(Some(run_config)),
        control: StdMutex::new(Some(control)),
        current_turn: StdMutex::new(Some((client_message_id.clone(), turn_index))),
        activity: StdMutex::new(WorkspaceActivitySnapshot::new(&goal)),
        feed: SessionFeed::default(),
        watch: TurnWatch::default(),
    });
    session.watch.begin();
    if enable_single_kv_owner_mode {
        // The measured aggregate budget cannot hold the selected active
        // envelope across concurrent streams plus the retained/mirrored prompt
        // cache. Apply both halves only after session preparation has
        // succeeded, but before the first agent request can allocate KV:
        // engine serialization prevents active overlap, and disabling retained
        // prompt entries prevents cache publication from recreating an extra
        // KV owner. If even one active owner is short, the selection keeps that
        // raw shortfall visible to the allocator guard.
        state.engine.enable_single_kv_owner_mode();
        super::disable_prompt_prefix_cache(&state);
    }
    *active = Some(Arc::clone(&session));
    // The turn starts HERE, not when a browser opens `/events`. That single move
    // is the fix: there is no claim to lose, so a refresh between this POST and
    // the GET that used to start the work no longer costs the turn. Called with
    // the session already published so a Stop that lands in this microsecond
    // finds it and is honoured by the loop's first cancel check.
    start_turn(&session, client_message_id);
    drop(active);

    (
        StatusCode::CREATED,
        Json(WorkspaceSessionResponse {
            id,
            workspace: simplify_path(&workspace),
            model_id: model.id,
            state: WorkspaceSessionState::Running.as_str(),
            max_steps,
            max_tokens,
            context_window,
            allow_writes,
            approval_mode,
            allow_network,
            mode,
            semantic_retrieval: embedding_model_id.is_some(),
            embedding_model_id,
        }),
    )
        .into_response()
}

pub(super) async fn session_changes(
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
    if !session.mode.is_code() {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_read_only",
            "File changes are available only in Code mode".to_string(),
            None,
        );
    }
    let workspace = session.workspace.clone();
    match run_workspace_blocking(move || workspace_changes_response(&workspace)).await {
        Ok(Ok(changes)) => Json(changes).into_response(),
        Ok(Err(error)) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_changes_unavailable",
            error.to_string(),
            None,
        ),
        Err(response) => response,
    }
}

pub(super) async fn undo_session_change(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<WorkspaceUndoRequest>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let session = match find_session(&state, &id).await {
        Ok(session) => session,
        Err(response) => return response,
    };
    if !session.mode.is_code() {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_read_only",
            "Undo is available only in Code mode".to_string(),
            None,
        );
    }
    if session
        .state
        .lock()
        .map(|status| status.blocks_model_transition())
        .unwrap_or(true)
    {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_turn_active",
            "stop or finish the active turn before undoing a file change".to_string(),
            None,
        );
    }
    let workspace = session.workspace.clone();
    let changes_workspace = workspace.clone();
    let result = match run_workspace_blocking(move || {
        let sandbox = crate::chat::tools::Sandbox::new(
            &workspace,
            false,
            std::time::Duration::from_secs(30),
        )?;
        // Undo must walk back the newest change in the WORKSPACE, which may be a
        // subagent's — see `workspace_changes_response`.
        crate::chat::checkpoint::sync_from_store(sandbox.root());
        crate::chat::checkpoint::undo(&sandbox, request.force).map_err(anyhow::Error::msg)
    })
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    match result {
        Ok(result) => match workspace_changes_response(&changes_workspace) {
            Ok(changes) => Json(WorkspaceUndoResponse {
                result,
                summary: changes.summary,
                diff: changes.diff,
                files: changes.files,
            })
            .into_response(),
            Err(error) => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "workspace_changes_unavailable",
                error.to_string(),
                None,
            ),
        },
        Err(error) => api_error(
            StatusCode::CONFLICT,
            "workspace_undo_refused",
            error.to_string(),
            None,
        ),
    }
}

fn workspace_changes_response(
    workspace: &std::path::Path,
) -> anyhow::Result<WorkspaceChangesResponse> {
    let sandbox =
        crate::chat::tools::Sandbox::new(workspace, false, std::time::Duration::from_secs(30))?;
    // Subagents write from their own processes, so their checkpoints live only
    // in the workspace journal until this pulls them in. Without it the change
    // set silently omits every file a delegated child touched.
    crate::chat::checkpoint::sync_from_store(sandbox.root());
    let checkpoints = crate::chat::checkpoint::all();
    Ok(WorkspaceChangesResponse {
        summary: crate::chat::checkpoint::summary(),
        diff: crate::chat::checkpoint::diff(&sandbox),
        files: checkpoints
            .into_iter()
            .map(|checkpoint| checkpoint.rel)
            .collect(),
    })
}

#[derive(Debug, Deserialize)]
pub(super) struct WorkspaceEventsQuery {
    /// Resume cursor: the highest envelope `sequence` this client has already
    /// applied. Taken as a string and parsed here so a malformed value gets this
    /// file's `api_error` JSON rather than axum's plain-text `Query` rejection.
    #[serde(default)]
    after: Option<String>,
}

/// A pure observer of a turn that is running whether or not anyone is watching.
///
/// Attaching starts nothing, consumes nothing, and excludes nobody: several
/// responses may follow the same turn at once, which is what makes a refresh
/// safe even while the socket it replaced is still being torn down.
pub(super) async fn session_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<WorkspaceEventsQuery>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let session = match find_session(&state, &id).await {
        Ok(session) => session,
        Err(response) => return response,
    };
    let requested_cursor = match query.after.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(value) => match value.parse::<u64>() {
            Ok(cursor) => Some(cursor),
            Err(_) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_workspace_event_cursor",
                    "after must be a whole number event sequence".to_string(),
                    Some("after"),
                )
            }
        },
    };
    // `Last-Event-ID` first because it only EXISTS on the browser's own
    // reconnect, where it is by construction fresher than the `?after=` frozen
    // into the URL when the EventSource was constructed. The query parameter is
    // the one that covers a page reload, which builds a brand-new EventSource
    // and never sends the header. Absent both, resume from the start of the
    // CURRENT turn — never from 0, which would replay a previous turn's
    // `session.finished` into a live page.
    let resume_from = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .or(requested_cursor)
        .unwrap_or_else(|| match session.feed.entries.lock() {
            Ok(entries) => entries.turn_start,
            Err(poisoned) => poisoned.into_inner().turn_start,
        });
    // An approval prompt is the one replayed event that can be actively wrong:
    // the loop may already have its decision, and re-rendering the card gives
    // the user buttons whose POST will 409. `try_decide` rejects a stale id
    // (workspace_bridge.rs:435-443), so the live pending id is the exact test.
    let pending_approval = session
        .control
        .lock()
        .ok()
        .and_then(|control| control.clone())
        .and_then(|control| control.pending_approval_id());

    let session_id = session.id.clone();
    let mut tip = session.feed.tip.subscribe();
    let observer = ObserverGuard::attach(&session);
    let stream = async_stream::stream! {
        // Held for the life of the response. Dropping it records that nobody is
        // watching; it does not end anything.
        let _observer = observer;
        let mut cursor = resume_from;
        let mut gap_pending = false;
        let mut replaying = true;
        loop {
            // Read "has the turn settled" BEFORE draining, so anything published
            // before we looked is still delivered by this pass. Fails closed on
            // an unreadable slot: ending the response sends the client to the
            // status poll, which is the recoverable direction.
            let settled = session
                .current_turn
                .lock()
                .map(|turn| turn.is_none())
                .unwrap_or(true);
            // Scoped deliberately: a std `MutexGuard` held across a `yield`
            // makes this generator non-Send and axum will not accept it.
            let (batch, complete) = {
                match session.feed.entries.lock() {
                    Ok(entries) => entries.since(cursor),
                    Err(poisoned) => poisoned.into_inner().since(cursor),
                }
            };
            if replaying {
                gap_pending = !complete;
            }
            for (sequence, event) in batch {
                cursor = sequence;
                if replaying {
                    if let WorkspaceEvent::ApprovalRequired { approval_id, .. } = &event {
                        if pending_approval.as_deref() != Some(approval_id.as_str()) {
                            continue;
                        }
                    }
                }
                let envelope = WorkspaceEventEnvelope {
                    sequence,
                    session_id: session_id.clone(),
                    replay_gap: std::mem::take(&mut gap_pending),
                    event,
                };
                match serde_json::to_string(&envelope) {
                    Ok(json) => {
                        // Stamped only once the frame is actually handed to the
                        // response body. This is what tells the supervisor that
                        // an attached reader is a consuming reader.
                        session.watch.note_delivered(sequence);
                        yield Ok::<Event, std::convert::Infallible>(
                            Event::default().event("workspace").id(sequence.to_string()).data(json)
                        );
                    }
                    Err(_) => continue,
                }
            }
            replaying = false;
            if settled {
                break;
            }
            // The sleep is a floor, not a poll: `tip` wakes this immediately on
            // any publish. It exists so a reader can never park forever on a
            // turn that settled without publishing anything — which a panicking
            // worker thread would produce.
            tokio::select! {
                changed = tip.changed() => { if changed.is_err() { break } }
                _ = tokio::time::sleep(SUPERVISOR_TICK) => {}
            }
        }
        // A definitive end-of-response marker. EventSource reconnects after ANY
        // close, clean or not, and cannot see the status line — so a reader that
        // attached to a turn which had already settled would otherwise reconnect
        // forever against a finished feed. The client closes on this; it carries
        // no sequence so it cannot collide with replay.
        yield Ok::<Event, std::convert::Infallible>(
            Event::default().event("workspace.closed").data("{}")
        );
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
        context_budget_tokens: session.context_window.effective_tokens,
        context_window: session.context_window,
        resident_cuda: crate::inference::resident_cuda_status(super::model_resident_cache_key(
            &session.model_id,
        )),
        allow_writes: session.allow_writes,
        approval_mode: session.approval_mode,
        allow_network: session.allow_network,
        mode: session.mode,
        semantic_retrieval: session.semantic_retriever.is_some(),
        embedding_model_id: session
            .semantic_retriever
            .as_ref()
            .map(|retriever| retriever.model_id().to_string()),
    })
    .into_response()
}

/// Browser-resilient view of the one Workspace session that currently owns the
/// runtime slot. Unlike the SSE feed, this snapshot is safe to poll and remains
/// available after a turn settles, so a remount never has to pretend that an
/// active or just-finished coding task does not exist.
pub(super) async fn current_activity(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let session = state.workspace_sessions.active.lock().await.clone();
    let Some(session) = session else {
        return Json(WorkspaceActivityEnvelope { activity: None }).into_response();
    };
    let state = session
        .state
        .lock()
        .map(|state| state.as_str())
        .unwrap_or("failed");
    let activity = match session.activity.lock() {
        Ok(activity) => activity.clone(),
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "workspace_activity_unavailable",
                "Workspace live activity is unavailable".to_string(),
                None,
            )
        }
    };
    Json(WorkspaceActivityEnvelope {
        activity: Some(WorkspaceActivityResponse {
            id: session.id.clone(),
            workspace: simplify_path(&session.workspace),
            model_id: session.model_id.clone(),
            state,
            context_window: session.context_window,
            approval_mode: session.approval_mode,
            allow_network: session.allow_network,
            mode: session.mode,
            activity,
        }),
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
            oversize_text_message("message", &text),
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
    let (model, family) = match active_tool_capable_model(&state).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    if model.id != session.model_id || model.lane.gguf_sha256 != session.model_sha256 {
        return api_error(
            StatusCode::CONFLICT,
            "workspace_model_changed",
            "resume this thread with the same model that created it".to_string(),
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
        family,
        max_steps: session.max_steps,
        max_tokens: session.max_tokens,
        context_budget_tokens: session.context_window.effective_tokens,
        temperature: session.temperature,
        mode: session.mode,
        approval_mode: session.approval_mode,
        allow_network: session.allow_network,
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
    start_turn(&session, client_message_id);
    (
        StatusCode::ACCEPTED,
        Json(WorkspaceMessageResponse {
            session_id: session.id.clone(),
            turn_index,
            state: WorkspaceSessionState::Running.as_str(),
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
    // No `was_waiting` fast path any more. Every installed turn has a live
    // forwarder (or was failed outright at install), and that forwarder is the
    // single writer of the terminal memory row via
    // `persist_aborted_turn_and_finish`. Persisting here as well would race it
    // for the same `client_message_id`.
    if let Ok(mut status) = session.state.lock() {
        *status = status.after_cancel_request();
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
            "the active exact model row has not earned tool-capable status".to_string(),
            None,
        )),
    }
}

/// Resolve the agent's total prompt + generation envelope from the active
/// model and memory available *after* that model has loaded. The native GGUF
/// context remains the hard model ceiling. Live RAM determines the ordinary
/// envelope; the exact Qwen3 4B Q8_0 Code row may instead use a 16K logical target when
/// bounded paging keeps every active request inside the validated 8K working set.
const QWEN3_4B_PAGED_CONTEXT_TARGET_TOKENS: u32 = 16_384;

fn paged_context_policy_for_row(
    row_id: Option<&str>,
    paging_config: Option<&crate::chat::context_paging::ContextPagingConfig>,
) -> (Option<u32>, Option<u32>) {
    let qwen3_4b_q8 = matches!(row_id, Some("qwen3_4b_instruct_q8_0"));
    match paging_config.filter(|config| config.enabled && qwen3_4b_q8) {
        Some(config) => (
            Some(QWEN3_4B_PAGED_CONTEXT_TARGET_TOKENS),
            Some(config.working_set_tokens()),
        ),
        None => (None, None),
    }
}

fn select_workspace_context_window(
    state: &AppState,
    model: &LoadedModel,
    generation_allowance_tokens: u32,
    paging_config: Option<&crate::chat::context_paging::ContextPagingConfig>,
) -> (ContextWindowSelection, bool) {
    let native_context_tokens = model
        .llama_config
        .as_ref()
        .map(|config| config.context_length)
        .unwrap_or(crate::chat::agent::AGENT_VALIDATED_CTX);
    let kv_bytes_per_token = model
        .llama_config
        .as_ref()
        .and_then(|config| crate::inference::conservative_host_kv_bytes_per_token(config).ok());
    let resident_capacity_tokens =
        crate::inference::resident_cuda_status(super::model_resident_cache_key(&model.id))
            .and_then(|status| u32::try_from(status.max_positions).ok());
    let filename = model
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let row_id = tool_capable_row_for_loaded_artifact(filename, &model.lane.gguf_sha256)
        .map(|(row_id, _)| row_id);
    let (paged_target_tokens, paged_working_set_tokens) =
        paged_context_policy_for_row(row_id, paging_config);

    let kv_owner_slots = if state.engine.single_kv_owner_mode() {
        1
    } else {
        u32::try_from(
            state
                .engine
                .continuous_batch_slots()
                // A Metal stream retains resident KV after publication also
                // materializes its CPU-authoritative mirror. Budget both for
                // every admitted stream, plus each retained cache clone.
                .saturating_mul(2)
                .saturating_add(super::prompt_prefix_cache_capacity()),
        )
        .unwrap_or(u32::MAX)
        .max(1)
    };
    select_context_window_for_kv_admission(ContextWindowInputs {
        native_context_tokens,
        // This is the legacy operational agent-loop envelope, not a promotion
        // of every tool-capable row's parity-qualified context ladder. In
        // particular, Qwen3-4B-Q4_K_M remains qualified only at 512/1024 and
        // is deliberately excluded from the 16K logical paging exception.
        validated_context_tokens: crate::chat::agent::AGENT_VALIDATED_CTX,
        server_context_tokens: u32::try_from(state.server_limits.max_prompt_tokens)
            .unwrap_or(u32::MAX)
            .saturating_add(
                generation_allowance_tokens.min(state.server_limits.max_generation_tokens),
            ),
        host_memory: crate::capability::live_host_memory_status(),
        kv_bytes_per_token,
        kv_owner_slots,
        resident_capacity_tokens,
        configured_max_tokens: configured_agent_context_max(),
        paged_target_tokens,
        paged_working_set_tokens,
    })
}

/// Select against the configured aggregate owner count first. If the raw
/// memory budget cannot hold the selected request's real resident working set,
/// preserve that supported context by admitting exactly one owner instead of
/// treating the 8K operational floor as aggregate-memory authority.
fn select_context_window_for_kv_admission(
    mut inputs: ContextWindowInputs,
) -> (ContextWindowSelection, bool) {
    let shared = select_context_window(inputs);
    let active_working_set = shared
        .paged_working_set_tokens
        .unwrap_or(shared.effective_tokens);
    let aggregate_shortfall = inputs.kv_owner_slots > 1
        && shared
            .memory_safe_max_tokens
            .is_some_and(|tokens| tokens < active_working_set);
    if !aggregate_shortfall {
        return (shared, false);
    }

    inputs.kv_owner_slots = 1;
    (select_context_window(inputs), true)
}

fn workspace_context_memory_shortfall(selection: &ContextWindowSelection) -> Option<(u32, u32)> {
    let required = selection
        .paged_working_set_tokens
        .unwrap_or(selection.effective_tokens);
    selection
        .memory_safe_max_tokens
        .filter(|memory_safe| *memory_safe < required)
        .map(|memory_safe| (memory_safe, required))
}

/// Name-only resolution, for listing which rows COULD serve Workspace (nothing
/// is loaded, so there are no bytes to check). Never use this to authorize a
/// loaded model — see `tool_capable_row_for_loaded_artifact`.
fn tool_capable_row_for_filename(filename: &str) -> Option<(&'static str, &'static str)> {
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

/// Tool capability for a LOADED artifact. `tool_capable` is earned per exact row
/// by a committed agent-eval receipt against specific bytes (the Ornith Q4_K_M
/// row is the live example), so a hash-pinned artifact must present its
/// certified digest before Workspace will drive it — otherwise a same-named
/// replacement inherits an agent battery it never passed.
fn tool_capable_row_for_loaded_artifact(
    filename: &str,
    gguf_sha256: &str,
) -> Option<(&'static str, &'static str)> {
    if supported_artifact_expected_sha256(filename)
        .is_some_and(|expected| !gguf_sha256.eq_ignore_ascii_case(expected))
    {
        return None;
    }
    tool_capable_row_for_filename(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context_window() -> ContextWindowSelection {
        select_context_window(ContextWindowInputs {
            native_context_tokens: 8_192,
            validated_context_tokens: 8_192,
            server_context_tokens: 131_072,
            host_memory: None,
            kv_bytes_per_token: None,
            kv_owner_slots: 1,
            resident_capacity_tokens: None,
            configured_max_tokens: None,
            paged_target_tokens: None,
            paged_working_set_tokens: None,
        })
    }

    #[test]
    fn low_memory_qwen_paging_preserves_8k_by_admitting_one_kv_owner() {
        const GIB: u64 = 1_073_741_824;
        let inputs = ContextWindowInputs {
            native_context_tokens: 40_960,
            validated_context_tokens: 8_192,
            server_context_tokens: 131_072,
            host_memory: Some(crate::capability::HostMemoryStatus {
                total_bytes: 16 * GIB,
                available_bytes: 52 * GIB / 10,
            }),
            kv_bytes_per_token: Some(294_912),
            // Two cooperative sessions can each retain resident + mirrored
            // CPU KV, alongside one retained prefix clone.
            kv_owner_slots: 5,
            resident_capacity_tokens: None,
            configured_max_tokens: None,
            paged_target_tokens: Some(16_384),
            paged_working_set_tokens: Some(8_000),
        };

        let configured = select_context_window(inputs);
        assert_eq!(configured.memory_safe_max_tokens, Some(2_048));
        let (selection, single_owner) = select_context_window_for_kv_admission(inputs);
        assert!(single_owner);
        assert_eq!(selection.kv_owner_slots, 1);
        assert_eq!(selection.memory_safe_max_tokens, Some(12_288));
        assert_eq!(selection.effective_tokens, 16_384);
        assert_eq!(selection.paged_working_set_tokens, Some(8_000));

        let active_kv_bytes = u64::from(selection.paged_working_set_tokens.unwrap())
            * selection.kv_bytes_per_token.unwrap();
        let memory_budget = selection.available_memory_bytes.unwrap() * 70 / 100;
        assert!(
            active_kv_bytes <= memory_budget,
            "the admitted 8K working set must fit the measured KV allowance"
        );
    }

    #[test]
    fn ample_memory_keeps_configured_kv_concurrency() {
        const GIB: u64 = 1_073_741_824;
        let inputs = ContextWindowInputs {
            native_context_tokens: 40_960,
            validated_context_tokens: 8_192,
            server_context_tokens: 131_072,
            host_memory: Some(crate::capability::HostMemoryStatus {
                total_bytes: 64 * GIB,
                available_bytes: 48 * GIB,
            }),
            kv_bytes_per_token: Some(294_912),
            kv_owner_slots: 5,
            resident_capacity_tokens: None,
            configured_max_tokens: None,
            paged_target_tokens: Some(16_384),
            paged_working_set_tokens: Some(8_000),
        };

        let (selection, single_owner) = select_context_window_for_kv_admission(inputs);
        assert!(!single_owner);
        assert_eq!(selection.kv_owner_slots, 5);
        assert_eq!(selection.effective_tokens, 16_384);
    }

    #[test]
    fn one_owner_shortfall_fails_workspace_admission_under_severe_pressure() {
        const GIB: u64 = 1_073_741_824;
        let inputs = ContextWindowInputs {
            native_context_tokens: 40_960,
            validated_context_tokens: 8_192,
            server_context_tokens: 131_072,
            host_memory: Some(crate::capability::HostMemoryStatus {
                total_bytes: 16 * GIB,
                available_bytes: GIB,
            }),
            kv_bytes_per_token: Some(294_912),
            kv_owner_slots: 5,
            resident_capacity_tokens: None,
            configured_max_tokens: None,
            paged_target_tokens: Some(16_384),
            paged_working_set_tokens: Some(8_000),
        };

        let (selection, single_owner) = select_context_window_for_kv_admission(inputs);
        assert!(single_owner);
        assert_eq!(selection.kv_owner_slots, 1);
        assert_eq!(selection.memory_safe_max_tokens, Some(2_048));
        assert_eq!(
            workspace_context_memory_shortfall(&selection),
            Some((2_048, 8_000))
        );
    }

    #[test]
    fn zero_available_memory_fails_workspace_admission_instead_of_using_fallback() {
        const GIB: u64 = 1_073_741_824;
        let inputs = ContextWindowInputs {
            native_context_tokens: 40_960,
            validated_context_tokens: 8_192,
            server_context_tokens: 131_072,
            host_memory: Some(crate::capability::HostMemoryStatus {
                total_bytes: 16 * GIB,
                available_bytes: 0,
            }),
            kv_bytes_per_token: Some(294_912),
            kv_owner_slots: 5,
            resident_capacity_tokens: None,
            configured_max_tokens: None,
            paged_target_tokens: Some(16_384),
            paged_working_set_tokens: Some(8_000),
        };

        let (selection, single_owner) = select_context_window_for_kv_admission(inputs);
        assert!(single_owner);
        assert_eq!(selection.kv_owner_slots, 1);
        assert_eq!(selection.available_memory_bytes, Some(0));
        assert_eq!(selection.memory_safe_max_tokens, Some(0));
        assert_eq!(
            workspace_context_memory_shortfall(&selection),
            Some((0, 8_000))
        );
    }

    #[test]
    fn a_written_out_task_spec_fits_the_goal_limit() {
        // The limit that rejected a real goal was 4 KiB. A spec that names the
        // module layout, the constraints, the CLI surface and the acceptance
        // criteria — the shape that actually makes an agent succeed — runs well
        // past that, and the rejection arrived only after it had been written.
        let spec = "# Goal\nBuild a small but complete Python application.\n\n\
             ## Architecture Requirements\nUse separate modules with clear \
             responsibilities. models.py defines the Task model. storage.py handles \
             persistence. queue.py contains task queue behavior. executor.py handles \
             task execution. main.py implements the CLI.\n\n"
            .repeat(24);
        assert!(
            spec.len() > 4 * 1024,
            "fixture must exceed the old cap to prove anything ({} bytes)",
            spec.len()
        );
        assert!(
            spec.len() <= MAX_GOAL_BYTES,
            "a written-out spec of {} bytes must fit the {MAX_GOAL_BYTES} byte limit",
            spec.len()
        );
    }

    #[test]
    fn an_oversize_goal_is_told_how_far_over_it_is() {
        let over = "x".repeat(MAX_GOAL_BYTES + 500);
        let message = oversize_text_message("goal", &over);
        assert!(message.contains(&over.len().to_string()), "{message}");
        assert!(
            message.contains("500"),
            "should name the overshoot: {message}"
        );
        assert_eq!(oversize_text_message("goal", ""), "goal cannot be empty");
    }

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
        // A row with no recorded digest keeps its existing filename gating.
        // Resolved dynamically: naming a specific file here rots the moment that
        // row gains a pin (it did — Qwen3-4B-Q4_K_M was the original example).
        let unpinned = curated_catalog()
            .into_iter()
            .map(|item| item.filename)
            .find(|name| {
                supported_artifact_expected_sha256(name).is_none()
                    && tool_capable_row_for_filename(name).is_some()
            })
            .expect("some tool-capable curated row still has no recorded digest");
        assert_eq!(
            tool_capable_row_for_loaded_artifact(unpinned, &"00".repeat(32)),
            tool_capable_row_for_filename(unpinned),
            "{unpinned} has no pin, so bytes must not gate it"
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
        assert!(WorkspaceSessionState::Running.blocks_model_transition());
        assert!(WorkspaceSessionState::Cancelling.blocks_model_transition());
        assert!(!WorkspaceSessionState::Idle.blocks_model_transition());
        assert!(!WorkspaceSessionState::Cancelled.blocks_model_transition());
        assert!(!WorkspaceSessionState::Failed.blocks_model_transition());
    }

    #[test]
    fn thread_id_mode_boundary_is_a_single_rule() {
        // Resuming across modes must fail closed in BOTH directions: a read-only
        // thread must not gain write tools by being reopened as Code, and a Code
        // thread must not be silently downgraded. The listings and the resume
        // gate share this rule, so a fix to one cannot leave the other behind.
        assert!(thread_id_belongs_to_mode(
            "code-1111",
            WorkspaceRunMode::Code
        ));
        assert!(!thread_id_belongs_to_mode(
            "workspace-1111",
            WorkspaceRunMode::Code
        ));
        assert!(thread_id_belongs_to_mode(
            "workspace-1111",
            WorkspaceRunMode::ReadOnly
        ));
        assert!(!thread_id_belongs_to_mode(
            "code-1111",
            WorkspaceRunMode::ReadOnly
        ));
        // Neither prefix is a prefix of the other, and an unprefixed id belongs
        // to no mode at all.
        assert!(!thread_id_belongs_to_mode("1111", WorkspaceRunMode::Code));
        assert!(!thread_id_belongs_to_mode(
            "1111",
            WorkspaceRunMode::ReadOnly
        ));
    }

    #[test]
    fn cancellation_stays_blocking_until_a_running_worker_exits() {
        let requested = WorkspaceSessionState::Running.after_cancel_request();
        assert_eq!(requested, WorkspaceSessionState::Cancelling);
        assert!(requested.blocks_model_transition());
        // A turn is Running from the moment it is installed, so a cancel request
        // always goes through Cancelling — there is no pre-Running state left that
        // could shortcut straight to Cancelled.
        assert_eq!(
            WorkspaceSessionState::Cancelled.after_cancel_request(),
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
    fn workspace_session_request_defaults_to_gated_and_offline() {
        let request: CreateWorkspaceSessionRequest = serde_json::from_value(serde_json::json!({
            "workspace": ".",
            "goal": "inspect the project",
            "mode": "code"
        }))
        .unwrap();
        assert_eq!(request.approval_mode, WorkspaceApprovalMode::ApprovalGated);
        assert!(!request.allow_network);
    }

    #[test]
    fn code_sessions_have_no_arbitrary_step_limit() {
        assert_eq!(workspace_max_steps(WorkspaceRunMode::Code, None), Ok(0));
        assert_eq!(workspace_max_steps(WorkspaceRunMode::Code, Some(20)), Ok(0));
        assert_eq!(
            workspace_max_steps(WorkspaceRunMode::ReadOnly, None),
            Ok(DEFAULT_MAX_STEPS)
        );
        assert!(workspace_max_steps(WorkspaceRunMode::ReadOnly, Some(MAX_STEPS + 1)).is_err());
    }

    #[test]
    fn live_activity_tracks_parent_tools_children_and_terminal_reason() {
        let mut activity = WorkspaceActivitySnapshot::new("build a graphical game");
        activity.apply(&WorkspaceEvent::Started {
            workspace: "C:/work".into(),
            model_id: "model".into(),
        });
        activity.apply(&WorkspaceEvent::ToolCall {
            detail: "write_file(game.py, 1200 bytes)".into(),
        });
        assert_eq!(activity.phase, "running");
        assert_eq!(activity.stage, "tool");
        assert_eq!(
            activity.current_tool.as_deref(),
            Some("write_file(game.py, 1200 bytes)")
        );

        activity.apply(&WorkspaceEvent::AgentUpdated {
            agent_id: "child-runtime-id".into(),
            parent_id: Some("main".into()),
            label: "game-logic".into(),
            status: "running".into(),
            task: "implement the computer player".into(),
            detail: "Delegated agent is working".into(),
        });
        assert_eq!(activity.agents.len(), 2);
        assert_eq!(activity.agents[1].task, "implement the computer player");

        activity.apply(&WorkspaceEvent::ModelTiming {
            total_ms: 1_250,
            ttft_ms: Some(980),
            output_tokens: Some(42),
            prefill_ms: Some(900),
            server_first_content_ms: Some(980),
            decode_ms: Some(270),
            prompt_cache_hit: Some(true),
            reused_tokens: Some(1_920),
            prefilled_tokens: Some(31),
            prompt_cache_decision: Some("block_prefix_hit".into()),
            common_prefix_tokens: Some(1_920),
            divergent_suffix_tokens: Some(31),
            candidate_tokens: Some(1_960),
            cache_block_tokens: Some(64),
            matched_cache_blocks: Some(30),
        });
        assert_eq!(activity.output_tokens, Some(42));
        assert_eq!(activity.total_model_ms, Some(1_250));
        assert_eq!(activity.ttft_ms, Some(980));
        assert_eq!(activity.prefill_ms, Some(900));
        assert_eq!(activity.prompt_cache_hit, Some(true));
        assert_eq!(activity.reused_tokens, Some(1_920));
        assert_eq!(activity.prefilled_tokens, Some(31));
        assert_eq!(
            activity.prompt_cache_decision.as_deref(),
            Some("block_prefix_hit")
        );
        assert_eq!(activity.common_prefix_tokens, Some(1_920));
        assert!(activity.detail.contains("prompt-cache hit"));

        activity.apply(&WorkspaceEvent::Finished {
            outcome: "repeated",
        });
        assert_eq!(activity.phase, "repeated");
        assert_eq!(activity.terminal_outcome.as_deref(), Some("repeated"));
        assert!(activity.detail.contains("repeated actions"));
        assert_eq!(activity.agents[0].status, "stopped");
        assert_eq!(activity.agents[1].status, "stopped");
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
                model_sha256: "sha-test".to_string(),
                context_window: test_context_window(),
                max_steps: 1,
                max_tokens: 1,
                temperature: 0.0,
                allow_writes: true,
                approval_mode: WorkspaceApprovalMode::ApprovalGated,
                allow_network: false,
                mode: WorkspaceRunMode::ReadOnly,
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
                feed: SessionFeed::default(),
                watch: TurnWatch::default(),
                activity: StdMutex::new(WorkspaceActivitySnapshot::new("test task")),
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
            model_sha256: "sha-test".to_string(),
            context_window: test_context_window(),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            allow_writes: true,
            approval_mode: WorkspaceApprovalMode::ApprovalGated,
            allow_network: false,
            mode: WorkspaceRunMode::ReadOnly,
            semantic_retriever: None,
            memory,
            state: StdMutex::new(WorkspaceSessionState::Idle),
            events: StdMutex::new(None),
            worker: StdMutex::new(Some(initial_worker)),
            run_config: StdMutex::new(None),
            control: StdMutex::new(Some(initial_control)),
            current_turn: StdMutex::new(None),
            feed: SessionFeed::default(),
            watch: TurnWatch::default(),
            activity: StdMutex::new(WorkspaceActivitySnapshot::new("test task")),
        };
        let config = WorkspaceRunConfig {
            addr: "127.0.0.1:8181".parse().unwrap(),
            workspace: dir.path().to_path_buf(),
            goal: "question".into(),
            client_message_id: "message-1".into(),
            turn_index: 3,
            memory: Default::default(),
            model_id: "model".into(),
            family: "qwen3".into(),
            max_steps: 1,
            max_tokens: 1,
            context_budget_tokens: test_context_window().effective_tokens,
            temperature: 0.0,
            mode: WorkspaceRunMode::ReadOnly,
            approval_mode: WorkspaceApprovalMode::ApprovalGated,
            allow_network: false,
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
            WorkspaceSessionState::Running
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
            model_sha256: "sha-test".to_string(),
            context_window: test_context_window(),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            allow_writes: false,
            approval_mode: WorkspaceApprovalMode::ApprovalGated,
            allow_network: false,
            mode: WorkspaceRunMode::ReadOnly,
            semantic_retriever: None,
            memory,
            state: StdMutex::new(WorkspaceSessionState::Cancelled),
            events: StdMutex::new(None),
            worker: StdMutex::new(None),
            run_config: StdMutex::new(None),
            control: StdMutex::new(None),
            current_turn: StdMutex::new(Some(("message-1".into(), 0))),
            feed: SessionFeed::default(),
            watch: TurnWatch::default(),
            activity: StdMutex::new(WorkspaceActivitySnapshot::new("test task")),
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
            model_sha256: "sha-test".to_string(),
            context_window: test_context_window(),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            allow_writes: false,
            approval_mode: WorkspaceApprovalMode::ApprovalGated,
            allow_network: false,
            mode: WorkspaceRunMode::ReadOnly,
            semantic_retriever: None,
            memory,
            state: StdMutex::new(WorkspaceSessionState::Cancelling),
            events: StdMutex::new(None),
            worker: StdMutex::new(None),
            run_config: StdMutex::new(None),
            control: StdMutex::new(None),
            current_turn: StdMutex::new(Some(("message-1".into(), 0))),
            feed: SessionFeed::default(),
            watch: TurnWatch::default(),
            activity: StdMutex::new(WorkspaceActivitySnapshot::new("test task")),
        };
        let run_config = WorkspaceRunConfig {
            addr: "127.0.0.1:8181".parse().unwrap(),
            workspace: dir.path().to_path_buf(),
            goal: "question".into(),
            client_message_id: "message-1".into(),
            turn_index: 0,
            memory: Default::default(),
            model_id: "model".into(),
            family: "qwen3".into(),
            max_steps: 1,
            max_tokens: 1,
            context_budget_tokens: test_context_window().effective_tokens,
            temperature: 0.0,
            mode: WorkspaceRunMode::ReadOnly,
            approval_mode: WorkspaceApprovalMode::ApprovalGated,
            allow_network: false,
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
    fn worker_return_waits_for_queued_terminal_events_to_be_forwarded() {
        let dir = tempfile::tempdir().unwrap();
        let missing_workspace = dir.path().join("missing-workspace");
        let memory = WorkspaceMemoryStore::open(dir.path().join("memory.sqlite3")).unwrap();
        memory.create_thread("thread", "root", "model").unwrap();
        // The immediate Sandbox error queues Error, the fallback ModelAnswer,
        // and Finished. Capacity four lets the worker return before anything
        // drains, deterministically reproducing the publication race.
        let (worker, client) = bridge(4);
        let (events, control) = client.into_parts();
        let session = Arc::new(ActiveWorkspaceSession {
            id: "thread".into(),
            workspace: dir.path().to_path_buf(),
            model_id: "model".into(),
            model_sha256: "sha-test".to_string(),
            context_window: test_context_window(),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            allow_writes: false,
            approval_mode: WorkspaceApprovalMode::ApprovalGated,
            allow_network: false,
            mode: WorkspaceRunMode::ReadOnly,
            semantic_retriever: None,
            memory,
            state: StdMutex::new(WorkspaceSessionState::Running),
            events: StdMutex::new(None),
            worker: StdMutex::new(None),
            run_config: StdMutex::new(None),
            control: StdMutex::new(Some(control.clone())),
            current_turn: StdMutex::new(Some(("message-1".into(), 0))),
            feed: SessionFeed::default(),
            watch: TurnWatch::default(),
            activity: StdMutex::new(WorkspaceActivitySnapshot::new("test task")),
        });
        let run_config = WorkspaceRunConfig {
            addr: "127.0.0.1:8181".parse().unwrap(),
            workspace: missing_workspace,
            goal: "question".into(),
            client_message_id: "message-1".into(),
            turn_index: 0,
            memory: Default::default(),
            model_id: "model".into(),
            family: "qwen3".into(),
            max_steps: 1,
            max_tokens: 1,
            context_budget_tokens: test_context_window().effective_tokens,
            temperature: 0.0,
            mode: WorkspaceRunMode::ReadOnly,
            approval_mode: WorkspaceApprovalMode::ApprovalGated,
            allow_network: false,
            semantic_retriever: None,
        };

        run_workspace_agent(run_config.clone(), worker);

        // Returning from run_live is not publication: the bridge still owns
        // all three terminal events, so the turn must remain visible as active.
        assert_eq!(
            session.state.lock().map(|state| *state).unwrap(),
            WorkspaceSessionState::Running
        );
        assert_eq!(session.pending_message("message-1"), Some(0));
        assert!(session
            .memory
            .turn_by_client_message("thread", "message-1")
            .unwrap()
            .is_none());

        forward_workspace_events(Arc::clone(&session), events, run_config, control);

        assert_eq!(
            session.state.lock().map(|state| *state).unwrap(),
            WorkspaceSessionState::Failed
        );
        assert_eq!(session.pending_message("message-1"), None);
        let turn = session
            .memory
            .turn_by_client_message("thread", "message-1")
            .unwrap()
            .unwrap();
        assert_eq!(turn.terminal_outcome, "driver_error");
        assert!(turn.assistant_text.contains("model/runtime error"));

        let (published, complete) = session.feed.entries.lock().unwrap().since(0);
        assert!(complete);
        assert_eq!(published.len(), 3);
        assert!(matches!(&published[0].1, WorkspaceEvent::Error { .. }));
        assert!(matches!(
            &published[1].1,
            WorkspaceEvent::ModelAnswer { .. }
        ));
        assert!(matches!(
            &published[2].1,
            WorkspaceEvent::Finished {
                outcome: "driver_error"
            }
        ));
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
                model_sha256: "sha-test".to_string(),
                context_window: test_context_window(),
                max_steps: 1,
                max_tokens: 1,
                temperature: 0.0,
                allow_writes: false,
                approval_mode: WorkspaceApprovalMode::ApprovalGated,
                allow_network: false,
                mode: WorkspaceRunMode::ReadOnly,
                semantic_retriever: None,
                memory,
                state: StdMutex::new(terminal_state),
                events: StdMutex::new(Some(stale_events)),
                worker: StdMutex::new(Some(stale_worker)),
                run_config: StdMutex::new(None),
                control: StdMutex::new(Some(stale_control)),
                current_turn: StdMutex::new(Some(("old-message".into(), 0))),
                feed: SessionFeed::default(),
                watch: TurnWatch::default(),
                activity: StdMutex::new(WorkspaceActivitySnapshot::new("test task")),
            };
            let config = WorkspaceRunConfig {
                addr: "127.0.0.1:8181".parse().unwrap(),
                workspace: dir.path().to_path_buf(),
                goal: "follow up".into(),
                client_message_id: "new-message".into(),
                turn_index: 1,
                memory: Default::default(),
                model_id: "model".into(),
                family: "qwen3".into(),
                max_steps: 1,
                max_tokens: 1,
                context_budget_tokens: test_context_window().effective_tokens,
                temperature: 0.0,
                mode: WorkspaceRunMode::ReadOnly,
                approval_mode: WorkspaceApprovalMode::ApprovalGated,
                allow_network: false,
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
                WorkspaceSessionState::Running
            );
        }
    }

    /// A session carrying nothing but the state the reaper actually reads.
    fn reaper_session(dir: &std::path::Path) -> ActiveWorkspaceSession {
        let memory = WorkspaceMemoryStore::open(dir.join("memory.sqlite3")).unwrap();
        let (worker, client) = bridge(1);
        let (_events, control) = client.into_parts();
        ActiveWorkspaceSession {
            id: "thread".into(),
            workspace: dir.to_path_buf(),
            model_id: "model".into(),
            model_sha256: "sha-test".to_string(),
            context_window: test_context_window(),
            max_steps: 1,
            max_tokens: 1,
            temperature: 0.0,
            allow_writes: false,
            approval_mode: WorkspaceApprovalMode::ApprovalGated,
            allow_network: false,
            mode: WorkspaceRunMode::ReadOnly,
            semantic_retriever: None,
            memory,
            state: StdMutex::new(WorkspaceSessionState::Running),
            events: StdMutex::new(None),
            worker: StdMutex::new(Some(worker)),
            run_config: StdMutex::new(None),
            control: StdMutex::new(Some(control)),
            current_turn: StdMutex::new(Some(("message-1".into(), 0))),
            feed: SessionFeed::default(),
            watch: TurnWatch::default(),
            activity: StdMutex::new(WorkspaceActivitySnapshot::new("test task")),
        }
    }

    /// A `now` far enough past the stamps `begin()` wrote that any deadline can
    /// be expressed as an offset from it.
    const TEST_NOW: u64 = 24 * 60 * 60 * 1_000;

    fn ms(d: std::time::Duration) -> u64 {
        d.as_millis() as u64
    }

    #[test]
    fn a_never_watched_turn_is_reaped_at_the_first_attach_deadline() {
        // Replaces `unclaimed_turn_expiry_persists_and_unblocks_the_session`.
        // Nothing "claims" a turn any more — it starts at install — so the case
        // that test covered is now "a turn no browser ever attached to", decided
        // by the supervisor's first-attach deadline rather than a one-shot expiry.
        let dir = tempfile::tempdir().unwrap();
        let session = reaper_session(dir.path());
        session.watch.begin();
        session
            .watch
            .turn_started
            .store(TEST_NOW, Ordering::Release);
        assert!(
            session.abandonment_reason_at(TEST_NOW).is_none(),
            "just started"
        );
        assert!(
            session
                .abandonment_reason_at(TEST_NOW + ms(FIRST_ATTACH_TIMEOUT) / 2)
                .is_none(),
            "still inside the first-attach window"
        );
        assert!(
            session
                .abandonment_reason_at(TEST_NOW + ms(FIRST_ATTACH_TIMEOUT) + 1)
                .is_some(),
            "a turn no stream ever attached to must be reaped"
        );
    }

    #[test]
    fn a_watched_turn_is_not_reaped_but_an_abandoned_one_is() {
        let dir = tempfile::tempdir().unwrap();
        let session = reaper_session(dir.path());
        session.watch.begin();
        session.watch.ever_observed.store(true, Ordering::Release);
        session.watch.observers.store(1, Ordering::Release);
        session.watch.note_delivered(10);
        session
            .watch
            .turn_started
            .store(TEST_NOW, Ordering::Release);
        assert!(
            session.abandonment_reason_at(TEST_NOW).is_none(),
            "a reader that is keeping up is watching"
        );

        session.watch.observers.store(0, Ordering::Release);
        session
            .watch
            .unobserved_since
            .store(TEST_NOW, Ordering::Release);
        assert!(
            session.abandonment_reason_at(TEST_NOW).is_none(),
            "a reader that just left still has its grace"
        );
        assert!(
            session
                .abandonment_reason_at(TEST_NOW + ms(ABANDON_GRACE) + 1)
                .is_some(),
            "a turn nobody came back for must be reaped"
        );
    }

    #[test]
    fn a_reader_that_stops_consuming_counts_as_abandoned() {
        // The regression guard for the deleted `try_send`-on-`Full` bound. A
        // half-open TCP peer keeps the refcount at one forever, so liveness has
        // to be what the reader actually drained, not that it exists.
        let dir = tempfile::tempdir().unwrap();
        let session = reaper_session(dir.path());
        session.watch.begin();
        session.watch.ever_observed.store(true, Ordering::Release);
        session.watch.observers.store(1, Ordering::Release);
        session.watch.note_delivered(10);
        if let Ok(mut entries) = session.feed.entries.lock() {
            entries.last = 500;
        }
        session
            .watch
            .turn_started
            .store(TEST_NOW, Ordering::Release);
        session
            .watch
            .delivered_at
            .store(TEST_NOW, Ordering::Release);
        assert!(
            session.abandonment_reason_at(TEST_NOW).is_none(),
            "a reader behind but inside the grace is still watching"
        );
        assert!(
            session
                .abandonment_reason_at(TEST_NOW + ms(ABANDON_GRACE) + 1)
                .is_some(),
            "an attached reader that stopped draining is not watching"
        );
    }

    #[test]
    fn the_wall_clock_ceiling_fires_even_with_a_live_reader() {
        let dir = tempfile::tempdir().unwrap();
        let session = reaper_session(dir.path());
        session.watch.begin();
        session.watch.ever_observed.store(true, Ordering::Release);
        session.watch.observers.store(1, Ordering::Release);
        session.watch.note_delivered(10);
        session
            .watch
            .turn_started
            .store(TEST_NOW, Ordering::Release);
        assert!(
            session
                .abandonment_reason_at(TEST_NOW + ms(TURN_WALL_CLOCK_CEILING) + 1)
                .is_some(),
            "the ceiling is the bound on a turn nobody stops"
        );
    }

    #[test]
    fn abandonment_reason_fails_closed_when_the_feed_is_poisoned() {
        // A reaper whose inputs can be poisoned by an unrelated panic is a reaper
        // that stops reaping. An unreadable feed must read as "cannot prove
        // anyone is keeping up", not as "everyone is fine".
        let dir = tempfile::tempdir().unwrap();
        let session = reaper_session(dir.path());
        session.watch.begin();
        session.watch.ever_observed.store(true, Ordering::Release);
        session.watch.observers.store(1, Ordering::Release);
        session.watch.note_delivered(10);
        session
            .watch
            .turn_started
            .store(TEST_NOW, Ordering::Release);
        session
            .watch
            .delivered_at
            .store(TEST_NOW, Ordering::Release);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = session.feed.entries.lock().unwrap();
            panic!("poison the feed");
        }));
        assert!(session.feed.entries.is_poisoned());
        assert!(
            session
                .abandonment_reason_at(TEST_NOW + ms(ABANDON_GRACE) + 1)
                .is_some(),
            "a poisoned feed must not read as a healthy reader"
        );
    }

    #[test]
    fn owns_turn_fails_closed_when_the_turn_identity_is_poisoned() {
        // Must stay true so the supervisor keeps ticking rather than exiting and
        // leaving the turn with nothing watching it at all.
        let dir = tempfile::tempdir().unwrap();
        let session = reaper_session(dir.path());
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = session.current_turn.lock().unwrap();
            panic!("poison the turn identity");
        }));
        assert!(session.current_turn.is_poisoned());
        assert!(session.owns_turn("message-1"));
    }

    #[test]
    fn every_earned_tool_capable_artifact_resolves_by_exact_filename() {
        let expected = [
            ("ornith-1.0-9b-Q4_K_M.gguf", "ornith_1_0_9b_q4_k_m"),
            ("ornith-1.0-9b-Q8_0.gguf", "Ornith 1.0 9B"),
            (
                "Llama-3.2-3B-Instruct-Q8_0.gguf",
                "llama32_3b_instruct_q8_0",
            ),
            ("Qwen3-4B-Q8_0.gguf", "qwen3_4b_instruct_q8_0"),
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
    }

    #[test]
    fn exact_qwen3_4b_paging_policy_gives_only_q8_the_16k_target() {
        let config = crate::chat::context_paging::ContextPagingConfig::default();
        assert_eq!(
            paged_context_policy_for_row(Some("qwen3_4b_instruct_q8_0"), Some(&config)),
            (Some(16_384), Some(8_000))
        );
        assert_eq!(
            paged_context_policy_for_row(Some("qwen3_4b_q4_k_m"), Some(&config)),
            (None, None),
            "Q4_K_M's legacy 8K operational envelope is not a promoted context bucket"
        );
        assert_eq!(
            paged_context_policy_for_row(Some("llama32_3b_instruct_q8_0"), Some(&config)),
            (None, None)
        );

        let mut disabled = config.clone();
        disabled.enabled = false;
        assert_eq!(
            paged_context_policy_for_row(Some("qwen3_4b_instruct_q8_0"), Some(&disabled)),
            (None, None)
        );

        let q4_operational = select_context_window(ContextWindowInputs {
            native_context_tokens: 40_960,
            validated_context_tokens: crate::chat::agent::AGENT_VALIDATED_CTX,
            server_context_tokens: 131_072,
            host_memory: None,
            kv_bytes_per_token: None,
            kv_owner_slots: 1,
            resident_capacity_tokens: None,
            configured_max_tokens: None,
            paged_target_tokens: paged_context_policy_for_row(
                Some("qwen3_4b_q4_k_m"),
                Some(&config),
            )
            .0,
            paged_working_set_tokens: paged_context_policy_for_row(
                Some("qwen3_4b_q4_k_m"),
                Some(&config),
            )
            .1,
        });
        assert_eq!(q4_operational.effective_tokens, 8_192);
        assert_eq!(q4_operational.paged_target_tokens, None);
    }
}
