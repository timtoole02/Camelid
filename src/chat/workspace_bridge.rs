//! Synchronous bridge between the UI-agnostic agent loop and an external
//! controller such as the Web Workspace API.
//!
//! The agent loop remains the sole tool-execution owner. This module only
//! transports rendered events and approval decisions over bounded channels.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{net::SocketAddr, path::PathBuf};

use serde::{Deserialize, Serialize};

use super::agent::{
    run_loop, AgentConfig, AgentMsg, Approver, ContextBudgetUsage, Decision, LiveDriver, LoopEnd,
    ModelStepMetrics, Reporter,
};
use super::audit::NoopSink;
use super::client::Client;
use super::shell_sandbox::ShellSandbox;
use super::tools::{Action, Sandbox, ToolOutcome, ToolProfile};
use super::workspace_memory::MemoryContext;

/// Wall-clock ceiling for one `run_shell` in web Code mode.
///
/// This is the constant most exposed to hardware variance: the SAME `cargo
/// test` or `npm install` takes several times longer on a low-end laptop than
/// on a dev box, so a wall-clock budget tuned against one machine silently
/// becomes a build killer on another. The previous 30s could not fit an
/// ordinary Rust or Node build even here — and a killed build does not read as
/// "too slow", it reads to the model as a FAILING command, which sends it off
/// fixing a defect that does not exist.
///
/// Kept finite because a hung command must not own the turn forever, and Stop
/// stays live throughout. Teardown at the deadline kills the whole process tree
/// on Windows (the job object; see `run_shell_timeout_tears_down_the_process_tree`)
/// and, for delegated work, on Unix too (the worker's process group). A command
/// the server itself runs on Unix is killed as a single process, so a descendant
/// build tree can outlive the deadline — `run_shell` in `tools.rs` records why.
const WEB_CODE_SHELL_TIMEOUT: Duration = Duration::from_secs(120);
/// Absolute wall-clock deadline for one model step in the read-only Workspace
/// lane: it starts at the first prompt preflight and covers every fitting
/// retry, the wait for generation headers, and the SSE body. Documented in
/// WORKSPACE_MEMORY_SPEC.md as a fail-closed guarantee of that surface.
const WORKSPACE_MODEL_STEP_TIMEOUT: Duration = Duration::from_secs(90);
const APPROVAL_POLL: Duration = Duration::from_millis(25);
const DEFAULT_APPROVAL_TIMEOUT: Duration = Duration::from_secs(5 * 60);
#[cfg(test)]
pub(crate) const WORKSPACE_CONTEXT_BUDGET_TOKENS: u32 = 4_096;
#[cfg(test)]
pub(crate) const CODE_CONTEXT_BUDGET_TOKENS: u32 = super::agent::AGENT_VALIDATED_CTX;

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkspaceRunMode {
    #[default]
    ReadOnly,
    Code,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkspaceApprovalMode {
    #[default]
    ApprovalGated,
    FullAuto,
}

impl WorkspaceApprovalMode {
    pub(crate) fn is_full_auto(self) -> bool {
        self == Self::FullAuto
    }
}

impl WorkspaceRunMode {
    pub(crate) fn is_code(self) -> bool {
        self == Self::Code
    }

    fn tool_profile(self) -> ToolProfile {
        match self {
            Self::ReadOnly => ToolProfile::WorkspaceReadOnly,
            Self::Code => ToolProfile::WebCode,
        }
    }

    fn shell_sandbox(self) -> ShellSandbox {
        match self {
            Self::ReadOnly => ShellSandbox::Disabled,
            Self::Code => ShellSandbox::Sandboxed,
        }
    }

    #[cfg(test)]
    pub(crate) fn context_budget_tokens(self) -> u32 {
        match self {
            Self::ReadOnly => WORKSPACE_CONTEXT_BUDGET_TOKENS,
            Self::Code => CODE_CONTEXT_BUDGET_TOKENS,
        }
    }
}

/// Keep obvious standalone-file creation on the parent model. Delegating these
/// tasks adds two extra inference turns (spawn + wait), competes for the same
/// resident engine, and on slow local hardware can take longer than doing the
/// work. Complex repository work retains the complete orchestration surface.
fn direct_creation_request(goal: &str) -> bool {
    let goal = goal.to_ascii_lowercase();
    let asks_to_create = [
        "code me",
        "create",
        "write me",
        "make me",
        "build me",
        "implement",
    ]
    .iter()
    .any(|phrase| goal.contains(phrase));
    let names_standalone_technology = [
        "python",
        "tkinter",
        "pygame",
        "single-file",
        "single file",
        "one-file",
        "one file",
    ]
    .iter()
    .any(|phrase| goal.contains(phrase));
    let names_complex_scope = [
        "repository",
        " repo",
        "project",
        "frontend",
        "backend",
        "multiple files",
        "refactor",
        "migrate",
        "audit",
        "investigate",
        "deep dive",
        "subagent",
        "agent runtime",
    ]
    .iter()
    .any(|phrase| goal.contains(phrase));
    asks_to_create && names_standalone_technology && !names_complex_scope
}

/// Choose a stable artifact name only for the direct standalone route. This is
/// not used for repository work, and the resulting relative path still goes
/// through the workspace sandbox before any write executes.
fn direct_creation_path(goal: &str) -> Option<String> {
    if !direct_creation_request(goal) {
        return None;
    }
    let lower = goal.to_ascii_lowercase();
    if lower.contains("tic tac toe") || lower.contains("tic-tac-toe") {
        return Some("tic_tac_toe.py".into());
    }
    if lower.contains("python") || lower.contains("tkinter") || lower.contains("pygame") {
        return Some("app.py".into());
    }
    None
}

/// Turn explicit standalone-artifact wording into a compact acceptance contract
/// for small local models. This does not invent features: each clause is gated
/// by words the user actually supplied, and exists to keep those requirements
/// from disappearing between a plan and the generated source.
fn direct_creation_contract(goal: &str) -> String {
    let lower = goal.to_ascii_lowercase();
    let mut requirements = vec![
        "Create the requested runnable artifact in the workspace with write_file; do not spend a turn on update_plan or delegation."
            .to_string(),
        "For every correction, replace the complete standalone artifact with write_file; do not attempt a narrow edit_file patch."
            .to_string(),
        "Preserve and implement every explicit requirement in the user's wording. A comment, filename, label, or completion claim is not implementation."
            .to_string(),
    ];
    if lower.contains("python") {
        requirements.push(
            "The delivered artifact must be runnable Python source in a .py file.".to_string(),
        );
    }
    if lower.contains("graphics") || lower.contains("graphical") || lower.contains("gui") {
        requirements.push(
            "Graphics means a real interactive GUI window (for example tkinter or pygame), not terminal input/output."
                .to_string(),
        );
    }
    if lower.contains("python")
        && (lower.contains("graphics") || lower.contains("graphical") || lower.contains("gui"))
    {
        requirements.push(
            "Prefer tkinter for a dependency-free Python GUI. Tkinter ships with the Python standard library: never try to install it with pip."
                .to_string(),
        );
    }
    if (lower.contains("computer")
        || lower.contains("one-player")
        || lower.contains("one player")
        || lower.contains("single-player")
        || lower.contains("single player"))
        && (lower.contains("player") || lower.contains("opponent") || lower.contains(" vs "))
    {
        requirements.push(
            "A human-vs-computer game means the human controls exactly one side and the program automatically chooses and performs every opposing move."
                .to_string(),
        );
    }
    if lower.contains("tic tac toe") || lower.contains("tic-tac-toe") {
        requirements.push(
            "Tic-tac-toe turn handling must keep the human as X: after each valid human click, check the human terminal state, automatically make exactly one legal O move when play continues, check the computer terminal state/draw, and return control to X. Occupied cells and clicks after game-over must do nothing."
                .to_string(),
        );
        requirements.push(
            "For tkinter board buttons created in loops, bind row and column in each callback using lambda defaults such as row=i, col=j; a bare lambda that closes over i/j makes every button target the final cell."
                .to_string(),
        );
        requirements.push(
            "Choose O only from the current list of empty cells, track a game_over state, detect all eight winning lines and a full-board draw after each side, show the result in the GUI with a status label or messagebox, and provide an in-window reset/new-game control."
                .to_string(),
        );
    }
    if lower.contains("play") || lower.contains("game") {
        requirements.push(
            "The interaction must be complete enough for the user to start, play through, and see the win/draw state without editing source."
                .to_string(),
        );
    }
    requirements.push(
        "After writing, re-read the final file, compare its behavior to every clause above, fix omissions, and run an available syntax/build check before answering."
            .to_string(),
    );
    format!(
        "\n\nDirect creation acceptance contract:\n- {}",
        requirements.join("\n- ")
    )
}

fn restrict_direct_creation_tools(
    specs: &mut Vec<super::tools::ToolSpec>,
    context_paging_enabled: bool,
) {
    specs.retain(|spec| {
        spec.name != "update_plan" && (context_paging_enabled || spec.name != "edit_file")
    });
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub(crate) enum WorkspaceEvent {
    #[serde(rename = "session.started")]
    Started { workspace: String, model_id: String },
    #[serde(rename = "turn.started")]
    TurnStarted { turn_index: u32 },
    #[serde(rename = "memory.updated")]
    MemoryUpdated {
        prompt_tokens: u32,
        generation_tokens: u32,
        budget_total: u32,
        system_tokens_estimate: u32,
        tool_definition_tokens_estimate: u32,
        message_tokens_estimate: u32,
        recent_memory_tokens_estimate: u32,
        retrieved_memory_tokens_estimate: u32,
        evidence_memory_tokens_estimate: u32,
        tool_result_tokens_estimate: u32,
    },
    #[serde(rename = "memory.compacted")]
    MemoryCompacted {
        compacted_through_turn: Option<u32>,
        archived_turns: u32,
        compaction_count: u32,
        trigger_tokens: u32,
        budget_total: u32,
    },
    #[serde(rename = "model.delta")]
    ModelDelta { content: String },
    #[serde(rename = "model.timing")]
    ModelTiming {
        total_ms: u64,
        ttft_ms: Option<u64>,
        output_tokens: Option<u32>,
        prefill_ms: Option<u64>,
        server_first_content_ms: Option<u64>,
        decode_ms: Option<u64>,
        prompt_cache_hit: Option<bool>,
        reused_tokens: Option<u32>,
        prefilled_tokens: Option<u32>,
    },
    #[serde(rename = "model.answer")]
    ModelAnswer { content: String },
    #[serde(rename = "tool.call")]
    ToolCall { detail: String },
    #[serde(rename = "approval.required")]
    ApprovalRequired {
        approval_id: String,
        tool: String,
        risk: String,
        detail: String,
    },
    #[serde(rename = "tool.result")]
    ToolResult {
        tool: String,
        outcome: &'static str,
        content: String,
    },
    #[serde(rename = "agent.updated")]
    AgentUpdated {
        agent_id: String,
        parent_id: Option<String>,
        label: String,
        status: String,
        task: String,
        detail: String,
    },
    #[serde(rename = "session.notice")]
    Notice { content: String },
    #[serde(rename = "session.finished")]
    Finished { outcome: &'static str },
    #[serde(rename = "session.error")]
    Error { message: String },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkspaceDecisionKind {
    AllowOnce,
    AlwaysTool,
    Deny,
    Abort,
}

#[derive(Debug)]
pub(crate) struct WorkspaceDecision {
    pub approval_id: String,
    pub decision: WorkspaceDecisionKind,
}

pub(crate) struct WorkspaceBridgeWorker {
    pub reporter: WorkspaceReporter,
    pub approver: WorkspaceApprover,
    pub cancel: Arc<AtomicBool>,
    pub delivery_failed: Arc<AtomicBool>,
}

pub(crate) struct WorkspaceBridgeClient {
    pub events: Receiver<WorkspaceEvent>,
    decisions: SyncSender<WorkspaceDecision>,
    cancel: Arc<AtomicBool>,
    pending_approval: Arc<Mutex<Option<String>>>,
}

#[derive(Clone)]
pub(crate) struct WorkspaceRunConfig {
    pub addr: SocketAddr,
    pub workspace: PathBuf,
    pub goal: String,
    pub client_message_id: String,
    pub turn_index: u32,
    pub memory: MemoryContext,
    pub model_id: String,
    pub family: String,
    pub max_steps: usize,
    pub max_tokens: u32,
    /// Total prompt + generation envelope selected from the active model and
    /// live machine memory when the session is created. Follow-up turns and
    /// child agents inherit the same frozen value.
    pub context_budget_tokens: u32,
    pub temperature: f32,
    pub mode: WorkspaceRunMode,
    pub approval_mode: WorkspaceApprovalMode,
    pub allow_network: bool,
    /// Optional session-scoped semantic index. When present, each turn gets a
    /// bounded set of relevant workspace excerpts before the model runs.
    pub semantic_retriever: Option<Arc<super::semantic_search::WorkspaceSemanticRetriever>>,
}

/// Makes delegated work share the exact lifetime of one Web Code turn, including
/// early returns and unwinding paths.
struct WorkspaceSubagentTurnGuard;

impl Drop for WorkspaceSubagentTurnGuard {
    fn drop(&mut self) {
        super::subagent::cancel_all();
    }
}

impl WorkspaceBridgeClient {
    #[cfg(test)]
    pub fn try_decide(
        &self,
        approval_id: String,
        decision: WorkspaceDecisionKind,
    ) -> Result<(), &'static str> {
        if self
            .pending_approval
            .lock()
            .map_err(|_| "the approval state is unavailable")?
            .as_deref()
            != Some(approval_id.as_str())
        {
            return Err("the approval is stale or does not belong to this session");
        }
        match self.decisions.try_send(WorkspaceDecision {
            approval_id,
            decision,
        }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err("a decision is already pending"),
            Err(TrySendError::Disconnected(_)) => Err("the workspace session has ended"),
        }
    }

    #[cfg(test)]
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub fn into_parts(self) -> (Receiver<WorkspaceEvent>, WorkspaceBridgeControl) {
        (
            self.events,
            WorkspaceBridgeControl {
                decisions: self.decisions,
                cancel: self.cancel,
                pending_approval: self.pending_approval,
            },
        )
    }
}

#[derive(Clone)]
pub(crate) struct WorkspaceBridgeControl {
    decisions: SyncSender<WorkspaceDecision>,
    cancel: Arc<AtomicBool>,
    pending_approval: Arc<Mutex<Option<String>>>,
}

impl WorkspaceBridgeControl {
    pub fn try_decide(
        &self,
        approval_id: String,
        decision: WorkspaceDecisionKind,
    ) -> Result<(), &'static str> {
        if self
            .pending_approval
            .lock()
            .map_err(|_| "the approval state is unavailable")?
            .as_deref()
            != Some(approval_id.as_str())
        {
            return Err("the approval is stale or does not belong to this session");
        }
        match self.decisions.try_send(WorkspaceDecision {
            approval_id,
            decision,
        }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err("a decision is already pending"),
            Err(TrySendError::Disconnected(_)) => Err("the workspace session has ended"),
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    /// The approval this turn is currently parked on, if any.
    ///
    /// Set before the request event is published and cleared on decision,
    /// timeout or cancel (`WorkspaceApprover::approve`), so it is the exact
    /// liveness test for whether a replayed approval prompt is still
    /// actionable — `try_decide` above refuses any other id.
    pub fn pending_approval_id(&self) -> Option<String> {
        self.pending_approval
            .lock()
            .ok()
            .and_then(|pending| pending.clone())
    }
}

pub(crate) fn bridge(capacity: usize) -> (WorkspaceBridgeWorker, WorkspaceBridgeClient) {
    bridge_with_timeout(capacity, DEFAULT_APPROVAL_TIMEOUT)
}

fn bridge_with_timeout(
    capacity: usize,
    approval_timeout: Duration,
) -> (WorkspaceBridgeWorker, WorkspaceBridgeClient) {
    let capacity = capacity.max(1);
    let (event_tx, event_rx) = sync_channel(capacity);
    let (decision_tx, decision_rx) = sync_channel(1);
    let cancel = Arc::new(AtomicBool::new(false));
    let delivery_failed = Arc::new(AtomicBool::new(false));
    let terminal_publication = Arc::new(Mutex::new(TerminalPublicationState::default()));
    let pending_approval = Arc::new(Mutex::new(None));
    (
        WorkspaceBridgeWorker {
            reporter: WorkspaceReporter {
                events: event_tx.clone(),
                delivery_failed: Arc::clone(&delivery_failed),
                terminal_publication,
            },
            approver: WorkspaceApprover {
                events: event_tx,
                decisions: decision_rx,
                cancel: Arc::clone(&cancel),
                delivery_failed: Arc::clone(&delivery_failed),
                pending_approval: Arc::clone(&pending_approval),
                approval_timeout,
            },
            cancel: Arc::clone(&cancel),
            delivery_failed,
        },
        WorkspaceBridgeClient {
            events: event_rx,
            decisions: decision_tx,
            cancel,
            pending_approval,
        },
    )
}

#[derive(Default)]
struct TerminalPublicationState {
    answer_emitted: bool,
    finished: bool,
}

#[derive(Clone)]
pub(crate) struct WorkspaceReporter {
    events: SyncSender<WorkspaceEvent>,
    delivery_failed: Arc<AtomicBool>,
    /// Shared by every reporter clone so a racing model answer cannot arrive
    /// after the terminal event or duplicate a deterministic fallback answer.
    terminal_publication: Arc<Mutex<TerminalPublicationState>>,
}

impl WorkspaceReporter {
    fn send(&self, event: WorkspaceEvent) {
        // A bounded blocking send provides backpressure without unbounded memory.
        // A dropped receiver ends delivery; the agent loop remains cancellable.
        if self.events.send(event).is_err() {
            self.delivery_failed.store(true, Ordering::Release);
        }
    }

    fn model_delta(&self, content: &str) {
        self.send(WorkspaceEvent::ModelDelta {
            content: content.to_string(),
        });
    }

    fn finish(&self, end: &LoopEnd) {
        let mut publication = self
            .terminal_publication
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if publication.finished {
            return;
        }
        let fallback = match end {
            LoopEnd::Repeated => Some(
                "I couldn't complete this request because the agent stopped after repeating an action without making progress.",
            ),
            LoopEnd::DriverError => Some(
                "I couldn't complete this request because the agent stopped on a model/runtime error before it could provide an answer.",
            ),
            LoopEnd::StepCapped => Some(
                "I couldn't complete this request because the agent reached its step limit before it could provide an answer.",
            ),
            LoopEnd::Answered | LoopEnd::Aborted => None,
        };
        if !publication.answer_emitted {
            if let Some(content) = fallback {
                self.send(WorkspaceEvent::ModelAnswer {
                    content: content.to_string(),
                });
                publication.answer_emitted = true;
            }
        }
        let outcome = match end {
            LoopEnd::Answered => "answered",
            LoopEnd::Aborted => "aborted",
            LoopEnd::StepCapped => "step_capped",
            LoopEnd::Repeated => "repeated",
            LoopEnd::DriverError => "driver_error",
        };
        self.send(WorkspaceEvent::Finished { outcome });
        publication.finished = true;
    }
}

impl Reporter for WorkspaceReporter {
    fn model_text(&mut self, text: &str) {
        let mut publication = self
            .terminal_publication
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if publication.finished {
            return;
        }
        self.send(WorkspaceEvent::ModelAnswer {
            content: text.to_string(),
        });
        publication.answer_emitted = true;
    }

    fn tool_call(&mut self, line: &str) {
        self.send(WorkspaceEvent::ToolCall {
            detail: line.to_string(),
        });
    }

    fn tool_result(&mut self, name: &str, outcome: &ToolOutcome) {
        self.send(WorkspaceEvent::ToolResult {
            tool: name.to_string(),
            outcome: if outcome.is_err() { "error" } else { "ok" },
            content: outcome.text().to_string(),
        });
    }

    fn notice(&mut self, text: &str) {
        self.send(WorkspaceEvent::Notice {
            content: text.to_string(),
        });
    }

    fn context_budget(&mut self, usage: ContextBudgetUsage) {
        self.send(WorkspaceEvent::MemoryUpdated {
            prompt_tokens: usage.prompt_tokens,
            generation_tokens: usage.generation_tokens,
            budget_total: usage.budget_tokens,
            system_tokens_estimate: usage.system_tokens_estimate,
            tool_definition_tokens_estimate: usage.tool_definition_tokens_estimate,
            message_tokens_estimate: usage.message_tokens_estimate,
            recent_memory_tokens_estimate: usage.recent_memory_tokens_estimate,
            retrieved_memory_tokens_estimate: usage.retrieved_memory_tokens_estimate,
            evidence_memory_tokens_estimate: usage.evidence_memory_tokens_estimate,
            tool_result_tokens_estimate: usage.tool_result_tokens_estimate,
        });
    }

    fn model_timing(&mut self, metrics: ModelStepMetrics) {
        self.send(WorkspaceEvent::ModelTiming {
            total_ms: metrics.total_ms,
            ttft_ms: metrics.ttft_ms,
            output_tokens: metrics.output_tokens,
            prefill_ms: metrics.prefill_ms,
            server_first_content_ms: metrics.server_first_content_ms,
            decode_ms: metrics.decode_ms,
            prompt_cache_hit: metrics.prompt_cache_hit,
            reused_tokens: metrics.reused_tokens,
            prefilled_tokens: metrics.prefilled_tokens,
        });
    }

    fn agent_update(
        &mut self,
        agent_id: &str,
        parent_id: Option<&str>,
        label: &str,
        status: &str,
        task: &str,
        detail: &str,
    ) {
        self.send(WorkspaceEvent::AgentUpdated {
            agent_id: agent_id.to_string(),
            parent_id: parent_id.map(str::to_string),
            label: label.to_string(),
            status: status.to_string(),
            task: task.to_string(),
            detail: detail.to_string(),
        });
    }
}

pub(crate) struct WorkspaceApprover {
    events: SyncSender<WorkspaceEvent>,
    decisions: Receiver<WorkspaceDecision>,
    cancel: Arc<AtomicBool>,
    delivery_failed: Arc<AtomicBool>,
    pending_approval: Arc<Mutex<Option<String>>>,
    approval_timeout: Duration,
}

impl WorkspaceApprover {
    fn clear_pending(&self) {
        if let Ok(mut pending) = self.pending_approval.lock() {
            *pending = None;
        }
    }
}

impl Approver for WorkspaceApprover {
    fn approve(&mut self, action: &Action, sandbox: &Sandbox) -> Decision {
        let approval_id = uuid::Uuid::new_v4().to_string();
        let Ok(mut pending) = self.pending_approval.lock() else {
            return Decision::Abort;
        };
        *pending = Some(approval_id.clone());
        drop(pending);
        let event = WorkspaceEvent::ApprovalRequired {
            approval_id: approval_id.clone(),
            tool: action.tool_name().to_string(),
            risk: action.risk().label().to_string(),
            detail: action.approval_detail(sandbox),
        };
        if self.events.send(event).is_err() {
            self.delivery_failed.store(true, Ordering::Release);
            self.clear_pending();
            return Decision::Abort;
        }

        let deadline = Instant::now() + self.approval_timeout;
        loop {
            if self.cancel.load(Ordering::Acquire) {
                self.clear_pending();
                return Decision::Abort;
            }
            if Instant::now() >= deadline {
                self.clear_pending();
                let _ = self.events.send(WorkspaceEvent::Notice {
                    content: "approval timed out; the session was aborted".to_string(),
                });
                return Decision::Abort;
            }
            match self.decisions.recv_timeout(APPROVAL_POLL) {
                Ok(decision) if decision.approval_id == approval_id => {
                    self.clear_pending();
                    return match decision.decision {
                        WorkspaceDecisionKind::AllowOnce => Decision::Once,
                        WorkspaceDecisionKind::AlwaysTool => Decision::AlwaysTool,
                        WorkspaceDecisionKind::Deny => Decision::No,
                        WorkspaceDecisionKind::Abort => Decision::Abort,
                    };
                }
                Ok(_) => continue,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    self.clear_pending();
                    return Decision::Abort;
                }
            }
        }
    }
}

pub(crate) fn run_live(
    config: WorkspaceRunConfig,
    mut worker: WorkspaceBridgeWorker,
) -> Result<LoopEnd, String> {
    let direct_creation = config.mode.is_code() && direct_creation_request(&config.goal);
    let default_write_path = direct_creation
        .then(|| direct_creation_path(&config.goal))
        .flatten();
    let _subagent_guard = if config.mode.is_code() {
        super::subagent::cancel_all();
        Some(WorkspaceSubagentTurnGuard)
    } else {
        None
    };
    // What this host can actually enforce, not what the mode asks for. On a host
    // where sandboxed confinement is unenforceable (macOS, unsupported arch) this
    // drops to `Disabled` so `run_shell` is never advertised to the model — see
    // `shell_sandbox::resolve_for_unattended_surface`.
    let (shell_sandbox, shell_unavailable) = super::shell_sandbox::resolve_for_unattended_surface(
        config.mode.shell_sandbox(),
        &config.workspace,
    );
    let tool_profile = config.mode.tool_profile();
    let sandbox = match Sandbox::new(
        &config.workspace,
        config.allow_network,
        WEB_CODE_SHELL_TIMEOUT,
    ) {
        Ok(sandbox) => sandbox.with_shell_mode(shell_sandbox),
        Err(error) => {
            let message = error.to_string();
            worker.reporter.send(WorkspaceEvent::Error {
                message: message.clone(),
            });
            worker.reporter.finish(&LoopEnd::DriverError);
            return Err(message);
        }
    };
    let mut policy = match super::agent::resolve_policy(
        false,
        config.approval_mode.is_full_auto(),
        super::agent::is_production(),
    ) {
        Ok(policy) => policy,
        Err(message) => {
            worker.reporter.send(WorkspaceEvent::Error {
                message: message.clone(),
            });
            worker.reporter.finish(&LoopEnd::DriverError);
            return Err(message);
        }
    };
    worker.reporter.send(WorkspaceEvent::AgentUpdated {
        agent_id: "main".to_string(),
        parent_id: None,
        label: "Camelid".to_string(),
        status: "running".to_string(),
        task: config.goal.clone(),
        detail: "Preparing the first model step".to_string(),
    });
    worker.reporter.send(WorkspaceEvent::Started {
        workspace: sandbox.root_display(),
        model_id: config.model_id.clone(),
    });
    // Say so when this host cannot run commands at all. The terminal lane prints
    // the same fact in its startup banner; without it a Code user on macOS just
    // sees the model never build or test anything, with no stated reason.
    if let Some(why) = shell_unavailable.as_deref() {
        worker.reporter.notice(&format!(
            "shell commands are unavailable on this host, so run_shell is not offered to the \
             model: {why}"
        ));
    }
    worker.reporter.send(WorkspaceEvent::TurnStarted {
        turn_index: config.turn_index,
    });

    // Register the subagent runtime for this turn. Without it `is_enabled()` is
    // false, `specs_for` never advertises spawn_subagent, and Code mode has no
    // way to delegate a scoped subtask — the CLI lane configured this and the
    // web lane silently did not. Children inherit this session's model, approval
    // posture, and shell sandbox; the depth limit keeps a child from spawning
    // grandchildren. Read-only Workspace stays single-agent.
    if config.mode.is_code() && !direct_creation {
        super::subagent::configure(super::subagent::SubagentConfig::for_web_code_session(
            config.addr,
            config.model_id.clone(),
            config.family.clone(),
            config.max_tokens,
            config.context_budget_tokens,
            config.approval_mode.is_full_auto(),
            config.allow_network,
            shell_sandbox,
        ));
    } else if direct_creation {
        worker.reporter.notice(
            "using direct file tools for this standalone creation; delegation is reserved for complex repository work",
        );
    }

    let context_paging_enabled = super::context_paging::ContextPagingConfig::from_env().enabled;
    let system = if config.mode.is_code() {
        let mut specs = super::tools::specs_for(tool_profile, config.allow_network, shell_sandbox);
        if direct_creation {
            restrict_direct_creation_tools(&mut specs, context_paging_enabled);
        }
        let project = super::agent::load_project_context(&sandbox);
        let mut system =
            super::agent::system_prompt_with_project(&sandbox, &specs, project.as_ref());
        if direct_creation {
            system.push_str(&direct_creation_contract(&config.goal));
        }
        system
    } else {
        super::agent::workspace_system_prompt(&sandbox)
    };
    let mut history = vec![AgentMsg::System(system)];
    if let Some(retriever) = config.semantic_retriever.as_ref() {
        worker.reporter.notice(&format!(
            "retrieving semantically relevant workspace excerpts with {}",
            retriever.model_id()
        ));
        match retriever.retrieve_context(&config.goal, 5) {
            Ok(Some(context)) => history.push(AgentMsg::Memory(context)),
            Ok(None) => {}
            Err(error) => worker
                .reporter
                .notice(&format!("semantic retrieval was unavailable: {error}")),
        }
    }
    if let Some(memory) = render_relevant_memory(&config.memory.relevant) {
        history.push(AgentMsg::Memory(memory));
    }
    if let Some(memory) = render_evidence_memory(&config.memory.evidence) {
        history.push(AgentMsg::Memory(memory));
    }
    if let Some(memory) = render_recent_memory(&config.memory.recent) {
        history.push(AgentMsg::Memory(memory));
    }
    history.push(AgentMsg::User(config.goal));
    let mut driver = LiveDriver::with(
        Client::new(config.addr),
        config.model_id,
        config.family,
        config.max_tokens,
        config.temperature,
    );
    driver.set_context_budget(Some(config.context_budget_tokens));
    driver.set_native_tool_history(true);
    // Code drops the wall-clock model-step deadline (a coding turn can sit in a
    // long prefill, and Stop stays authoritative), but the read-only lane keeps
    // it: a bounded turn that fails closed on a stalled server is its published
    // contract, and it did not ask to be changed.
    match config.mode {
        WorkspaceRunMode::Code => driver.set_stream_cancel(Arc::clone(&worker.cancel)),
        WorkspaceRunMode::ReadOnly => {
            driver.set_stream_control(Arc::clone(&worker.cancel), WORKSPACE_MODEL_STEP_TIMEOUT)
        }
    }
    let delta_reporter = worker.reporter.clone();
    driver.set_delta_sink(Some(Box::new(move |delta| {
        delta_reporter.model_delta(delta);
    })));
    let agent_config = AgentConfig {
        workdir: config.workspace,
        max_steps: config.max_steps,
        auto_approve: config.approval_mode.is_full_auto(),
        yolo: config.approval_mode.is_full_auto(),
        allow_net: config.allow_network,
        allow_fs: false,
        shell_timeout: Duration::from_secs(30),
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        audit: Box::new(NoopSink),
        shell_sandbox,
        tool_profile,
        allow_plan: !direct_creation,
        default_write_path,
        ctx_budget: None,
        context_paging: context_paging_enabled,
    };
    let end = run_loop(
        &mut driver,
        &mut worker.approver,
        &mut worker.reporter,
        &sandbox,
        &agent_config,
        worker.cancel.as_ref(),
        &mut policy,
        &mut history,
    );
    worker.reporter.finish(&end);
    Ok(end)
}

fn render_relevant_memory(relevant: &[super::workspace_memory::StoredTurn]) -> Option<String> {
    if relevant.is_empty() {
        return None;
    }
    let mut rendered = String::from("Relevant earlier conversation excerpts:\n");
    for turn in relevant {
        rendered.push_str(&format!(
            "- Earlier user: {}\n  Earlier assistant: {}\n",
            turn.user_text, turn.assistant_text
        ));
    }
    Some(rendered)
}

fn render_recent_memory(recent: &[super::workspace_memory::StoredTurn]) -> Option<String> {
    if recent.is_empty() {
        return None;
    }
    let mut rendered = String::from("Recent conversation excerpts:\n");
    for turn in recent {
        rendered.push_str(&format!(
            "- Earlier user: {}\n  Earlier assistant: {}\n",
            turn.user_text, turn.assistant_text
        ));
    }
    Some(rendered)
}

fn render_evidence_memory(evidence: &[super::workspace_memory::StoredEvidence]) -> Option<String> {
    if evidence.is_empty() {
        return None;
    }
    let mut rendered = String::from("Evidence recorded for selected earlier turns:\n");
    for entry in evidence {
        rendered.push_str(&format!(
            "- Tool: {}\n  Call: {}\n  Observation: {}\n  SHA-256: {}\n",
            entry.tool, entry.detail, entry.observation, entry.observation_sha256
        ));
    }
    Some(rendered)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Barrier};
    use std::thread;

    use serde_json::{json, Value};

    use super::*;
    use crate::chat::agent::{
        run_loop, AgentConfig, AgentMsg, LoopEnd, ModelDriver, ModelStep, Policy,
    };
    use crate::chat::audit::NoopSink;
    use crate::chat::shell_sandbox::ShellSandbox;
    use crate::chat::tools::{Action, ApprovalTier, ToolCall, ToolProfile, ToolSpec};

    /// Code mode must be able to delegate a scoped subtask. The tools existed
    /// but the web lane never configured the subagent runtime, so `is_enabled()`
    /// was false, `specs_for` never advertised them, and delegation was silently
    /// impossible on this surface while the CLI lane had it all along.
    #[test]
    fn code_mode_advertises_the_subagent_tools_once_the_runtime_is_configured() {
        // Keep the tool-set baseline deterministic for this test thread.
        let _lock = crate::chat::mcp::tests::registry_lock();
        let profile = WorkspaceRunMode::Code.tool_profile();
        assert!(profile.allows("spawn_subagent"));
        assert!(profile.allows("await_subagent"));
        assert!(profile.allows("check_subagent_status"));

        // Built exactly as `run_live` builds it. The terminal `for_session`
        // constructor leaves `web_code: false`, which hands the child
        // `ToolProfile::Full` — GUI, Windows control, MCP, none of which this
        // parent may touch — and silently drops `allow_net` and the confirmed
        // full-auto Exec posture. Asserting the resulting boundary here is what
        // catches a call site that reached for the wrong constructor.
        let config = crate::chat::subagent::SubagentConfig::for_web_code_session(
            "127.0.0.1:9".parse().unwrap(),
            "model".into(),
            "llama".into(),
            2048,
            32_768,
            true,
            true,
            WorkspaceRunMode::Code.shell_sandbox(),
        );
        assert!(
            config.web_code,
            "a child of Code mode must run on the narrow WebCode profile, not Full"
        );
        assert!(config.allow_net, "the parent's network switch must carry");
        assert!(config.yolo, "confirmed full auto must carry to the child");
        assert_eq!(config.context_budget_tokens, 32_768);
        assert_eq!(
            config.shell_mode,
            ShellSandbox::Sandboxed,
            "the child must never widen the parent's shell confinement"
        );
        let _configured = crate::chat::subagent::configure_for_test(config);
        let advertised =
            crate::chat::tools::specs_for(profile, false, WorkspaceRunMode::Code.shell_sandbox())
                .into_iter()
                .map(|tool| tool.name)
                .collect::<Vec<_>>();
        assert!(advertised.iter().any(|name| name == "spawn_subagent"));
        assert!(advertised.iter().any(|name| name == "await_subagent"));
        assert!(advertised
            .iter()
            .any(|name| name == "check_subagent_status"));

        // Read-only Workspace stays single-agent even with the runtime up.
        let read_only = WorkspaceRunMode::ReadOnly.tool_profile();
        assert!(!read_only.allows("spawn_subagent"));
    }

    struct ScriptedDriver {
        steps: Vec<ModelStep>,
        next: usize,
    }

    impl ModelDriver for ScriptedDriver {
        fn step(
            &mut self,
            _history: &[AgentMsg],
            _tools: &[ToolSpec],
        ) -> Result<ModelStep, String> {
            let step = self
                .steps
                .get(self.next)
                .ok_or_else(|| "script exhausted".to_string())?;
            self.next += 1;
            Ok(match step {
                ModelStep::Text(text) => ModelStep::Text(text.clone()),
                ModelStep::Calls(calls) => ModelStep::Calls(calls.clone()),
            })
        }
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            name: name.to_string(),
            args,
        }
    }

    fn config(root: &std::path::Path) -> AgentConfig {
        AgentConfig {
            workdir: root.to_path_buf(),
            max_steps: 4,
            auto_approve: false,
            yolo: false,
            allow_net: false,
            allow_fs: false,
            shell_timeout: Duration::from_secs(5),
            max_tokens: 64,
            temperature: 0.0,
            audit: Box::new(NoopSink),
            shell_sandbox: ShellSandbox::Disabled,
            tool_profile: ToolProfile::Full,
            allow_plan: true,
            default_write_path: None,
            ctx_budget: None,
            context_paging: false,
        }
    }

    fn run_write_loop(
        root: std::path::PathBuf,
        worker: WorkspaceBridgeWorker,
    ) -> thread::JoinHandle<LoopEnd> {
        thread::spawn(move || {
            let sandbox = Sandbox::new(&root, false, Duration::from_secs(5)).unwrap();
            let mut driver = ScriptedDriver {
                steps: vec![
                    ModelStep::Calls(vec![call(
                        "write_file",
                        json!({"path":"result.txt","content":"approved"}),
                    )]),
                    ModelStep::Text("done".to_string()),
                ],
                next: 0,
            };
            let mut reporter = worker.reporter;
            let mut approver = worker.approver;
            let mut history = vec![AgentMsg::User("write the result".to_string())];
            run_loop(
                &mut driver,
                &mut approver,
                &mut reporter,
                &sandbox,
                &config(&root),
                worker.cancel.as_ref(),
                &mut Policy::default(),
                &mut history,
            )
        })
    }

    fn next_approval(client: &WorkspaceBridgeClient) -> String {
        loop {
            match client.events.recv_timeout(Duration::from_secs(2)).unwrap() {
                WorkspaceEvent::ApprovalRequired { approval_id, .. } => return approval_id,
                _ => continue,
            }
        }
    }

    #[test]
    fn code_network_switch_exposes_real_web_tools_only_when_enabled() {
        let offline =
            crate::chat::tools::specs_for(ToolProfile::WebCode, false, ShellSandbox::Sandboxed);
        let online =
            crate::chat::tools::specs_for(ToolProfile::WebCode, true, ShellSandbox::Sandboxed);

        for name in ["web_search", "http_fetch"] {
            assert!(offline.iter().all(|tool| tool.name != name));
            assert!(online.iter().any(|tool| tool.name == name));
        }
    }

    #[test]
    fn full_auto_workspace_policy_promotes_exec_and_network_actions() {
        let policy = crate::chat::agent::resolve_policy(false, true, false).unwrap();
        assert_eq!(
            policy.tier_for(&Action::RunShell {
                command: "cargo test".to_string()
            }),
            ApprovalTier::Auto
        );
        assert_eq!(
            policy.tier_for(&Action::WebSearch {
                query: "Camelid".to_string()
            }),
            ApprovalTier::Auto
        );
    }

    #[test]
    fn workspace_access_defaults_fail_closed() {
        assert_eq!(
            WorkspaceApprovalMode::default(),
            WorkspaceApprovalMode::ApprovalGated
        );
        assert!(!WorkspaceApprovalMode::default().is_full_auto());
    }

    #[test]
    fn legacy_test_budget_distinguishes_code_from_read_only_workspace() {
        assert_eq!(
            WorkspaceRunMode::ReadOnly.context_budget_tokens(),
            WORKSPACE_CONTEXT_BUDGET_TOKENS
        );
        assert_eq!(
            WorkspaceRunMode::Code.context_budget_tokens(),
            crate::chat::agent::AGENT_VALIDATED_CTX
        );
        assert!(
            WorkspaceRunMode::Code.context_budget_tokens()
                > WorkspaceRunMode::ReadOnly.context_budget_tokens()
        );
    }

    #[test]
    fn standalone_python_creation_stays_direct_but_repo_work_can_delegate() {
        assert!(direct_creation_request(
            "Can you code me tic tac toe, one player vs the computer. In Python with graphics so I can play"
        ));
        assert!(direct_creation_request(
            "Create one file in Python that displays a desktop clock"
        ));
        assert!(!direct_creation_request(
            "Deep dive this repository and refactor the Python agent frontend"
        ));
        assert!(!direct_creation_request(
            "Investigate the backend and implement the fix across multiple files"
        ));
        assert_eq!(
            direct_creation_path(
                "Can you code me tic tac toe, one player vs the computer. In Python with graphics so I can play"
            )
            .as_deref(),
            Some("tic_tac_toe.py")
        );
        assert_eq!(
            direct_creation_path("Create a small Python GUI utility").as_deref(),
            Some("app.py")
        );
        assert!(direct_creation_path("Refactor this Python repository").is_none());
    }

    #[test]
    fn direct_game_contract_keeps_graphics_and_computer_behavior_explicit() {
        let contract = direct_creation_contract(
            "Can you code me tic tac toe, one player vs the computer. In Python with graphics so I can play",
        );
        assert!(contract.contains("runnable Python source"));
        assert!(contract.contains("real interactive GUI window"));
        assert!(contract.contains("human controls exactly one side"));
        assert!(contract.contains("automatically chooses and performs every opposing move"));
        assert!(contract.contains("keep the human as X"));
        assert!(contract.contains("exactly one legal O move"));
        assert!(contract.contains("return control to X"));
        assert!(contract.contains("lambda defaults"));
        assert!(contract.contains("all eight winning lines"));
        assert!(contract.contains("status label or messagebox"));

        let implied_opponent = direct_creation_contract(
            "Code me a one-player tic tac toe game in Python using graphics.",
        );
        assert!(implied_opponent.contains("human controls exactly one side"));
        assert!(implied_opponent.contains("automatically chooses and performs every opposing move"));
    }

    #[test]
    fn context_paged_direct_creation_keeps_narrow_edit_recovery_available() {
        let mut paged =
            crate::chat::tools::specs_for(ToolProfile::WebCode, false, ShellSandbox::Disabled);
        restrict_direct_creation_tools(&mut paged, true);
        assert!(paged.iter().any(|tool| tool.name == "write_file"));
        assert!(paged.iter().any(|tool| tool.name == "edit_file"));
        assert!(!paged.iter().any(|tool| tool.name == "update_plan"));

        let mut legacy =
            crate::chat::tools::specs_for(ToolProfile::WebCode, false, ShellSandbox::Disabled);
        restrict_direct_creation_tools(&mut legacy, false);
        assert!(legacy.iter().any(|tool| tool.name == "write_file"));
        assert!(!legacy.iter().any(|tool| tool.name == "edit_file"));
    }

    #[test]
    fn write_waits_for_matching_approval_before_execution() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge(16);
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let approval_id = next_approval(&client);
        assert!(!root.path().join("result.txt").exists());

        client
            .try_decide(approval_id, WorkspaceDecisionKind::AllowOnce)
            .unwrap();
        assert_eq!(join.join().unwrap(), LoopEnd::Answered);
        assert_eq!(
            std::fs::read_to_string(root.path().join("result.txt")).unwrap(),
            "approved"
        );
    }

    #[test]
    fn denied_write_never_executes() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge(16);
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let approval_id = next_approval(&client);
        client
            .try_decide(approval_id, WorkspaceDecisionKind::Deny)
            .unwrap();

        assert_eq!(join.join().unwrap(), LoopEnd::Answered);
        assert!(!root.path().join("result.txt").exists());
    }

    #[test]
    fn cancellation_while_approval_is_pending_aborts_without_writing() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge(16);
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let _approval_id = next_approval(&client);
        client.cancel();

        assert_eq!(join.join().unwrap(), LoopEnd::Aborted);
        assert!(!root.path().join("result.txt").exists());
    }

    #[test]
    fn read_only_calls_do_not_request_approval() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("note.txt"), "hello").unwrap();
        let (mut worker, client) = bridge(16);
        let sandbox = Sandbox::new(root.path(), false, Duration::from_secs(5)).unwrap();
        let mut driver = ScriptedDriver {
            steps: vec![
                ModelStep::Calls(vec![call("read_file", json!({"path":"note.txt"}))]),
                ModelStep::Text("done".to_string()),
            ],
            next: 0,
        };
        let mut history = vec![AgentMsg::User("read note.txt".to_string())];
        let mut read_only_config = config(root.path());
        read_only_config.tool_profile = ToolProfile::WorkspaceReadOnly;
        let end = run_loop(
            &mut driver,
            &mut worker.approver,
            &mut worker.reporter,
            &sandbox,
            &read_only_config,
            &AtomicBool::new(false),
            &mut Policy::default(),
            &mut history,
        );
        assert_eq!(end, LoopEnd::Answered);

        let events = client.events.try_iter().collect::<Vec<_>>();
        assert!(events
            .iter()
            .all(|event| !matches!(event, WorkspaceEvent::ApprovalRequired { .. })));
        assert!(events.iter().any(|event| matches!(
            event,
            WorkspaceEvent::ToolResult { tool, outcome: "ok", .. } if tool == "read_file"
        )));
    }

    #[test]
    fn workspace_profile_rejects_an_unadvertised_exec_tool() {
        let root = tempfile::tempdir().unwrap();
        let (mut worker, client) = bridge(16);
        let sandbox = Sandbox::new(root.path(), false, Duration::from_secs(5)).unwrap();
        let mut driver = ScriptedDriver {
            steps: vec![
                ModelStep::Calls(vec![call("run_shell", json!({"command":"echo unsafe"}))]),
                ModelStep::Text("stopped".to_string()),
            ],
            next: 0,
        };
        let mut history = vec![AgentMsg::User("run a command".to_string())];
        let mut read_only_config = config(root.path());
        read_only_config.tool_profile = ToolProfile::WorkspaceReadOnly;
        let end = run_loop(
            &mut driver,
            &mut worker.approver,
            &mut worker.reporter,
            &sandbox,
            &read_only_config,
            &AtomicBool::new(false),
            &mut Policy::default(),
            &mut history,
        );
        assert_eq!(end, LoopEnd::Answered);
        let events = client.events.try_iter().collect::<Vec<_>>();
        assert!(events
            .iter()
            .all(|event| !matches!(event, WorkspaceEvent::ApprovalRequired { .. })));
        assert!(events.iter().any(|event| matches!(
            event,
            WorkspaceEvent::ToolResult { outcome: "error", content, .. }
                if content.contains("not available in this agent mode")
        )));
    }

    #[test]
    fn stale_approval_id_is_rejected_before_it_reaches_the_worker() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge(16);
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let approval_id = next_approval(&client);
        assert_eq!(
            client.try_decide("not-current".to_string(), WorkspaceDecisionKind::AllowOnce),
            Err("the approval is stale or does not belong to this session")
        );
        client
            .try_decide(approval_id, WorkspaceDecisionKind::Deny)
            .unwrap();
        assert_eq!(join.join().unwrap(), LoopEnd::Answered);
        assert!(!root.path().join("result.txt").exists());
    }

    #[test]
    fn approval_timeout_aborts_without_writing() {
        let root = tempfile::tempdir().unwrap();
        let (worker, client) = bridge_with_timeout(16, Duration::from_millis(40));
        let join = run_write_loop(root.path().to_path_buf(), worker);
        let _approval_id = next_approval(&client);
        assert_eq!(join.join().unwrap(), LoopEnd::Aborted);
        assert!(!root.path().join("result.txt").exists());
    }

    #[test]
    fn terminal_failures_emit_one_answer_before_finished() {
        for end in [LoopEnd::Repeated, LoopEnd::DriverError, LoopEnd::StepCapped] {
            let (worker, client) = bridge(4);
            worker.reporter.finish(&end);
            let events = client.events.try_iter().collect::<Vec<_>>();
            assert_eq!(events.len(), 2, "unexpected terminal events for {end:?}");
            assert!(matches!(events[0], WorkspaceEvent::ModelAnswer { .. }));
            assert!(matches!(events[1], WorkspaceEvent::Finished { .. }));
        }
    }

    #[test]
    fn terminal_failure_never_duplicates_a_real_answer_from_a_reporter_clone() {
        let (worker, client) = bridge(4);
        let mut clone = worker.reporter.clone();
        clone.model_text("the real answer");
        worker.reporter.finish(&LoopEnd::Repeated);

        let events = client.events.try_iter().collect::<Vec<_>>();
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            WorkspaceEvent::ModelAnswer { content } if content == "the real answer"
        ));
        assert!(matches!(
            events[1],
            WorkspaceEvent::Finished {
                outcome: "repeated"
            }
        ));
    }

    #[test]
    fn model_answer_after_finished_is_suppressed() {
        let (worker, client) = bridge(4);
        worker.reporter.finish(&LoopEnd::Repeated);
        let mut clone = worker.reporter.clone();
        clone.model_text("too late");

        let events = client.events.try_iter().collect::<Vec<_>>();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], WorkspaceEvent::ModelAnswer { .. }));
        assert!(matches!(events[1], WorkspaceEvent::Finished { .. }));
    }

    #[test]
    fn racing_model_answer_and_finish_publish_one_ordered_answer() {
        let (worker, client) = bridge(4);
        let barrier = Arc::new(Barrier::new(3));
        let mut answer_reporter = worker.reporter.clone();
        let answer_barrier = Arc::clone(&barrier);
        let answer = thread::spawn(move || {
            answer_barrier.wait();
            answer_reporter.model_text("racing answer");
        });
        let finish_reporter = worker.reporter.clone();
        let finish_barrier = Arc::clone(&barrier);
        let finish = thread::spawn(move || {
            finish_barrier.wait();
            finish_reporter.finish(&LoopEnd::Repeated);
        });
        barrier.wait();
        answer.join().unwrap();
        finish.join().unwrap();

        let events = client.events.try_iter().collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, WorkspaceEvent::ModelAnswer { .. }))
                .count(),
            1
        );
        assert!(matches!(
            events.last(),
            Some(WorkspaceEvent::Finished { .. })
        ));
    }

    #[test]
    fn manual_abort_finishes_without_a_synthetic_answer() {
        let (worker, client) = bridge(4);
        worker.reporter.finish(&LoopEnd::Aborted);
        assert_eq!(
            client.events.try_iter().collect::<Vec<_>>(),
            vec![WorkspaceEvent::Finished { outcome: "aborted" }]
        );
    }
}
