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
    ModelStepMetrics, Policy, Reporter,
};
use super::audit::NoopSink;
use super::client::Client;
use super::shell_sandbox::ShellSandbox;
use super::tools::{Action, Sandbox, ToolOutcome, ToolProfile};
use super::workspace_memory::MemoryContext;

const APPROVAL_POLL: Duration = Duration::from_millis(25);
const DEFAULT_APPROVAL_TIMEOUT: Duration = Duration::from_secs(5 * 60);
pub(crate) const WORKSPACE_CONTEXT_BUDGET_TOKENS: u32 = 4_096;
const WORKSPACE_MODEL_STEP_TIMEOUT: Duration = Duration::from_secs(90);

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
    pub temperature: f32,
    /// Optional session-scoped semantic index. When present, each turn gets a
    /// bounded set of relevant workspace excerpts before the model runs.
    pub semantic_retriever: Option<Arc<super::semantic_search::WorkspaceSemanticRetriever>>,
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
    let pending_approval = Arc::new(Mutex::new(None));
    (
        WorkspaceBridgeWorker {
            reporter: WorkspaceReporter {
                events: event_tx.clone(),
                delivery_failed: Arc::clone(&delivery_failed),
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

#[derive(Clone)]
pub(crate) struct WorkspaceReporter {
    events: SyncSender<WorkspaceEvent>,
    delivery_failed: Arc<AtomicBool>,
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
}

impl Reporter for WorkspaceReporter {
    fn model_text(&mut self, text: &str) {
        self.send(WorkspaceEvent::ModelAnswer {
            content: text.to_string(),
        });
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
    let sandbox = match Sandbox::new(&config.workspace, false, Duration::from_secs(30)) {
        Ok(sandbox) => sandbox.with_shell_mode(ShellSandbox::Disabled),
        Err(error) => {
            let message = error.to_string();
            worker.reporter.send(WorkspaceEvent::Error {
                message: message.clone(),
            });
            worker.reporter.send(WorkspaceEvent::Finished {
                outcome: "driver_error",
            });
            return Err(message);
        }
    };
    worker.reporter.send(WorkspaceEvent::Started {
        workspace: sandbox.root_display(),
        model_id: config.model_id.clone(),
    });
    worker.reporter.send(WorkspaceEvent::TurnStarted {
        turn_index: config.turn_index,
    });

    let system = super::agent::workspace_system_prompt(&sandbox);
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
    driver.set_context_budget(Some(WORKSPACE_CONTEXT_BUDGET_TOKENS));
    driver.set_native_tool_history(true);
    driver.set_stream_control(Arc::clone(&worker.cancel), WORKSPACE_MODEL_STEP_TIMEOUT);
    let delta_reporter = worker.reporter.clone();
    driver.set_delta_sink(Some(Box::new(move |delta| {
        delta_reporter.model_delta(delta);
    })));
    let agent_config = AgentConfig {
        workdir: config.workspace,
        max_steps: config.max_steps,
        auto_approve: false,
        yolo: false,
        allow_net: false,
        allow_fs: false,
        shell_timeout: Duration::from_secs(30),
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        audit: Box::new(NoopSink),
        shell_sandbox: ShellSandbox::Disabled,
        tool_profile: ToolProfile::WorkspaceReadOnly,
        ctx_budget: None,
    };
    let end = run_loop(
        &mut driver,
        &mut worker.approver,
        &mut worker.reporter,
        &sandbox,
        &agent_config,
        worker.cancel.as_ref(),
        &mut Policy::default(),
        &mut history,
    );
    let outcome = match end {
        LoopEnd::Answered => "answered",
        LoopEnd::Aborted => "aborted",
        LoopEnd::StepCapped => "step_capped",
        LoopEnd::Repeated => "repeated",
        LoopEnd::DriverError => "driver_error",
    };
    worker.reporter.send(WorkspaceEvent::Finished { outcome });
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
    use std::thread;

    use serde_json::{json, Value};

    use super::*;
    use crate::chat::agent::{
        run_loop, AgentConfig, AgentMsg, LoopEnd, ModelDriver, ModelStep, Policy,
    };
    use crate::chat::audit::NoopSink;
    use crate::chat::shell_sandbox::ShellSandbox;
    use crate::chat::tools::{ToolCall, ToolProfile, ToolSpec};

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
            ctx_budget: None,
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
}
