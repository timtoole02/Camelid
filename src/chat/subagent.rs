//! Subagent orchestration: spawn child `camelid` processes that each run the
//! non-interactive agent loop for ONE scoped goal, with file-based IPC.
//!
//! Design (see Phase-2 recon): the spawn plumbing reuses the proven self-reinvoke
//! pattern (`current_exe` + a hidden `__subagent` subcommand) rather than a new
//! IPC layer; the child SHARES the parent's serve (same `--addr`, so the resident
//! model is reused — never a second model load that would OOM a small box). IPC is
//! files under `.camelid/subagents/` (`task_<id>.json` in, `result_<id>.json` out)
//! so `/subagents` can list live/finished children.
//!
//! Honesty + safety: orchestration is isolation-first, NOT a speedup. A child is
//! itself an agent and is NEVER more privileged than the parent: it inherits the
//! parent's tool profile, network grant, shell-sandbox mode, and approval posture
//! (with the CAMELID_PRODUCTION fail-closed honoured). Because a child is
//! non-interactive, any action that would prompt is DENIED. A gated child is
//! therefore read-capable by default; a confirmed Web Code full-auto child may
//! also write, use explicitly enabled network tools, and run sandboxed commands.
//! Mandatory caps: a concurrency ceiling, a spawn-tree DEPTH LIMIT of 1 by default
//! (fork-bomb guard), a per-child hard timeout (→ INCONCLUSIVE, never a silent
//! hang), and reaping of wedged children. No VirtualLock / memory pinning. A
//! child's stdout/result is UNTRUSTED data.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

const SUBAGENT_DIR: &str = ".camelid/subagents";
const DEFAULT_CONCURRENCY: usize = 2;
const DEFAULT_DEPTH_LIMIT: usize = 1;
const DEFAULT_TIMEOUT_SECS: u64 = 300;

// Worker-side hard caps. The worker treats its task file as UNTRUSTED data
// (defense-in-depth: the parent validated it, but a hand-crafted file must not
// run unbounded or traverse on write), so it re-validates and clamps.
const MAX_WORKER_STEPS: usize = 30;
const MAX_WORKER_TOKENS: u32 = 4096;
const MAX_WORKER_DEPTH: usize = 8;

/// Env var carrying a child's spawn-tree depth (0 = top-level agent).
pub const DEPTH_ENV: &str = "CAMELID_SUBAGENT_DEPTH";

/// Validate a subtask id: `^[a-z0-9-]{1,64}$`. Used ONLY as a filename component —
/// no path separators, no traversal, no case games.
pub fn valid_subtask_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Convert a model-facing task name into the strict filename-safe id used by
/// the worker protocol. Human-readable aliases are deliberately more tolerant
/// than the storage boundary: case is folded and `_`, whitespace, and repeated
/// separators become one `-`. Path punctuation and every other character fail
/// closed instead of being silently laundered into a filename.
pub fn normalize_subtask_id(id: &str) -> Result<String, String> {
    let id = id.trim();
    if id.is_empty() {
        return Err("subtask_id must not be empty".into());
    }

    let mut normalized = String::with_capacity(id.len().min(64));
    let mut pending_separator = false;
    for byte in id.bytes() {
        if byte.is_ascii_alphanumeric() {
            if pending_separator && !normalized.is_empty() && normalized.len() < 64 {
                normalized.push('-');
            }
            pending_separator = false;
            if normalized.len() < 64 {
                normalized.push(byte.to_ascii_lowercase() as char);
            }
        } else if byte == b'-' || byte == b'_' || byte.is_ascii_whitespace() {
            pending_separator = !normalized.is_empty();
        } else {
            return Err(format!(
                "invalid subtask_id {id:?}; use letters, numbers, spaces, `_`, or `-`"
            ));
        }
    }
    while normalized.ends_with('-') {
        normalized.pop();
    }
    if !valid_subtask_id(&normalized) {
        return Err(format!("invalid subtask_id {id:?}"));
    }
    Ok(normalized)
}

/// Resolve the optional model-facing alias to a stable internal id. When the
/// model omits it, derive a readable, deterministic id from the goal so a retry
/// of the same spawn request is idempotent rather than creating duplicate work.
pub fn canonical_subtask_id(alias: Option<&str>, goal: &str) -> Result<String, String> {
    if let Some(alias) = alias.filter(|value| !value.trim().is_empty()) {
        return normalize_subtask_id(alias);
    }

    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in goal.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    let mut slug = String::new();
    let mut pending_separator = false;
    for byte in goal.bytes() {
        if byte.is_ascii_alphanumeric() {
            if pending_separator && !slug.is_empty() && slug.len() < 41 {
                slug.push('-');
            }
            pending_separator = false;
            if slug.len() < 41 {
                slug.push(byte.to_ascii_lowercase() as char);
            }
        } else {
            pending_separator = !slug.is_empty();
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    let generated = if slug.is_empty() {
        format!("task-{hash:016x}")
    } else {
        format!("task-{slug}-{hash:016x}")
    };
    debug_assert!(valid_subtask_id(&generated));
    Ok(generated)
}

/// The scoped instructions handed to a child (NOT the parent's full history).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub subtask_id: String,
    pub goal: String,
    pub addr: String,
    pub model_id: String,
    pub family: String,
    pub workdir: String,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub depth: usize,
    /// The parent's resolved approval posture, inherited so a child is never more
    /// privileged than its parent (auto-approve still fails closed under
    /// CAMELID_PRODUCTION in the worker).
    pub auto_approve: bool,
    /// The parent's shell-sandbox mode (as_str), inherited — never hardcoded.
    pub shell_mode: String,
    /// The parent's explicit network grant. False for older task files and for
    /// every CLI subagent unless its session deliberately opts in.
    #[serde(default)]
    pub allow_net: bool,
    /// Whether the parent granted unattended Exec as well as writes. This is
    /// separate from `auto_approve`, whose historical contract does not include
    /// shell execution.
    #[serde(default)]
    pub yolo: bool,
    /// Keep browser/Desktop Code children on the narrow WebCode allowlist
    /// instead of silently expanding them to the terminal's Full profile.
    #[serde(default)]
    pub web_code: bool,
    /// Test hook: when set, the worker uses a deterministic canned driver (one
    /// read-only tool call, then this answer) instead of contacting a model — so
    /// orchestration mechanics are verifiable without a tool-capable model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canned_answer: Option<String>,
    /// Test hook: canned worker sleeps this long before answering, so the
    /// concurrency-cap and timeout/reaping cases have a deterministically-live
    /// child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canned_sleep_ms: Option<u64>,
}

/// The child's terminal report, written on exit (success OR failure).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentResult {
    pub subtask_id: String,
    /// `completed` | `failed` | `inconclusive`.
    pub status: String,
    pub answer: String,
    #[serde(default)]
    pub tool_calls: Vec<String>,
    pub note: String,
}

/// Internal worker result. Keep the typed outcome beside the serialized report
/// until the process exit code is chosen; status text is an output boundary,
/// not control-flow input.
struct TaskExecution {
    report: SubagentResult,
    outcome: super::agent::RunOutcome,
}

/// Per-session orchestration settings, installed once at session start.
#[derive(Clone)]
pub struct SubagentConfig {
    pub addr: SocketAddr,
    pub model_id: String,
    pub family: String,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub concurrency: usize,
    pub depth_limit: usize,
    pub timeout: Duration,
    /// The parent's approval posture, inherited by every child.
    pub auto_approve: bool,
    /// The parent's shell-sandbox mode, inherited by every child.
    pub shell_mode: super::shell_sandbox::ShellSandbox,
    /// The parent's explicit network grant.
    pub allow_net: bool,
    /// The parent's unattended Exec grant.
    pub yolo: bool,
    /// True when the child must remain on the WebCode tool allowlist.
    pub web_code: bool,
}

impl SubagentConfig {
    /// A config for a real agent session (caps at conservative defaults). The
    /// child inherits the parent's `auto_approve` + `shell_mode` so it is never
    /// more privileged than the parent.
    pub fn for_session(
        addr: SocketAddr,
        model_id: String,
        family: String,
        max_tokens: u32,
        auto_approve: bool,
        shell_mode: super::shell_sandbox::ShellSandbox,
    ) -> Self {
        Self {
            addr,
            model_id,
            family,
            max_steps: 12,
            max_tokens,
            concurrency: DEFAULT_CONCURRENCY,
            depth_limit: DEFAULT_DEPTH_LIMIT,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            auto_approve,
            shell_mode,
            allow_net: false,
            yolo: false,
            web_code: false,
        }
    }

    /// Browser/Desktop Code child configuration. Unlike the legacy terminal
    /// constructor, this preserves the complete parent capability boundary:
    /// narrow WebCode tools, the explicit network switch, and full-auto Exec.
    pub fn for_web_code_session(
        addr: SocketAddr,
        model_id: String,
        family: String,
        max_tokens: u32,
        full_auto: bool,
        allow_net: bool,
        shell_mode: super::shell_sandbox::ShellSandbox,
    ) -> Self {
        let mut config =
            Self::for_session(addr, model_id, family, max_tokens, full_auto, shell_mode);
        config.allow_net = allow_net;
        config.yolo = full_auto;
        config.web_code = true;
        config
    }
}

struct ChildEntry {
    subtask_id: String,
    child: Child,
    #[cfg(windows)]
    job: Option<super::win_job::JobObject>,
    started: Instant,
    timeout: Duration,
    task_path: PathBuf,
    result_path: PathBuf,
}

struct SessionState {
    config: Option<SubagentConfig>,
    children: Vec<ChildEntry>,
}

fn registry() -> &'static Mutex<SessionState> {
    static REG: OnceLock<Mutex<SessionState>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(SessionState {
            config: None,
            children: Vec::new(),
        })
    })
}

/// Lock the registry, recovering from a poisoned lock (the state is a plain
/// bookkeeping registry; a panic elsewhere must not wedge orchestration).
fn lock_registry() -> MutexGuard<'static, SessionState> {
    registry().lock().unwrap_or_else(|e| e.into_inner())
}

/// Install the orchestration config for this process/session. Until called,
/// spawning is refused and the tools are not advertised.
pub fn configure(config: SubagentConfig) {
    lock_registry().config = Some(config);
}

/// Test-only: install `config` and get a guard that restores the previous
/// registry state on drop. The registry is process-global, so a bare
/// `configure` in a test leaks into every later test in the binary — the
/// tool-set pins in `tools` assert on the unconfigured baseline and fail
/// under the test harness's scheduling, not deterministically.
#[cfg(test)]
pub(crate) fn configure_for_test(config: SubagentConfig) -> TestConfigGuard {
    TestConfigGuard(lock_registry().config.replace(config))
}

#[cfg(test)]
pub(crate) struct TestConfigGuard(Option<SubagentConfig>);

#[cfg(test)]
impl Drop for TestConfigGuard {
    fn drop(&mut self) {
        lock_registry().config = self.0.take();
    }
}

/// Stop every child owned by the current session. Web Code calls this at both
/// ends of a turn so the visible Stop control is authoritative for delegated
/// work too, and a child cannot keep editing after its parent has finished.
pub fn cancel_all() {
    let children = {
        let mut state = lock_registry();
        std::mem::take(&mut state.children)
    };
    for mut entry in children {
        terminate_tree(&mut entry);
        let _ = std::fs::remove_file(&entry.task_path);
    }
}

/// Terminate a child and everything it spawned, then reap it.
///
/// Windows tears the tree down through the kill-on-close job object; Unix
/// through the process group the worker was spawned into. `Child::kill` remains
/// the backstop for the case where neither could be established (it reaps only
/// the direct child, which is why it is not the primary mechanism).
fn terminate_tree(entry: &mut ChildEntry) {
    #[cfg(windows)]
    if let Some(ref j) = entry.job {
        j.terminate();
    }
    #[cfg(unix)]
    {
        // Negative pid = "every process in this group". Best-effort: the group
        // may already be gone, and a worker spawned before `process_group` was
        // wired simply falls through to the direct-child kill below.
        let pid = entry.child.id() as i32;
        if pid > 0 {
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
    let _ = entry.child.kill();
    let _ = entry.child.wait();
}

/// This process's spawn-tree depth (0 for the top-level agent).
pub fn current_depth() -> usize {
    std::env::var(DEPTH_ENV)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

/// Whether spawn_subagent should be advertised/usable now: configured AND below
/// the depth limit (the depth-1 default means subagents do not see the tool).
pub fn is_enabled() -> bool {
    let state = lock_registry();
    match state.config.as_ref() {
        Some(c) => current_depth() < c.depth_limit,
        None => false,
    }
}

fn subagent_dir(root: &Path) -> PathBuf {
    root.join(SUBAGENT_DIR)
}
fn task_path(root: &Path, id: &str) -> PathBuf {
    subagent_dir(root).join(format!("task_{id}.json"))
}
fn result_path(root: &Path, id: &str) -> PathBuf {
    subagent_dir(root).join(format!("result_{id}.json"))
}

/// Spawn a subagent for `goal`, returning a human/agent-readable status line.
/// Enforces the depth guard, the concurrency cap, and subtask_id validity.
pub fn spawn(root: &Path, subtask_id: &str, goal: &str) -> Result<String, String> {
    spawn_inner(root, subtask_id, goal, None)
}

/// Spawn a subagent that runs the deterministic canned driver (test/gate hook):
/// it makes one read-only tool call, optionally sleeps `sleep_ms`, then answers.
pub fn spawn_canned(
    root: &Path,
    subtask_id: &str,
    goal: &str,
    answer: &str,
    sleep_ms: u64,
) -> Result<String, String> {
    spawn_inner(root, subtask_id, goal, Some((answer.to_string(), sleep_ms)))
}

fn spawn_inner(
    root: &Path,
    subtask_id: &str,
    goal: &str,
    canned: Option<(String, u64)>,
) -> Result<String, String> {
    if !valid_subtask_id(subtask_id) {
        return Err(format!(
            "invalid subtask_id {subtask_id:?} (allowed: ^[a-z0-9-]{{1,64}}$)"
        ));
    }

    let mut state = lock_registry();
    let config = state
        .config
        .clone()
        .ok_or_else(|| "subagent orchestration is not configured for this session".to_string())?;

    // Depth guard (fork-bomb): subagents may not spawn deeper by default.
    let depth = current_depth();
    if depth >= config.depth_limit {
        return Err(format!(
            "subagent depth limit reached ({depth} >= {}); deeper spawning is disabled",
            config.depth_limit
        ));
    }

    // Reap finished/timed-out children before counting live ones.
    reap_locked(&mut state);

    // Repeating an already-admitted spawn is an idempotent status lookup, not a
    // validation failure. Small local models commonly retry a call after missing
    // its result; launching a second child or bouncing the retry back as another
    // error both turn a harmless recovery into a loop.
    if state.children.iter().any(|c| c.subtask_id == subtask_id)
        || result_path(root, subtask_id).exists()
        || task_path(root, subtask_id).exists()
    {
        drop(state);
        return status(root, subtask_id).map(|report| {
            format!("{report}\nnote: this subtask was already admitted; do not spawn it again")
        });
    }

    let live = state.children.len();
    if live >= config.concurrency {
        return Err(format!(
            "subagent concurrency cap reached ({live}/{}); wait for one to finish (check_subagent_status)",
            config.concurrency
        ));
    }

    let dir = subagent_dir(root);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;

    // Eval hook: force a deterministic canned subagent for a tool-driven spawn
    // (CAMELID_SUBAGENT_FORCE_CANNED). Used ONLY by the rung-3 real-model eval to
    // isolate the model's orchestration-driving from subagent inference. Unset in
    // production, and a model cannot set process env, so this is inert there.
    let canned = canned.or_else(|| {
        std::env::var("CAMELID_SUBAGENT_FORCE_CANNED")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|a| (a, 0))
    });
    let (canned_answer, canned_sleep_ms) = match canned {
        Some((answer, sleep_ms)) => (Some(answer), Some(sleep_ms)),
        None => (None, None),
    };
    let task = TaskSpec {
        subtask_id: subtask_id.to_string(),
        goal: goal.to_string(),
        addr: config.addr.to_string(),
        model_id: config.model_id.clone(),
        family: config.family.clone(),
        workdir: root.display().to_string(),
        max_steps: config.max_steps,
        max_tokens: config.max_tokens,
        depth: depth + 1,
        auto_approve: config.auto_approve,
        shell_mode: config.shell_mode.as_str().to_string(),
        allow_net: config.allow_net,
        yolo: config.yolo,
        web_code: config.web_code,
        canned_answer,
        canned_sleep_ms,
    };
    let tpath = task_path(root, subtask_id);
    let task_json = serde_json::to_string_pretty(&task).map_err(|e| e.to_string())?;
    std::fs::write(&tpath, task_json).map_err(|e| format!("cannot write task file: {e}"))?;

    // Self-reinvoke as the hidden worker (reuses the gait-trial spawn template).
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate camelid binary: {e}"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("__subagent")
        .arg("--task-file")
        .arg(&tpath)
        .env(DEPTH_ENV, (depth + 1).to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    // Unix counterpart of the Windows job object below: put the worker in its own
    // process group so teardown can signal the GROUP. A worker that is mid
    // `run_shell` has a live `/bin/sh -c …` tree, and `Child::kill` signals only
    // the worker pid — the shell tree would survive Stop and keep writing into the
    // workspace, which is exactly what the "children are killed when the parent
    // turn ends or is stopped" contract forbids. Descendants inherit the group, so
    // one `killpg` reaps the whole tree.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let child = cmd.spawn().map_err(|e| {
        let _ = std::fs::remove_file(&tpath);
        format!("spawn failed: {e}")
    })?;

    #[cfg(windows)]
    let job = {
        use std::os::windows::io::AsRawHandle;
        let j = super::win_job::JobObject::new().ok();
        if let Some(ref jj) = j {
            let _ = jj.assign(child.as_raw_handle());
        }
        j
    };

    state.children.push(ChildEntry {
        subtask_id: subtask_id.to_string(),
        child,
        #[cfg(windows)]
        job,
        started: Instant::now(),
        timeout: config.timeout,
        task_path: tpath,
        result_path: result_path(root, subtask_id),
    });

    Ok(format!(
        "status: accepted\nsubtask_id: {subtask_id}\nnote: child started at depth {}; use this returned subtask_id with check_subagent_status",
        depth + 1
    ))
}

/// Report a subagent's status (reaps first). Result text is UNTRUSTED data.
pub fn status(root: &Path, subtask_id: &str) -> Result<String, String> {
    if !valid_subtask_id(subtask_id) {
        return Err(format!("invalid subtask_id {subtask_id:?}"));
    }
    reap_locked(&mut lock_registry());

    let rpath = result_path(root, subtask_id);
    if let Ok(text) = std::fs::read_to_string(&rpath) {
        return Ok(match serde_json::from_str::<SubagentResult>(&text) {
            Ok(res) => format!(
                "status: {}\nsubtask_id: {subtask_id}\nnote: {}\ntool_calls: {}\nanswer:\n{}",
                res.status,
                res.note,
                res.tool_calls.join(", "),
                res.answer
            ),
            // A malformed/partial result is treated as failed data, never a crash.
            Err(e) => format!("status: failed\nnote: result file is malformed ({e})"),
        });
    }

    let elapsed = lock_registry()
        .children
        .iter()
        .find(|c| c.subtask_id == subtask_id)
        .map(|c| c.started.elapsed());
    if elapsed.is_some() || task_path(root, subtask_id).exists() {
        // Carry elapsed time so consecutive polls are not byte-identical. The
        // caller's no-progress guard stops a call that returns the SAME text
        // REPEAT_LIMIT times running; a constant "has not finished yet" made a
        // healthy child look stuck after three polls and got the whole turn
        // killed. `RUNNING_STATUS_PREFIX` is the belt to this suspenders — see
        // `is_running_status`.
        let waited = elapsed.map(|e| e.as_secs_f64()).unwrap_or(0.0);
        Ok(format!(
            "{RUNNING_STATUS_PREFIX}\nsubtask_id: {subtask_id}\nnote: subagent {subtask_id:?} has not finished yet \
             ({waited:.1}s elapsed)"
        ))
    } else {
        Err(format!("no subagent {subtask_id:?} found"))
    }
}

/// Leading line of a still-running status report.
const RUNNING_STATUS_PREFIX: &str = "status: running";

/// Whether a tool result reports a subagent that is still working.
///
/// Waiting on a child is progress, not repetition: the agent loop's no-progress
/// guard must not count these against `REPEAT_LIMIT`. Polling a child is the
/// documented way to wait for one (`spawn_subagent` says so in its own result),
/// and Code mode has no step cap, so this guard is the loop's main terminator —
/// mistaking a poll for a stall ends the turn AND kills the child with it.
pub fn is_running_status(text: &str) -> bool {
    text.starts_with(RUNNING_STATUS_PREFIX)
}

/// A compact, truncated listing of this session's subagents — live (from the
/// registry) and finished (from result files on disk) — for the `/subagents`
/// command. The child statuses/answers it surfaces are UNTRUSTED data.
pub fn list_summary(root: &Path) -> String {
    const MAX_LISTED: usize = 40;
    reap_locked(&mut lock_registry());

    let mut lines: Vec<String> = Vec::new();
    {
        let state = lock_registry();
        for c in &state.children {
            lines.push(format!(
                "  {} — running ({:.0}s)",
                c.subtask_id,
                c.started.elapsed().as_secs_f64()
            ));
        }
    }
    if let Ok(entries) = std::fs::read_dir(subagent_dir(root)) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(id) = name
                .strip_prefix("result_")
                .and_then(|s| s.strip_suffix(".json"))
            {
                let status = std::fs::read_to_string(e.path())
                    .ok()
                    .and_then(|t| serde_json::from_str::<SubagentResult>(&t).ok())
                    .map(|r| r.status)
                    .unwrap_or_else(|| "malformed".to_string());
                lines.push(format!("  {id} — {status}"));
            }
        }
    }
    if lines.is_empty() {
        return "no subagents".to_string();
    }
    lines.sort();
    lines.dedup();
    let shown = lines.len().min(MAX_LISTED);
    let mut out = format!(
        "subagents (untrusted child output):\n{}",
        lines[..shown].join("\n")
    );
    if lines.len() > shown {
        out.push_str(&format!("\n  …and {} more", lines.len() - shown));
    }
    out
}

/// Reap children that finished or exceeded their timeout. A timed-out child is
/// terminated along with its process tree and recorded INCONCLUSIVE; one that
/// vanished without a result is recorded failed. Removes them from the live set.
fn reap_locked(state: &mut SessionState) {
    state.children.retain_mut(|entry| {
        if entry.result_path.exists() {
            // Produced its own result → no longer live. Reap so a completed child
            // doesn't linger as a zombie (no-op on Windows; matters on Linux).
            let _ = entry.child.wait();
            return false;
        }
        match entry.child.try_wait() {
            Ok(Some(_)) => {
                // try_wait already reaped the exited child.
                write_terminal_result(entry, "failed", "subagent exited without a result");
                false
            }
            Ok(None) => {
                if entry.started.elapsed() >= entry.timeout {
                    terminate_tree(entry);
                    write_terminal_result(
                        entry,
                        "inconclusive",
                        "subagent timed out and was terminated",
                    );
                    false
                } else {
                    true
                }
            }
            Err(_) => {
                let _ = entry.child.wait();
                false
            }
        }
    });
}

/// Write a result file atomically — a same-dir temp then rename — so a concurrent
/// reader never sees a half-written (and thus "malformed") file.
fn write_result_atomic(path: &Path, contents: &str) {
    let name = format!(
        "{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("result")
    );
    let mut tmp = path.to_path_buf();
    tmp.set_file_name(name);
    if std::fs::write(&tmp, contents).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn write_terminal_result(entry: &ChildEntry, status: &str, note: &str) {
    if entry.result_path.exists() {
        return;
    }
    let res = SubagentResult {
        subtask_id: entry.subtask_id.clone(),
        status: status.to_string(),
        answer: String::new(),
        tool_calls: Vec::new(),
        note: note.to_string(),
    };
    if let Ok(j) = serde_json::to_string_pretty(&res) {
        write_result_atomic(&entry.result_path, &j);
    }
}

// --- the worker (hidden `__subagent` subcommand) --------------------------

/// Worker entry: read the task file, run ONE scoped agent loop (real or canned),
/// and write the result file. Returns the exit code (0/1/3 = completed/failed/
/// inconclusive).
pub fn run_worker(task_file: &Path) -> anyhow::Result<i32> {
    let text = std::fs::read_to_string(task_file)?;
    let task: TaskSpec = serde_json::from_str(&text)?;

    // Defense-in-depth: the task file is untrusted. subtask_id is a filename
    // component of the result path, so re-validate it — a hand-crafted task file
    // must not traverse on write. Bound the depth too (in case the env/file was
    // tampered) before doing any work.
    if !valid_subtask_id(&task.subtask_id) {
        anyhow::bail!("worker refused: invalid subtask_id {:?}", task.subtask_id);
    }
    if task.depth > MAX_WORKER_DEPTH {
        anyhow::bail!(
            "worker refused: depth {} exceeds ceiling {MAX_WORKER_DEPTH}",
            task.depth
        );
    }

    let execution = execute_task(&task);

    let dir = task_file.parent().unwrap_or_else(|| Path::new("."));
    let rpath = dir.join(format!("result_{}.json", task.subtask_id));
    let mut j = serde_json::to_string_pretty(&execution.report)?;
    j.push('\n');
    write_result_atomic(&rpath, &j);

    // Consume the task file (cleanup); the result file remains for polling.
    let _ = std::fs::remove_file(task_file);

    Ok(execution.outcome.exit_code())
}

fn execute_task(task: &TaskSpec) -> TaskExecution {
    use super::agent::{self, AgentMsg, LiveDriver};
    use super::tools::Sandbox;
    use std::sync::atomic::AtomicBool;

    let fail = |outcome: agent::RunOutcome, note: String| TaskExecution {
        report: SubagentResult {
            subtask_id: task.subtask_id.clone(),
            status: outcome.subagent_status().to_string(),
            answer: String::new(),
            tool_calls: Vec::new(),
            note,
        },
        outcome,
    };

    // Inherit the parent's confinement posture — NEVER hardcode Unrestricted. A
    // subagent must never be more privileged than the parent that spawned it.
    let shell_mode = task
        .shell_mode
        .parse::<super::shell_sandbox::ShellSandbox>()
        .unwrap_or(super::shell_sandbox::ShellSandbox::Sandboxed);
    // Clamp the loop budget — a crafted/buggy task file can't make a child loop
    // or generate unbounded.
    let max_steps = task.max_steps.clamp(1, MAX_WORKER_STEPS);
    let max_tokens = task.max_tokens.clamp(1, MAX_WORKER_TOKENS);
    let root = Path::new(&task.workdir);
    let tool_profile = if task.web_code {
        super::tools::ToolProfile::WebCode
    } else {
        super::tools::ToolProfile::Full
    };
    let sandbox = match Sandbox::new(root, task.allow_net, Duration::from_secs(60)) {
        Ok(s) => s.with_shell_mode(shell_mode),
        Err(e) => return fail(agent::RunOutcome::Failed, format!("sandbox: {e}")),
    };
    let tools = super::tools::specs_for(tool_profile, task.allow_net, sandbox.shell_mode());

    let mut reporter = CaptureReporter::default();
    // A subagent is NON-INTERACTIVE: there is no human to confirm an action, so
    // anything that would prompt (Write/Network unless auto-approved; Exec always)
    // is DENIED. The child is therefore strictly no more privileged than the
    // parent — it cannot run an unattended shell.
    let mut approver = NonInteractiveApprover;
    let cancel = AtomicBool::new(false);
    let cfg = agent::AgentConfig {
        workdir: root.to_path_buf(),
        max_steps,
        auto_approve: task.auto_approve,
        yolo: task.yolo,
        allow_net: task.allow_net,
        allow_fs: false,
        shell_timeout: Duration::from_secs(60),
        max_tokens,
        temperature: 0.0,
        audit: Box::new(super::audit::NoopSink),
        shell_sandbox: shell_mode,
        tool_profile,
        // A subagent runs a real, open-ended goal, so it gets the same context
        // protection the parent has.
        ctx_budget: Some(agent::AGENT_VALIDATED_CTX),
    };
    // The parent's approval posture, with the production fail-closed honoured:
    // resolve_policy refuses blanket auto-approve under CAMELID_PRODUCTION, so a
    // child can never silently run write/network there.
    // Web Code's confirmed full-auto mode includes Exec; approval-gated children
    // still deny every confirm-tier action because nobody is present to answer.
    let mut policy = agent::resolve_policy(task.auto_approve, task.yolo, agent::is_production())
        .unwrap_or_default();
    // A subagent does real work in the user's workspace, so it gets the same
    // project context its parent has. (The gate harnesses in agent_eval.rs and
    // agent_orchestration.rs deliberately do not — see D-DROVER-6.)
    let project = agent::load_project_context(&sandbox);
    let mut history = vec![
        AgentMsg::System(agent::system_prompt_with_project(
            &sandbox,
            &tools,
            project.as_ref(),
        )),
        AgentMsg::User(task.goal.clone()),
    ];

    let end = if let Some(answer) = &task.canned_answer {
        // Deterministic, model-free path (test/gate).
        let mut driver = CannedDriver::new(answer.clone(), task.canned_sleep_ms.unwrap_or(0));
        agent::run_loop(
            &mut driver,
            &mut approver,
            &mut reporter,
            &sandbox,
            &cfg,
            &cancel,
            &mut policy,
            &mut history,
        )
    } else {
        // Real path: attach to the parent's shared serve (resident model reused).
        let addr: SocketAddr = match task.addr.parse() {
            Ok(a) => a,
            Err(e) => {
                return fail(
                    agent::RunOutcome::Failed,
                    format!("bad addr {:?}: {e}", task.addr),
                )
            }
        };
        let client = super::client::Client::new(addr);
        let _server = match super::server::ServerHandle::ensure(addr, &client) {
            Ok(s) => s,
            Err(e) => {
                return fail(
                    agent::RunOutcome::Inconclusive,
                    format!("shared serve unavailable: {e}"),
                )
            }
        };
        let mut driver = LiveDriver::with(
            client,
            task.model_id.clone(),
            task.family.clone(),
            max_tokens,
            0.0,
        );
        agent::run_loop(
            &mut driver,
            &mut approver,
            &mut reporter,
            &sandbox,
            &cfg,
            &cancel,
            &mut policy,
            &mut history,
        )
    };

    // A finished loop is classified in exactly one place -- the shared
    // `RunOutcome` tri-state in `agent` -- so this worker and the headless
    // `agent exec` front end can never disagree on what a step-capped or
    // repeating run means. Both now call a step-capped or repeating run
    // *inconclusive* (exit 3), not a failure: the run ran out of budget or got
    // stuck, which re-running with more steps or a different model may resolve.
    // Only a driver error is a hard failure here. (Resolves the divergence D18
    // addendum 1 recorded; see D18 addendum 3.)
    let outcome = agent::RunOutcome::classify(&end);
    TaskExecution {
        report: SubagentResult {
            subtask_id: task.subtask_id.clone(),
            status: outcome.subagent_status().to_string(),
            answer: reporter.answer,
            tool_calls: reporter.calls,
            note: format!("loop ended: {end:?}"),
        },
        outcome,
    }
}

#[derive(Default)]
struct CaptureReporter {
    answer: String,
    calls: Vec<String>,
}
impl super::agent::Reporter for CaptureReporter {
    fn model_text(&mut self, text: &str) {
        self.answer = text.to_string();
    }
    fn tool_call(&mut self, line: &str) {
        self.calls.push(line.to_string());
    }
    fn tool_result(&mut self, _name: &str, _outcome: &super::tools::ToolOutcome) {}
    fn notice(&mut self, _text: &str) {}
}

/// A subagent runs unattended, so there is no human to confirm a gated action.
/// This approver DENIES every action it is consulted for (i.e. every Confirm-tier
/// action: Write/Network unless the policy auto-approved them, and Exec unless
/// the parent explicitly enabled a non-production full-auto posture). The child
/// therefore cannot exceed the parent's posture.
/// Denies everything that needs approval. Used wherever no human is present:
/// a subagent worker, and `agent exec` without `--yolo`. "Nobody to ask" means
/// *more* conservative, not less.
pub(crate) struct NonInteractiveApprover;
impl super::agent::Approver for NonInteractiveApprover {
    fn approve(
        &mut self,
        _action: &super::tools::Action,
        _sandbox: &super::tools::Sandbox,
    ) -> super::agent::Decision {
        super::agent::Decision::No
    }
}

/// A deterministic driver: one read-only tool call (proving the subagent executes
/// tools), then the canned final answer. No model, no server.
struct CannedDriver {
    answer: String,
    sleep_ms: u64,
    step: usize,
}
impl CannedDriver {
    fn new(answer: String, sleep_ms: u64) -> Self {
        Self {
            answer,
            sleep_ms,
            step: 0,
        }
    }
}
impl super::agent::ModelDriver for CannedDriver {
    fn step(
        &mut self,
        _history: &[super::agent::AgentMsg],
        _tools: &[super::tools::ToolSpec],
    ) -> Result<super::agent::ModelStep, String> {
        self.step += 1;
        if self.step == 1 {
            Ok(super::agent::ModelStep::Calls(vec![
                super::tools::ToolCall {
                    name: "list_dir".to_string(),
                    args: serde_json::json!({ "path": "." }),
                },
            ]))
        } else {
            // Stay alive long enough for the cap/timeout cases to observe a live
            // child before the final answer.
            if self.sleep_ms > 0 {
                std::thread::sleep(Duration::from_millis(self.sleep_ms));
            }
            Ok(super::agent::ModelStep::Text(self.answer.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtask_id_validation() {
        assert!(valid_subtask_id("abc-123"));
        assert!(valid_subtask_id("a"));
        assert!(!valid_subtask_id(""));
        assert!(!valid_subtask_id("../etc"));
        assert!(!valid_subtask_id("has space"));
        assert!(!valid_subtask_id("UPPER"));
        assert!(!valid_subtask_id("dir/child"));
        assert!(!valid_subtask_id("dot.dot"));
        assert!(!valid_subtask_id(&"x".repeat(65)));
    }

    #[test]
    fn model_facing_subtask_names_are_normalized_before_storage() {
        assert_eq!(
            normalize_subtask_id("Generate_Tic Tac--Toe_Code").unwrap(),
            "generate-tic-tac-toe-code"
        );
        assert!(normalize_subtask_id("../escape").is_err());
        assert!(normalize_subtask_id("dir/child").is_err());
        assert!(normalize_subtask_id("all.good").is_err());
        assert!(normalize_subtask_id("___").is_err());
    }

    #[test]
    fn omitted_subtask_id_is_safe_stable_and_goal_specific() {
        let goal = "Inspect the coding frontend and report how tools are admitted.";
        let first = canonical_subtask_id(None, goal).unwrap();
        let retry = canonical_subtask_id(None, goal).unwrap();
        let other = canonical_subtask_id(None, "Inspect a different feature.").unwrap();
        assert!(valid_subtask_id(&first), "{first}");
        assert_eq!(first, retry);
        assert_ne!(first, other);
        assert!(first.len() <= 64);
    }

    #[test]
    fn web_code_children_inherit_the_complete_parent_boundary() {
        let config = SubagentConfig::for_web_code_session(
            "127.0.0.1:8181".parse().unwrap(),
            "model".into(),
            "qwen3".into(),
            2048,
            true,
            true,
            super::super::shell_sandbox::ShellSandbox::Sandboxed,
        );
        assert!(config.auto_approve);
        assert!(config.yolo);
        assert!(config.allow_net);
        assert!(config.web_code);
        assert_eq!(
            config.shell_mode,
            super::super::shell_sandbox::ShellSandbox::Sandboxed
        );
    }

    #[test]
    fn older_task_files_default_to_the_conservative_terminal_boundary() {
        let task: TaskSpec = serde_json::from_value(serde_json::json!({
            "subtask_id": "legacy",
            "goal": "inspect",
            "addr": "127.0.0.1:8181",
            "model_id": "model",
            "family": "qwen3",
            "workdir": ".",
            "max_steps": 4,
            "max_tokens": 64,
            "depth": 1,
            "auto_approve": false,
            "shell_mode": "sandboxed"
        }))
        .unwrap();
        assert!(!task.allow_net);
        assert!(!task.yolo);
        assert!(!task.web_code);
    }

    #[test]
    fn malformed_result_is_handled_not_crashed() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(subagent_dir(root)).unwrap();
        std::fs::write(result_path(root, "bad"), "{ not json").unwrap();
        let out = status(root, "bad").unwrap();
        assert!(out.contains("failed") && out.contains("malformed"), "{out}");
    }

    #[test]
    fn status_rejects_traversal_id() {
        let dir = tempfile::tempdir().unwrap();
        assert!(status(dir.path(), "../escape").is_err());
    }

    #[test]
    fn spawn_refused_when_unconfigured() {
        // No configure() in this unit (the global may be configured by another
        // test, so only assert the validation path here).
        let dir = tempfile::tempdir().unwrap();
        assert!(spawn(dir.path(), "bad id", "goal").is_err());
    }

    #[test]
    fn list_summary_lists_finished_and_empty() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(subagent_dir(root)).unwrap();
        let res = SubagentResult {
            subtask_id: "job-1".to_string(),
            status: "completed".to_string(),
            answer: "hi".to_string(),
            tool_calls: vec![],
            note: "n".to_string(),
        };
        std::fs::write(
            result_path(root, "job-1"),
            serde_json::to_string(&res).unwrap(),
        )
        .unwrap();
        let out = list_summary(root);
        assert!(out.contains("job-1") && out.contains("completed"), "{out}");
        assert!(out.contains("untrusted"), "{out}");
        // A root with no subagents dir → "no subagents".
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(list_summary(empty.path()), "no subagents");
    }

    fn canned_task(root: &Path, id: &str) -> TaskSpec {
        TaskSpec {
            subtask_id: id.to_string(),
            goal: "g".to_string(),
            addr: "127.0.0.1:8181".to_string(),
            model_id: "x".to_string(),
            family: "llama".to_string(),
            workdir: root.display().to_string(),
            max_steps: 4,
            max_tokens: 64,
            depth: 1,
            auto_approve: false,
            shell_mode: "sandboxed".to_string(),
            allow_net: false,
            yolo: false,
            web_code: false,
            canned_answer: Some("WORKER-OK".to_string()),
            canned_sleep_ms: None,
        }
    }

    #[test]
    fn worker_canned_roundtrip_writes_result_and_consumes_task() {
        // The canned worker runs the loop IN-PROCESS (no subprocess), so this is
        // real end-to-end coverage of run_worker.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(subagent_dir(root)).unwrap();
        let tpath = task_path(root, "wt-1");
        std::fs::write(
            &tpath,
            serde_json::to_string(&canned_task(root, "wt-1")).unwrap(),
        )
        .unwrap();

        let code = run_worker(&tpath).unwrap();
        assert_eq!(code, 0);
        let res: SubagentResult =
            serde_json::from_str(&std::fs::read_to_string(result_path(root, "wt-1")).unwrap())
                .unwrap();
        assert_eq!(res.status, "completed");
        assert!(res.answer.contains("WORKER-OK"), "{}", res.answer);
        assert!(res.tool_calls.iter().any(|c| c.contains("list_dir")));
        assert!(!tpath.exists(), "task file should be consumed");
    }

    #[test]
    fn worker_step_cap_is_inconclusive_in_status_and_exit_code() {
        // One canned model step performs list_dir; there is no second step in
        // which to answer, so this drives the real worker path to StepCapped.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(subagent_dir(root)).unwrap();
        let tpath = task_path(root, "step-cap");
        let mut task = canned_task(root, "step-cap");
        task.max_steps = 1;
        std::fs::write(&tpath, serde_json::to_string(&task).unwrap()).unwrap();

        let code = run_worker(&tpath).unwrap();
        assert_eq!(code, 3);
        let res: SubagentResult =
            serde_json::from_str(&std::fs::read_to_string(result_path(root, "step-cap")).unwrap())
                .unwrap();
        assert_eq!(res.status, "inconclusive");
        assert_eq!(res.note, "loop ended: StepCapped");
        assert!(res.tool_calls.iter().any(|call| call.contains("list_dir")));
        assert!(!tpath.exists(), "task file should be consumed");
    }

    #[test]
    fn worker_refuses_invalid_subtask_id_in_task_file() {
        // Defense-in-depth: a hand-crafted task file with a traversing subtask_id
        // is refused before any work or write.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(subagent_dir(root)).unwrap();
        let mut task = canned_task(root, "placeholder");
        task.subtask_id = "../evil".to_string();
        let tpath = subagent_dir(root).join("task_evil.json");
        std::fs::write(&tpath, serde_json::to_string(&task).unwrap()).unwrap();
        assert!(run_worker(&tpath).is_err());
    }

    #[test]
    fn subagent_denies_gated_actions() {
        // A non-interactive subagent has no human to confirm, so any gated
        // (Confirm-tier) action is denied — it can never run an unattended shell.
        use super::super::agent::{Approver, Decision};
        use super::super::tools::{Action, Sandbox};
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path(), false, Duration::from_secs(5)).unwrap();
        let mut approver = NonInteractiveApprover;
        let exec = Action::RunShell {
            command: "echo hi".to_string(),
        };
        assert_eq!(approver.approve(&exec, &sb), Decision::No);
    }
}
