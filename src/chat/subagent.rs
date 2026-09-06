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
//! Honesty + safety: orchestration is isolation-first, NOT a speedup. Workers are
//! strictly workspace-read-only: they can inspect and report, but never mutate,
//! execute, use the network, or spawn another agent. The parent remains the only
//! writer because its in-process checkpoint journal cannot safely cover edits
//! made by a separate child process.
//! Mandatory caps: a concurrency ceiling, a spawn-tree DEPTH LIMIT of 1 by default
//! (fork-bomb guard), a per-child hard timeout (→ INCONCLUSIVE, never a silent
//! hang), and reaping of wedged children. No VirtualLock / memory pinning. A
//! child's stdout/result is UNTRUSTED data.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

const SUBAGENT_DIR: &str = ".camelid/subagents";
const DEFAULT_CONCURRENCY: usize = 2;
const DEFAULT_DEPTH_LIMIT: usize = 1;
// The whole child must never expire before one model step is allowed to finish.
// LiveDriver's shared prompt+generation step ceiling is 30 minutes; retain a
// one-minute process/setup/result-publication reserve around it.
const DEFAULT_TIMEOUT_SECS: u64 = 31 * 60;

// Worker-side hard caps. The worker treats its task file as UNTRUSTED data
// (defense-in-depth: the parent validated it, but a hand-crafted file must not
// run unbounded or traverse on write), so it re-validates and clamps.
const MAX_WORKER_STEPS: usize = 30;
const MAX_WORKER_DEPTH: usize = 8;
const MAX_TASK_BYTES: u64 = 128 * 1024;
const MAX_RESULT_BYTES: u64 = 2 * 1024 * 1024;
const WORKER_TOOL_PROFILE: super::tools::ToolProfile = super::tools::ToolProfile::WorkspaceReadOnly;
const WORKER_READ_ONLY_INSTRUCTION: &str = "You are a read-only subagent. Inspect the workspace and report findings or a proposed patch to the parent. Do not claim to have modified files: only the parent process may write, execute commands, access the network, or apply changes so its undo journal remains complete.";

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

/// The scoped instructions handed to a child (NOT the parent's full history).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub subtask_id: String,
    pub goal: String,
    pub addr: String,
    pub model_id: String,
    /// Exact parent artifact identity. The child includes it on every model
    /// request, so sharing a serve never lets a same-id replacement inherit
    /// the parent's tool authority or transcript.
    pub model_sha256: String,
    pub family: String,
    pub workdir: String,
    pub max_steps: usize,
    /// Exact parent prompt-plus-reply fitter budget after all runtime ceilings.
    pub context_budget_tokens: u32,
    /// Exact parent reply allowance after the server generation ceiling.
    pub max_tokens: u32,
    pub depth: usize,
    /// Parent posture metadata retained for task-file compatibility. Worker tool
    /// selection is unconditionally workspace-read-only regardless of this bit.
    pub auto_approve: bool,
    /// The parent's shell-sandbox mode (as_str), inherited — never hardcoded.
    pub shell_mode: String,
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
    pub model_sha256: String,
    pub family: String,
    pub max_steps: usize,
    pub context_budget_tokens: u32,
    pub max_tokens: u32,
    pub concurrency: usize,
    pub depth_limit: usize,
    pub timeout: Duration,
    /// Parent posture metadata retained in the task receipt. It never broadens
    /// the worker's fixed workspace-read-only tool profile.
    pub auto_approve: bool,
    /// The parent's shell-sandbox mode, inherited by every child.
    pub shell_mode: super::shell_sandbox::ShellSandbox,
}

impl SubagentConfig {
    /// A config for a real agent session (caps at conservative defaults). The
    /// child inherits the parent's `auto_approve` + `shell_mode` so it is never
    /// more privileged than the parent.
    // These values form one exact parent-to-child security identity: endpoint,
    // artifact, protocol family, budgets, and inherited privilege posture.
    #[expect(clippy::too_many_arguments)]
    pub fn for_session(
        addr: SocketAddr,
        model_id: String,
        model_sha256: String,
        family: String,
        context_budget_tokens: u32,
        max_tokens: u32,
        auto_approve: bool,
        shell_mode: super::shell_sandbox::ShellSandbox,
    ) -> Self {
        Self {
            addr,
            model_id,
            model_sha256,
            family,
            max_steps: 12,
            context_budget_tokens,
            max_tokens,
            concurrency: DEFAULT_CONCURRENCY,
            depth_limit: DEFAULT_DEPTH_LIMIT,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            auto_approve,
            shell_mode,
        }
    }
}

fn task_from_config(
    config: &SubagentConfig,
    root: &Path,
    subtask_id: &str,
    goal: &str,
    depth: usize,
    canned_answer: Option<String>,
    canned_sleep_ms: Option<u64>,
) -> TaskSpec {
    TaskSpec {
        subtask_id: subtask_id.to_string(),
        goal: goal.to_string(),
        addr: config.addr.to_string(),
        model_id: config.model_id.clone(),
        model_sha256: config.model_sha256.clone(),
        family: config.family.clone(),
        workdir: root.display().to_string(),
        max_steps: config.max_steps,
        context_budget_tokens: config.context_budget_tokens,
        max_tokens: config.max_tokens,
        depth,
        auto_approve: config.auto_approve,
        shell_mode: config.shell_mode.as_str().to_string(),
        canned_answer,
        canned_sleep_ms,
    }
}

struct ChildEntry {
    subtask_id: String,
    child: Child,
    #[cfg(windows)]
    job: super::win_job::JobObject,
    started: Instant,
    timeout: Duration,
    result_path: PathBuf,
}

struct SessionState {
    config: Option<SubagentConfig>,
    children: Vec<ChildEntry>,
    generation: u64,
}

fn registry() -> &'static Mutex<SessionState> {
    static REG: OnceLock<Mutex<SessionState>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(SessionState {
            config: None,
            children: Vec::new(),
            generation: 0,
        })
    })
}

/// Lock the registry, recovering from a poisoned lock (the state is a plain
/// bookkeeping registry; a panic elsewhere must not wedge orchestration).
fn lock_registry() -> MutexGuard<'static, SessionState> {
    registry().lock().unwrap_or_else(|e| e.into_inner())
}

fn install_config(config: SubagentConfig) -> u64 {
    let generation = {
        let mut state = lock_registry();
        // A new owner must not inherit detached workers from the prior session.
        cancel_children_locked(&mut state, "subagent session was replaced");
        state.generation = state.generation.wrapping_add(1).max(1);
        state.config = Some(config);
        state.generation
    };
    start_watchdog();
    generation
}

/// RAII owner for one CLI/TUI/exec orchestration session. Dropping it disables
/// spawning and terminates every still-running child process tree. The
/// generation check prevents an old guard from shutting down a newer session.
#[must_use = "hold this guard for the full agent session"]
pub struct SessionGuard {
    generation: u64,
}

pub fn configure_scoped(config: SubagentConfig) -> SessionGuard {
    SessionGuard {
        generation: install_config(config),
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        let mut state = lock_registry();
        if state.generation != self.generation {
            return;
        }
        cancel_children_locked(&mut state, "parent agent session ended");
        state.config = None;
    }
}

/// Cancel live children for the current goal while keeping the session's
/// orchestration configuration available for the next interactive goal.
pub(crate) fn cancel_all(note: &str) {
    cancel_children_locked(&mut lock_registry(), note);
}

/// Timeouts must be enforced even when the parent model never polls the status
/// tool again. One detached process-local watchdog reaps all configured child
/// agents; the registry remains the single synchronization point.
fn start_watchdog() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        if let Err(err) = std::thread::Builder::new()
            .name("camelid-subagent-watchdog".to_string())
            .spawn(|| loop {
                std::thread::sleep(Duration::from_millis(250));
                let mut state = lock_registry();
                if !state.children.is_empty() {
                    reap_locked(&mut state);
                }
            })
        {
            // Poll-driven reaping remains active as a conservative fallback;
            // never panic an agent session merely because thread creation was
            // denied by an exhausted host.
            tracing::error!(error = %err, "failed to start subagent timeout watchdog");
        }
    });
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

/// Build `.camelid/subagents` one component at a time and reject symlinks or
/// non-directories at both levels. `create_dir_all` would follow a pre-planted
/// `.camelid` symlink and turn agent bookkeeping into an arbitrary filesystem
/// write outside the workspace.
fn secure_subagent_dir(root: &Path) -> Result<PathBuf, String> {
    let root = root
        .canonicalize()
        .map_err(|e| format!("cannot resolve workspace {}: {e}", root.display()))?;
    let camelid = root.join(".camelid");
    ensure_plain_directory(&camelid)?;
    let dir = camelid.join("subagents");
    ensure_plain_directory(&dir)?;
    Ok(dir)
}

fn ensure_plain_directory(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => Err(format!(
            "refusing symlink state directory {}",
            path.display()
        )),
        Ok(meta) if !meta.is_dir() => Err(format!(
            "refusing non-directory state path {}",
            path.display()
        )),
        Ok(_) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => std::fs::create_dir(path)
            .map_err(|e| format!("cannot create state directory {}: {e}", path.display())),
        Err(err) => Err(format!("cannot inspect {}: {err}", path.display())),
    }
}

fn path_is_present(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}

fn create_private_new(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

fn read_regular_bounded(path: &Path, max_bytes: u64) -> Result<String, String> {
    let meta = std::fs::symlink_metadata(path)
        .map_err(|e| format!("cannot inspect {}: {e}", path.display()))?;
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(format!(
            "refusing non-regular state file {}",
            path.display()
        ));
    }
    if meta.len() > max_bytes {
        return Err(format!(
            "state file {} exceeds the {max_bytes}-byte limit",
            path.display()
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Close the inspect/open race at the final component and ensure a raced
        // FIFO/device cannot block the parent watchdog or status path.
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Open the reparse point itself instead of following it; the post-open
        // regular-file check below then rejects symlinks/junction-like files.
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options
        .open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let opened = file
        .metadata()
        .map_err(|e| format!("cannot inspect opened {}: {e}", path.display()))?;
    if !opened.file_type().is_file() {
        return Err(format!(
            "refusing non-regular state file {}",
            path.display()
        ));
    }
    if opened.len() > max_bytes {
        return Err(format!(
            "state file {} exceeds the {max_bytes}-byte limit",
            path.display()
        ));
    }
    let mut bytes = Vec::with_capacity((opened.len() as usize).min(max_bytes as usize));
    file.take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        return Err(format!(
            "state file {} exceeds the {max_bytes}-byte limit",
            path.display()
        ));
    }
    String::from_utf8(bytes).map_err(|_| format!("state file {} is not UTF-8", path.display()))
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

    let live = state.children.len();
    if live >= config.concurrency {
        return Err(format!(
            "subagent concurrency cap reached ({live}/{}); wait for one to finish (check_subagent_status)",
            config.concurrency
        ));
    }

    // Refuse a reused id (live, or an existing task/result on disk).
    if state.children.iter().any(|c| c.subtask_id == subtask_id)
        || path_is_present(&result_path(root, subtask_id))
        || path_is_present(&task_path(root, subtask_id))
    {
        return Err(format!("subtask_id {subtask_id:?} is already in use"));
    }

    let dir = secure_subagent_dir(root)?;

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
    let task = task_from_config(
        &config,
        root,
        subtask_id,
        goal,
        depth + 1,
        canned_answer,
        canned_sleep_ms,
    );
    let tpath = dir.join(format!("task_{subtask_id}.json"));
    let task_json = serde_json::to_string_pretty(&task).map_err(|e| e.to_string())?;
    if task_json.len() as u64 > MAX_TASK_BYTES {
        return Err(format!(
            "subagent task exceeds the {MAX_TASK_BYTES}-byte limit"
        ));
    }
    let mut task_file =
        create_private_new(&tpath).map_err(|e| format!("cannot create task file: {e}"))?;
    task_file
        .write_all(task_json.as_bytes())
        .and_then(|_| task_file.sync_all())
        .map_err(|e| format!("cannot write task file: {e}"))?;

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
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Each worker owns a process group so cancelling/reaping the worker also
        // tears down any server/helper descendants it may have spawned.
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        use windows_sys::Win32::System::Threading::CREATE_SUSPENDED;
        cmd.creation_flags(0x0800_0000 | CREATE_SUSPENDED); // CREATE_NO_WINDOW
    }

    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut child = cmd.spawn().map_err(|e| {
        let _ = std::fs::remove_file(&tpath);
        format!("spawn failed: {e}")
    })?;

    #[cfg(windows)]
    let job = match super::win_job::JobObject::contain_suspended(&mut child) {
        Ok(job) => job,
        Err(error) => {
            let _ = std::fs::remove_file(&tpath);
            return Err(format!("could not contain subagent process tree: {error}"));
        }
    };

    state.children.push(ChildEntry {
        subtask_id: subtask_id.to_string(),
        child,
        #[cfg(windows)]
        job,
        started: Instant::now(),
        timeout: config.timeout,
        result_path: dir.join(format!("result_{subtask_id}.json")),
    });

    Ok(format!(
        "spawned read-only subagent {subtask_id:?} (depth {}); poll it with check_subagent_status",
        depth + 1
    ))
}

/// Report a subagent's status (reaps first). Result text is UNTRUSTED data.
pub fn status(root: &Path, subtask_id: &str) -> Result<String, String> {
    if !valid_subtask_id(subtask_id) {
        return Err(format!("invalid subtask_id {subtask_id:?}"));
    }
    reap_locked(&mut lock_registry());

    let dir = secure_subagent_dir(root)?;
    let rpath = dir.join(format!("result_{subtask_id}.json"));
    if path_is_present(&rpath) {
        let text = read_regular_bounded(&rpath, MAX_RESULT_BYTES)?;
        return Ok(match serde_json::from_str::<SubagentResult>(&text) {
            Ok(res) => format!(
                "status: {}\nnote: {}\ntool_calls: {}\nanswer:\n{}",
                res.status,
                res.note,
                res.tool_calls.join(", "),
                res.answer
            ),
            // A malformed/partial result is treated as failed data, never a crash.
            Err(e) => format!("status: failed\nnote: result file is malformed ({e})"),
        });
    }

    let live = lock_registry()
        .children
        .iter()
        .any(|c| c.subtask_id == subtask_id);
    let tpath = dir.join(format!("task_{subtask_id}.json"));
    if live
        || std::fs::symlink_metadata(&tpath)
            .map(|meta| meta.is_file() && !meta.file_type().is_symlink())
            .unwrap_or(false)
    {
        Ok(format!(
            "status: running\nnote: subagent {subtask_id:?} has not finished yet"
        ))
    } else {
        Err(format!("no subagent {subtask_id:?} found"))
    }
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
    let dir = match secure_subagent_dir(root) {
        Ok(dir) => dir,
        Err(err) => return format!("subagent state unavailable: {err}"),
    };
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(id) = name
                .strip_prefix("result_")
                .and_then(|s| s.strip_suffix(".json"))
                .filter(|id| valid_subtask_id(id))
            {
                let status = read_regular_bounded(&e.path(), MAX_RESULT_BYTES)
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

fn kill_child_descendants(entry: &ChildEntry) {
    #[cfg(windows)]
    entry.job.terminate();
    #[cfg(unix)]
    unsafe {
        // Workers are spawned as process-group leaders. A negative pid targets
        // the worker plus all descendants without affecting unrelated jobs.
        libc::kill(-(entry.child.id() as i32), libc::SIGKILL);
    }
}

fn terminate_child_tree(entry: &mut ChildEntry) {
    kill_child_descendants(entry);
    // Direct-child backstop for a failed group/job setup, then mandatory reap.
    let _ = entry.child.kill();
    let _ = entry.child.wait();
}

fn cancel_children_locked(state: &mut SessionState, note: &str) {
    // First consume children that already exited and published a valid result;
    // only live entries should be relabelled inconclusive below.
    reap_locked(state);
    for mut entry in state.children.drain(..) {
        terminate_child_tree(&mut entry);
        write_terminal_result(&entry, "inconclusive", note);
    }
}

/// Reap children that finished or exceeded their timeout. A timed-out child is
/// terminated as a whole process tree and recorded INCONCLUSIVE; one that
/// vanished without a result is recorded failed. Removes them from the live set.
fn reap_locked(state: &mut SessionState) {
    state.children.retain_mut(|entry| {
        match entry.child.try_wait() {
            Ok(Some(_)) => {
                // A worker can exit after launching a background helper. Its
                // process group/job belongs to this subtask and must not outlive
                // the result boundary.
                kill_child_descendants(entry);
                // `try_wait` already reaped the child. Only a regular bounded
                // result counts; a pre-planted symlink/pipe must not detach a
                // still-running child or become trusted state.
                if read_regular_bounded(&entry.result_path, MAX_RESULT_BYTES).is_err() {
                    write_terminal_result(entry, "failed", "subagent exited without a result");
                }
                remove_task_file(entry);
                false
            }
            Ok(None) => {
                if entry.started.elapsed() >= entry.timeout {
                    terminate_child_tree(entry);
                    write_terminal_result(
                        entry,
                        "inconclusive",
                        "subagent timed out and was terminated",
                    );
                    remove_task_file(entry);
                    false
                } else {
                    true
                }
            }
            Err(_) => {
                terminate_child_tree(entry);
                write_terminal_result(entry, "failed", "could not inspect subagent process");
                remove_task_file(entry);
                false
            }
        }
    });
}

fn remove_task_file(entry: &ChildEntry) {
    let mut path = entry.result_path.clone();
    path.set_file_name(format!("task_{}.json", entry.subtask_id));
    let _ = std::fs::remove_file(path);
}

/// Write a result file with bounded, no-clobber publication. Hard links provide
/// an atomic fast path; filesystems without link support use the platform's
/// atomic no-replace rename primitive.
fn write_result_atomic(path: &Path, contents: &str) -> Result<(), String> {
    if contents.len() as u64 > MAX_RESULT_BYTES {
        return Err(format!(
            "subagent result exceeds the {MAX_RESULT_BYTES}-byte limit"
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| "subagent result has no parent directory".to_string())?;
    let parent_meta = std::fs::symlink_metadata(parent)
        .map_err(|e| format!("cannot inspect result directory: {e}"))?;
    if parent_meta.file_type().is_symlink() || !parent_meta.is_dir() {
        return Err(format!(
            "refusing non-directory result parent {}",
            parent.display()
        ));
    }
    if path_is_present(path) {
        return Err(format!(
            "refusing to replace result file {}",
            path.display()
        ));
    }

    static TEMP_ID: AtomicU64 = AtomicU64::new(0);
    let serial = TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let base = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("result");
    let tmp = parent.join(format!(".{base}.{}.{}.tmp", std::process::id(), serial));
    let mut file =
        create_private_new(&tmp).map_err(|e| format!("cannot create result temp file: {e}"))?;
    let written = file
        .write_all(contents.as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("cannot write result temp file: {e}"));
    drop(file);
    if let Err(err) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }

    let publish = super::tools::publish_temp_noclobber(&tmp, path);
    let _ = std::fs::remove_file(&tmp);
    publish
}

fn write_terminal_result(entry: &ChildEntry, status: &str, note: &str) {
    if path_is_present(&entry.result_path) {
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
        let _ = write_result_atomic(&entry.result_path, &j);
    }
}

// --- the worker (hidden `__subagent` subcommand) --------------------------

/// Worker entry: read the task file, run ONE scoped agent loop (real or canned),
/// and write the result file. Returns the exit code (0/1/3 = completed/failed/
/// inconclusive).
pub fn run_worker(task_file: &Path) -> anyhow::Result<i32> {
    let text = read_regular_bounded(task_file, MAX_TASK_BYTES).map_err(anyhow::Error::msg)?;
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

    // The hidden worker accepts a path from the command line, so prove it is
    // the one scoped task file inside this task's canonical workspace state
    // directory before using its parent as the result destination.
    let expected_dir = secure_subagent_dir(Path::new(&task.workdir)).map_err(anyhow::Error::msg)?;
    let expected_dir = expected_dir.canonicalize()?;
    let actual_dir = task_file
        .parent()
        .ok_or_else(|| anyhow::anyhow!("worker refused: task file has no parent"))?
        .canonicalize()?;
    let expected_name = format!("task_{}.json", task.subtask_id);
    if actual_dir != expected_dir
        || task_file.file_name().and_then(|name| name.to_str()) != Some(expected_name.as_str())
    {
        anyhow::bail!(
            "worker refused: task file is not the scoped workspace task for {:?}",
            task.subtask_id
        );
    }

    let execution = execute_task(&task);

    let rpath = expected_dir.join(format!("result_{}.json", task.subtask_id));
    let mut j = serde_json::to_string_pretty(&execution.report)?;
    j.push('\n');
    write_result_atomic(&rpath, &j).map_err(anyhow::Error::msg)?;

    // Consume the task file (cleanup); the result file remains for polling.
    // Re-check the parent after generation before cleanup; if the workspace
    // state directory changed underneath us, leave the task in place instead
    // of applying a pathname operation through an attacker-controlled parent.
    if task_file
        .parent()
        .and_then(|parent| parent.canonicalize().ok())
        .as_deref()
        == Some(expected_dir.as_path())
        && std::fs::symlink_metadata(task_file)
            .map(|meta| meta.is_file() && !meta.file_type().is_symlink())
            .unwrap_or(false)
    {
        let _ = std::fs::remove_file(task_file);
    }

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
    // Clamp the step count, but preserve the parent's exact runtime-derived
    // context/reply pair. A hand-crafted task that exceeds the validated lane or
    // leaves no prompt room fails closed instead of silently gaining a budget.
    let max_steps = task.max_steps.clamp(1, MAX_WORKER_STEPS);
    if task.context_budget_tokens < 2
        || task.context_budget_tokens > agent::AGENT_VALIDATED_CTX
        || task.max_tokens == 0
        || task.max_tokens >= task.context_budget_tokens
    {
        return fail(
            agent::RunOutcome::Failed,
            format!(
                "invalid runtime budget: context={} reply={} (validated context ceiling={})",
                task.context_budget_tokens,
                task.max_tokens,
                agent::AGENT_VALIDATED_CTX
            ),
        );
    }
    let max_tokens = task.max_tokens;
    let root = Path::new(&task.workdir);
    let sandbox = match Sandbox::new(root, false, Duration::from_secs(60)) {
        Ok(s) => s.with_shell_mode(shell_mode),
        Err(e) => return fail(agent::RunOutcome::Failed, format!("sandbox: {e}")),
    };
    let tools = super::tools::specs_for(WORKER_TOOL_PROFILE, false, sandbox.shell_mode());

    let mut reporter = CaptureReporter::default();
    // Defense in depth behind the read-only profile: a worker is unattended, so
    // any action which somehow reaches approval is denied as well.
    let mut approver = NonInteractiveApprover;
    let cancel = AtomicBool::new(false);
    let cfg = agent::AgentConfig {
        workdir: root.to_path_buf(),
        max_steps,
        auto_approve: false,
        yolo: false,
        allow_net: false,
        allow_fs: false,
        shell_timeout: Duration::from_secs(60),
        max_tokens,
        temperature: 0.0,
        audit: Box::new(super::audit::NoopSink),
        shell_sandbox: shell_mode,
        tool_profile: WORKER_TOOL_PROFILE,
        // Keep the exact effective parent budget: children share the same
        // resident model and server process, so neither ceiling may drift.
        ctx_budget: Some(task.context_budget_tokens),
    };
    // Do not inherit write auto-approval into another process: checkpoint state
    // is process-local, so the parent must remain the only writer.
    let mut policy = super::tools::ApprovalPolicy::default();
    // A subagent does real work in the user's workspace, so it gets the same
    // project context its parent has. (The gate harnesses in agent_eval.rs and
    // agent_orchestration.rs deliberately do not — see D-DROVER-6.)
    let project = agent::load_project_context(&sandbox);
    let system_prompt = format!(
        "{}\n\n{WORKER_READ_ONLY_INSTRUCTION}",
        agent::system_prompt_with_project(&sandbox, &tools, project.as_ref())
    );
    let mut history = vec![
        AgentMsg::System(system_prompt),
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
            task.model_sha256.clone(),
            task.family.clone(),
            max_tokens,
            0.0,
        );
        driver.set_context_budget(cfg.ctx_budget);
        driver.set_stream_timeout(agent::AGENT_MODEL_STEP_TIMEOUT);
        driver.set_delta_sink(Some(Box::new(|_| {})));
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
/// Its advertised and validated profile is already workspace-read-only; this
/// approver DENIES every action it is nevertheless consulted for as a second
/// boundary behind profile validation.
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
            model_sha256: "00".repeat(32),
            family: "llama".to_string(),
            workdir: root.display().to_string(),
            max_steps: 4,
            context_budget_tokens: 512,
            max_tokens: 64,
            depth: 1,
            auto_approve: false,
            shell_mode: "sandboxed".to_string(),
            canned_answer: Some("WORKER-OK".to_string()),
            canned_sleep_ms: None,
        }
    }

    #[test]
    fn worker_profile_and_prompt_are_strictly_workspace_read_only() {
        let names = super::super::tools::specs_for(
            WORKER_TOOL_PROFILE,
            true,
            super::super::shell_sandbox::ShellSandbox::Unrestricted,
        )
        .into_iter()
        .map(|tool| tool.name)
        .collect::<Vec<_>>();
        assert_eq!(names, vec!["read_file", "list_dir", "search"]);
        assert!(!WORKER_TOOL_PROFILE.allows("write_file"));
        assert!(!WORKER_TOOL_PROFILE.allows("edit_file"));
        assert!(!WORKER_TOOL_PROFILE.allows("run_shell"));
        assert!(!WORKER_TOOL_PROFILE.allows("http_fetch"));
        assert!(WORKER_READ_ONLY_INSTRUCTION.contains("only the parent process may write"));
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
    fn timed_out_subagent_is_reaped_and_its_task_file_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(subagent_dir(root)).unwrap();
        let tpath = task_path(root, "timeout-cleanup");
        std::fs::write(&tpath, "stale task").unwrap();

        #[cfg(windows)]
        let mut command = {
            let mut command = std::process::Command::new("cmd.exe");
            command.args(["/C", "ping -n 6 127.0.0.1 >NUL"]);
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
            command
        };
        #[cfg(not(windows))]
        let mut command = {
            let mut command = std::process::Command::new("sh");
            command.args(["-c", "sleep 5"]);
            command
        };
        let child = command.spawn().unwrap();

        #[cfg(windows)]
        let job = {
            use std::os::windows::io::AsRawHandle;
            let job = super::super::win_job::JobObject::new().unwrap();
            job.assign(child.as_raw_handle()).unwrap();
            job
        };

        let rpath = result_path(root, "timeout-cleanup");
        let mut state = SessionState {
            config: None,
            children: vec![ChildEntry {
                subtask_id: "timeout-cleanup".into(),
                child,
                #[cfg(windows)]
                job,
                started: Instant::now(),
                timeout: Duration::ZERO,
                result_path: rpath.clone(),
            }],
            generation: 0,
        };
        reap_locked(&mut state);

        assert!(state.children.is_empty());
        assert!(
            !tpath.exists(),
            "timed-out task file must not remain reusable"
        );
        let result: SubagentResult =
            serde_json::from_str(&std::fs::read_to_string(rpath).unwrap()).unwrap();
        assert_eq!(result.status, "inconclusive");
        assert!(result.note.contains("timed out"));
    }

    #[test]
    fn result_publication_is_no_clobber() {
        let dir = tempfile::tempdir().unwrap();
        let state = secure_subagent_dir(dir.path()).unwrap();
        let result = state.join("result_job.json");
        write_result_atomic(&result, "first").unwrap();
        assert!(write_result_atomic(&result, "second").is_err());
        assert_eq!(std::fs::read_to_string(result).unwrap(), "first");
    }

    #[cfg(unix)]
    #[test]
    fn subagent_state_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let task = dir.path().join("task.json");
        create_private_new(&task).unwrap();
        assert_eq!(
            std::fs::metadata(&task).unwrap().permissions().mode() & 0o777,
            0o600
        );

        let result = dir.path().join("result.json");
        write_result_atomic(&result, "result").unwrap();
        assert_eq!(
            std::fs::metadata(&result).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn session_child_timeout_cannot_undercut_one_model_step() {
        let config = SubagentConfig::for_session(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            "model".to_string(),
            "00".repeat(32),
            "llama".to_string(),
            512,
            128,
            false,
            super::super::shell_sandbox::ShellSandbox::Sandboxed,
        );
        assert!(config.timeout > super::super::agent::AGENT_MODEL_STEP_TIMEOUT);
        assert_eq!(config.context_budget_tokens, 512);
        assert_eq!(config.max_tokens, 128);
    }

    #[test]
    fn low_ceiling_parent_budget_is_copied_exactly_into_child_task() {
        let config = SubagentConfig::for_session(
            SocketAddr::from(([127, 0, 0, 1], 8181)),
            "model".to_string(),
            "ab".repeat(32),
            "llama".to_string(),
            320,
            64,
            false,
            super::super::shell_sandbox::ShellSandbox::Sandboxed,
        );
        let root = tempfile::tempdir().unwrap();
        let task = task_from_config(
            &config,
            root.path(),
            "budget-child",
            "inspect",
            1,
            None,
            None,
        );
        assert_eq!(task.context_budget_tokens, 320);
        assert_eq!(task.max_tokens, 64);
        assert_eq!(task.model_sha256, "ab".repeat(32));
    }

    #[test]
    fn worker_rejects_a_child_budget_that_exceeds_or_consumes_context() {
        let root = tempfile::tempdir().unwrap();
        let mut task = canned_task(root.path(), "budget");
        task.context_budget_tokens = 96;
        task.max_tokens = 96;
        let execution = execute_task(&task);
        assert_eq!(execution.outcome, super::super::agent::RunOutcome::Failed);
        assert!(execution.report.note.contains("invalid runtime budget"));

        task.context_budget_tokens = super::super::agent::AGENT_VALIDATED_CTX + 1;
        task.max_tokens = 64;
        let execution = execute_task(&task);
        assert_eq!(execution.outcome, super::super::agent::RunOutcome::Failed);
        assert!(execution.report.note.contains("validated context ceiling"));
    }

    #[cfg(unix)]
    #[test]
    fn state_paths_refuse_symlinks_without_touching_the_target() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), root.path().join(".camelid")).unwrap();
        assert!(secure_subagent_dir(root.path()).is_err());
        assert!(!outside.path().join("subagents").exists());

        let root = tempfile::tempdir().unwrap();
        let state = secure_subagent_dir(root.path()).unwrap();
        let victim = outside.path().join("victim.txt");
        std::fs::write(&victim, "safe").unwrap();
        symlink(&victim, state.join("result_evil.json")).unwrap();
        assert!(status(root.path(), "evil").is_err());
        assert_eq!(std::fs::read_to_string(victim).unwrap(), "safe");
    }

    #[cfg(unix)]
    #[test]
    fn worker_refuses_a_symlink_task_file() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let state = secure_subagent_dir(root.path()).unwrap();
        let real = root.path().join("task.json");
        std::fs::write(
            &real,
            serde_json::to_string(&canned_task(root.path(), "linked")).unwrap(),
        )
        .unwrap();
        let linked = state.join("task_linked.json");
        symlink(real, &linked).unwrap();
        assert!(run_worker(&linked).is_err());
        assert!(!state.join("result_linked.json").exists());
    }

    #[cfg(unix)]
    #[test]
    fn bounded_state_read_refuses_fifo_without_blocking() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("result_fifo.json");
        let path = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        let started = Instant::now();
        assert!(read_regular_bounded(&fifo, MAX_RESULT_BYTES).is_err());
        assert!(started.elapsed() < Duration::from_millis(250));
    }

    #[cfg(unix)]
    #[test]
    fn terminate_child_tree_kills_unix_descendants() {
        use std::os::unix::process::CommandExt;

        let dir = tempfile::tempdir().unwrap();
        let mut command = std::process::Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("sleep 30 & wait")
            .process_group(0)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let child = command.spawn().unwrap();
        let pgid = child.id() as i32;
        let mut entry = ChildEntry {
            subtask_id: "tree-test".to_string(),
            child,
            started: Instant::now(),
            timeout: Duration::from_secs(30),
            result_path: dir.path().join("unused-result.json"),
        };
        std::thread::sleep(Duration::from_millis(100));
        terminate_child_tree(&mut entry);

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let signal_result = unsafe { libc::kill(-pgid, 0) };
            let alive = signal_result == 0
                || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
            if !alive {
                break;
            }
            assert!(Instant::now() < deadline, "process group {pgid} survived");
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    #[cfg(unix)]
    #[test]
    fn completed_subagent_cannot_leave_background_descendants() {
        use std::os::unix::process::CommandExt;

        let dir = tempfile::tempdir().unwrap();
        let mut command = std::process::Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("sleep 30 & exit 0")
            .process_group(0)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let child = command.spawn().unwrap();
        let pgid = child.id() as i32;
        let mut entry = ChildEntry {
            subtask_id: "completed-tree".to_string(),
            child,
            started: Instant::now(),
            timeout: Duration::from_secs(30),
            result_path: dir.path().join("unused-result.json"),
        };
        let exit_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if entry.child.try_wait().unwrap().is_some() {
                break;
            }
            assert!(Instant::now() < exit_deadline, "shell did not exit");
            std::thread::sleep(Duration::from_millis(10));
        }
        kill_child_descendants(&entry);

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let signal_result = unsafe { libc::kill(-pgid, 0) };
            let alive = signal_result == 0
                || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
            if !alive {
                break;
            }
            assert!(Instant::now() < deadline, "process group {pgid} survived");
            std::thread::sleep(Duration::from_millis(25));
        }
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
