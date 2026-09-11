// Sidecar lifecycle for the camelid engine.
//
// The desktop app does NOT reimplement any inference, tokenization, or HTTP surface. It
// spawns the shipped `camelid` server binary as a loopback-only sidecar, health-gates it,
// then points the WebView at the sidecar's already-embedded UI. Behavior is therefore
// byte-identical to running `camelid serve` + the web UI manually. See DECISIONS.md D11.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::lifetime::{
    parse_health_body, restart_transition, Containment, EngineSnapshot, EngineStatus, ExitSummary,
    HealthObservation, ProbeResult, StatusStore,
};

/// Resolved engine binary stem. The crate/binary is `camelid`; the legacy
/// `backendinference` name must never be reintroduced (DECISIONS.md D2). Everything that
/// names the engine references THIS constant rather than scattering string literals.
pub const ENGINE_BINARY_STEM: &str = "camelid";

/// Platform file name for the engine binary, derived from [`ENGINE_BINARY_STEM`] so the
/// resolved name has exactly one source of truth (never a scattered literal).
pub fn engine_binary_file() -> String {
    if cfg!(windows) {
        format!("{ENGINE_BINARY_STEM}.exe")
    } else {
        ENGINE_BINARY_STEM.to_string()
    }
}

/// Health-gate budget: poll `/v1/health` for up to this long before declaring failure.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(40);
/// Backoff between health polls (model load can dominate; keep polls cheap and patient).
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(350);
/// Connect and read budget for the tray's health probe of a running engine.
const PROBE_TIMEOUT: Duration = Duration::from_millis(1500);
const HEALTH_RESPONSE_LIMIT: u64 = 64 * 1024;
/// How much engine stderr is kept once the splash no longer owns the pipe.
const STDERR_TAIL_BYTES: usize = 16 * 1024;
const FEATURE_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Optional engine flag (D6). Passed only when the engine's own `serve --help` lists it, so
/// an older engine, or one found on PATH, still starts exactly as before.
pub const EXIT_WHEN_STDIN_CLOSES_FLAG: &str = "--exit-when-stdin-closes";

/// Stable startup-failure classes rendered by the bundled splash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineErrorKind {
    MissingBinary,
    PortUnavailable,
    StartupTimeout,
    StartupFailed,
}

/// A fatal error during sidecar startup, carrying any captured engine stderr so the splash
/// can surface the *real* failure rather than a fake "ready" state.
#[derive(Debug)]
pub struct EngineError {
    kind: EngineErrorKind,
    pub message: String,
    pub stderr: Option<String>,
}

impl EngineError {
    fn new(kind: EngineErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            stderr: None,
        }
    }
    fn with_stderr(
        kind: EngineErrorKind,
        message: impl Into<String>,
        stderr: Option<String>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            stderr,
        }
    }
    /// Stable summary rendered prominently on the splash after sidecar startup fails.
    pub fn splash_title(&self) -> &'static str {
        match self.kind {
            EngineErrorKind::MissingBinary => "Camelid engine is missing",
            EngineErrorKind::PortUnavailable => "Sidecar port unavailable",
            EngineErrorKind::StartupTimeout => "Engine startup timed out",
            EngineErrorKind::StartupFailed => "Engine startup failed",
        }
    }

    /// Actionable next step paired with [`Self::splash_title`] on the splash.
    pub fn splash_guidance(&self) -> &'static str {
        match self.kind {
            EngineErrorKind::MissingBinary => {
                "Reinstall Camelid Desktop or restore its bundled Camelid engine, then retry."
            }
            EngineErrorKind::PortUnavailable => {
                "Another local process claimed Camelid's selected loopback port. Close it and retry."
            }
            EngineErrorKind::StartupTimeout => {
                "The engine did not pass its 40-second health check. Retry, then review the technical details if it persists."
            }
            EngineErrorKind::StartupFailed => {
                "The engine stopped before it became healthy. Review the technical details, then retry."
            }
        }
    }

    /// Human-readable detail block for the splash error pane.
    pub fn detail(&self) -> String {
        match &self.stderr {
            Some(s) if !s.trim().is_empty() => {
                format!("{}\n\n--- engine stderr ---\n{}", self.message, s.trim())
            }
            _ => self.message.clone(),
        }
    }
}

/// The loopback base URL the WebView navigates to. UI and API are same-origin.
pub fn base_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/")
}

/// A spawned sidecar that has not yet passed its health gate.
pub struct Launched {
    child: Child,
    containment: Containment,
    stdin: Option<ChildStdin>,
    #[cfg(windows)]
    job: Option<JobObject>,
}

/// A running sidecar plus the loopback port it bound. Dropping/`shutdown` kills the child.
pub struct Engine {
    child: Child,
    port: u16,
    containment: Containment,
    // Never written. Holding it open is the whole point: the OS closes it when this process
    // dies, and an engine started with --exit-when-stdin-closes then exits by itself.
    _stdin: Option<ChildStdin>,
    stderr: Option<ChildStderr>,
    tail: Arc<StderrTail>,
    #[cfg(windows)]
    _job: Option<JobObject>,
}

impl Engine {
    /// Promotes a sidecar that passed its health gate. From here on nothing else reads its
    /// stderr, so the pipe is drained into a bounded tail.
    fn adopt(launched: Launched, port: u16) -> Engine {
        let Launched {
            mut child,
            containment,
            stdin,
            #[cfg(windows)]
            job,
        } = launched;
        let stderr = child.stderr.take();
        let mut engine = Engine {
            child,
            port,
            containment,
            _stdin: stdin,
            stderr,
            tail: Arc::new(StderrTail::default()),
            #[cfg(windows)]
            _job: job,
        };
        engine.attach_stderr_drain();
        engine
    }

    /// An engine that logs more than one pipe buffer (~64 KiB) of stderr would otherwise
    /// block inside its next write, mid-request, for as long as the app stays open.
    fn attach_stderr_drain(&mut self) {
        let Some(stderr) = self.stderr.take() else {
            return;
        };
        let tail = Arc::clone(&self.tail);
        let spawned = std::thread::Builder::new()
            .name("engine-stderr".into())
            .spawn(move || tail.drain(stderr));
        if let Err(err) = spawned {
            eprintln!("[desktop] could not start the engine stderr reader: {err}");
        }
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Non-blocking: `try_wait` never waits on a live child.
    pub fn poll_exit(&mut self) -> Option<ExitStatus> {
        self.child.try_wait().ok().flatten()
    }

    /// The last non-empty line the engine wrote to stderr, if any.
    pub fn stderr_tail(&self) -> Option<String> {
        self.tail.last_line()
    }

    /// Graceful-ish shutdown. On Windows there is no SIGTERM; the child is loopback-only and
    /// holds no external state, so `TerminateProcess` (via `Child::kill`) is the clean stop.
    /// The kill-on-close job object (set in `launch_contained`) is the backstop if the
    /// parent crashes.
    pub fn shutdown(&mut self) {
        let _ = self.stop();
    }

    fn stop(&mut self) -> Option<ExitStatus> {
        let _ = self.child.kill();
        self.child.wait().ok()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Default)]
struct StderrTail(Mutex<Vec<u8>>);

impl StderrTail {
    fn drain(&self, mut stderr: ChildStderr) {
        let mut chunk = [0u8; 8192];
        loop {
            match stderr.read(&mut chunk) {
                Ok(0) => return,
                Ok(n) => self.push(&chunk[..n]),
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => return,
            }
        }
    }

    fn push(&self, bytes: &[u8]) {
        let mut buf = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        buf.extend_from_slice(bytes);
        if buf.len() > STDERR_TAIL_BYTES {
            let excess = buf.len() - STDERR_TAIL_BYTES;
            buf.drain(..excess);
        }
    }

    fn last_line(&self) -> Option<String> {
        let buf = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        String::from_utf8_lossy(&buf)
            .lines()
            .rev()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(str::to_string)
    }
}

pub fn exit_summary(status: &ExitStatus) -> ExitSummary {
    if let Some(code) = status.code() {
        return ExitSummary::Code(code);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return ExitSummary::Signal(signal);
        }
    }
    ExitSummary::Unknown
}

/// Locate the `camelid` engine binary. Resolution order:
/// 1. Beside the desktop executable (the bundled/portable case — and the dev case, since both
///    workspace binaries land in `target/<profile>/`).
/// 2. An explicit `resource_dir/sidecar/<platform engine>` (Tauri-bundled resource layout).
/// 3. Bare name on `PATH` (developer convenience).
pub fn resolve_engine_path(resource_dir: Option<PathBuf>) -> Result<PathBuf, EngineError> {
    let file = engine_binary_file();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join(&file);
            if beside.is_file() {
                return Ok(simplify_extended_path(beside));
            }
        }
    }
    if let Some(res) = resource_dir {
        let bundled = res.join("sidecar").join(&file);
        if bundled.is_file() {
            return Ok(simplify_extended_path(bundled));
        }
    }
    // Fall back to PATH resolution by bare name; spawn will surface a clear error if absent.
    Ok(PathBuf::from(file))
}

/// Strip the Windows `\\?\` extended-length ("verbatim") prefix from a path.
///
/// Tauri's `resource_dir()` can return a verbatim path (`\\?\C:\...`). Win32 does
/// NOT separator-normalize verbatim paths, so once such a path becomes the
/// sidecar's `--models-dir` and the web UI joins it with a forward slash
/// (`<models_dir>/<file>.gguf`) the result is an invalid name and the engine's
/// open fails with os error 123. Handing the sidecar a normal path avoids that.
/// The engine binary itself is derived from the same resolved path, so stripping
/// here keeps every downstream path (models dir, catalog downloads) normalized.
///
/// No-op off Windows, and for anything that is not a plain drive (`C:\...`) or UNC
/// verbatim path (`\\?\UNC\server\share`).
#[cfg(windows)]
fn simplify_extended_path(p: PathBuf) -> PathBuf {
    let Some(s) = p.to_str() else { return p };
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        let bytes = rest.as_bytes();
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            return PathBuf::from(rest);
        }
    }
    p
}

#[cfg(not(windows))]
fn simplify_extended_path(p: PathBuf) -> PathBuf {
    p
}

/// Reserve an OS-assigned ephemeral port on loopback. We bind, read the assigned port, then
/// release it and hand the number to the sidecar. There is a small TOCTOU window between
/// release and the sidecar re-binding; this is the standard trade-off and is handled by the
/// health gate failing loudly (never silently) if the bind races.
pub fn pick_ephemeral_port() -> Result<u16, EngineError> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| {
        EngineError::new(
            EngineErrorKind::PortUnavailable,
            format!("could not reserve a loopback port: {e}"),
        )
    })?;
    let port = listener
        .local_addr()
        .map_err(|e| {
            EngineError::new(
                EngineErrorKind::PortUnavailable,
                format!("could not read reserved port: {e}"),
            )
        })?
        .port();
    drop(listener);
    Ok(port)
}

/// Optional flags the resolved engine advertises in `serve --help`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngineFeatures {
    pub exit_when_stdin_closes: bool,
}

impl EngineFeatures {
    /// Stdin is piped only together with the flag that makes the engine watch it, so a held
    /// pipe always means a crash backstop.
    pub fn pipes_stdin(&self) -> bool {
        self.exit_when_stdin_closes
    }
}

/// A flag counts only as a whole token, so a longer flag that merely starts with the same
/// text is not mistaken for it.
pub fn parse_advertised_flags(help: &str) -> EngineFeatures {
    let advertised = |flag: &str| {
        help.split(|c: char| c.is_whitespace() || matches!(c, ',' | '[' | ']' | '=' | '<' | '>'))
            .any(|token| token == flag)
    };
    EngineFeatures {
        exit_when_stdin_closes: advertised(EXIT_WHEN_STDIN_CLOSES_FLAG),
    }
}

/// Runs `<engine> serve --help`. Clap answers before the engine does any real work, so this
/// costs one short process start.
pub fn probe_engine_features(engine_path: &Path) -> Result<EngineFeatures, String> {
    let mut command = Command::new(engine_path);
    command.arg("serve").arg("--help");
    probe_features_with(command, FEATURE_PROBE_TIMEOUT)
}

fn probe_features_with(mut command: Command, timeout: Duration) -> Result<EngineFeatures, String> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    no_console_window(&mut command);
    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|err| format!("could not run `serve --help`: {err}"))?;
    let stdout = child.stdout.take();
    let reader = std::thread::spawn(move || {
        let mut text = Vec::new();
        if let Some(stdout) = stdout {
            let _ = stdout.take(256 * 1024).read_to_end(&mut text);
        }
        text
    });
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "`serve --help` did not finish within {:.1} s",
                    timeout.as_secs_f32()
                ));
            }
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("could not wait for `serve --help`: {err}"));
            }
        }
    };
    let text = reader.join().unwrap_or_default();
    if !status.success() {
        return Err(format!("`serve --help` exited with {status}"));
    }
    Ok(parse_advertised_flags(&String::from_utf8_lossy(&text)))
}

/// A failed or slow probe starts the engine with no optional flags: an unknown flag would
/// make clap reject every launch.
pub fn features_or_none(probe: Result<EngineFeatures, String>) -> (EngineFeatures, String) {
    match probe {
        Ok(features) => (
            features,
            format!(
                "[desktop] engine probe: exit-when-stdin-closes {}",
                if features.exit_when_stdin_closes {
                    "advertised"
                } else {
                    "not advertised"
                }
            ),
        ),
        Err(reason) => (
            EngineFeatures::default(),
            format!("[desktop] engine probe: {reason}; starting without optional flags"),
        ),
    }
}

/// `camelid serve --addr 127.0.0.1:<port> --no-open --models-dir <abs>`, bound to loopback
/// only, plus the optional flags the engine advertised. An explicit models directory lets
/// packaged platforms keep mutable model data outside the signed application bundle; Windows
/// retains its existing engine-adjacent directory when no override is supplied.
pub fn build_serve_command(
    engine_path: &Path,
    port: u16,
    models_dir: Option<&Path>,
    features: &EngineFeatures,
) -> Command {
    let addr = format!("127.0.0.1:{port}");

    let mut command = Command::new(engine_path);
    command
        .arg("serve")
        .arg("--addr")
        .arg(&addr)
        .arg("--no-open");
    // The desktop's model store is the `models/` folder beside the engine binary (the
    // installed layout: camelid.exe and models/ side by side; NSIS upgrades preserve that
    // folder). The sidecar inherits an arbitrary launch-context CWD (Explorer, a shortcut,
    // the shell), so pin the directory explicitly as an ABSOLUTE path — otherwise the
    // engine's local-models scan, catalog downloads, and relative /api/models/load paths
    // resolve against whatever directory Windows happened to launch the app from.
    let models_dir = models_dir
        .map(Path::to_path_buf)
        .or_else(|| sidecar_models_dir(engine_path));
    if let Some(models_dir) = models_dir {
        command.arg("--models-dir").arg(models_dir);
    }
    if features.exit_when_stdin_closes {
        command.arg(EXIT_WHEN_STDIN_CLOSES_FLAG);
    }
    command
        .stdin(if features.pipes_stdin() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    no_console_window(&mut command);
    command
}

/// Spawns the sidecar and reports how it is contained. On Windows it goes into a
/// kill-on-close job, so a desktop crash takes it down too; a failed assignment is reported
/// as uncontained rather than assumed.
pub fn launch_contained(mut command: Command) -> Result<Launched, EngineError> {
    let mut child = command.spawn().map_err(|e| {
        let kind = if e.kind() == std::io::ErrorKind::NotFound {
            EngineErrorKind::MissingBinary
        } else {
            EngineErrorKind::StartupFailed
        };
        EngineError::new(
            kind,
            format!(
                "failed to launch the camelid engine at {}: {e}",
                Path::new(command.get_program()).display()
            ),
        )
    })?;
    let stdin = child.stdin.take();

    #[cfg(windows)]
    let job = match JobObject::assign(&child) {
        Ok(job) => Some(job),
        Err(err) => {
            eprintln!("[desktop] lifetime: the engine is not in a kill-on-close job: {err}");
            None
        }
    };
    #[cfg(windows)]
    let containment = if job.is_some() {
        Containment::JobObject
    } else if stdin.is_some() {
        Containment::StdinPipe
    } else {
        Containment::Uncontained
    };
    #[cfg(not(windows))]
    let containment = if stdin.is_some() {
        Containment::StdinPipe
    } else {
        Containment::Uncontained
    };

    Ok(Launched {
        child,
        containment,
        stdin,
        #[cfg(windows)]
        job,
    })
}

/// One-shot startup outside the engine slot: the path the splash-contract tests exercise.
/// The app starts through [`EngineHost::start`], which shares every step below.
#[cfg_attr(not(test), allow(dead_code))]
pub fn spawn(engine_path: &Path, models_dir: Option<&Path>) -> Result<Engine, EngineError> {
    let port = pick_ephemeral_port()?;
    let command = build_serve_command(engine_path, port, models_dir, &EngineFeatures::default());
    let mut launched = launch_contained(command)?;
    match wait_for_health_with_timeout(
        port,
        &mut launched.child,
        HEALTH_TIMEOUT,
        HEALTH_POLL_INTERVAL,
    ) {
        Ok(()) => Ok(Engine::adopt(launched, port)),
        Err(err) => Err(finish_startup_failure(&mut launched.child, err)),
    }
}

/// The desktop's models directory: the `models/` folder beside the engine binary, as an
/// absolute path. `None` when the engine was resolved as a bare `PATH` name (the developer
/// fallback in `resolve_engine_path`), where the engine's own exe-relative default is the
/// right authority. The directory deliberately does NOT need to exist yet — on a fresh
/// install it is created by the engine's first catalog download.
fn sidecar_models_dir(engine_path: &Path) -> Option<PathBuf> {
    let parent = engine_path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())?;
    let models_dir = parent.join("models");
    if models_dir.is_absolute() {
        Some(models_dir)
    } else {
        std::path::absolute(&models_dir).ok()
    }
}

enum ChildPoll {
    Alive,
    Exited(ExitStatus),
    /// The sidecar is no longer this gate's to wait for: it was reaped or replaced.
    Cancelled,
}

enum GateStop {
    Failed(EngineError),
    Cancelled,
}

/// Poll `/v1/health` until it returns 200, the engine exits, or the budget elapses.
fn health_gate(
    port: u16,
    timeout: Duration,
    poll_interval: Duration,
    mut poll_child: impl FnMut() -> ChildPoll,
) -> Result<(), GateStop> {
    let deadline = Instant::now() + timeout;
    loop {
        // If the engine already exited, fail immediately with its status (stderr added later).
        match poll_child() {
            ChildPoll::Alive => {}
            ChildPoll::Cancelled => return Err(GateStop::Cancelled),
            ChildPoll::Exited(status) => {
                return Err(GateStop::Failed(EngineError::new(
                    EngineErrorKind::StartupFailed,
                    format!("the camelid engine exited before becoming healthy (status: {status})"),
                )))
            }
        }
        if http_health_ok(port) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(GateStop::Failed(EngineError::new(
                EngineErrorKind::StartupTimeout,
                format!(
                    "the camelid engine did not report healthy on 127.0.0.1:{port} within {}s",
                    HEALTH_TIMEOUT.as_secs()
                ),
            )));
        }
        std::thread::sleep(poll_interval);
    }
}

/// The gate over a child owned by the caller rather than by the engine slot.
#[cfg_attr(not(test), allow(dead_code))]
fn wait_for_health_with_timeout(
    port: u16,
    child: &mut Child,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<(), EngineError> {
    health_gate(port, timeout, poll_interval, || match child.try_wait() {
        Ok(Some(status)) => ChildPoll::Exited(status),
        _ => ChildPoll::Alive,
    })
    .map_err(|stop| match stop {
        GateStop::Failed(err) => err,
        GateStop::Cancelled => EngineError::new(
            EngineErrorKind::StartupFailed,
            "the engine start was cancelled",
        ),
    })
}

/// Outcome of starting one engine generation.
#[derive(Debug)]
pub enum StartOutcome {
    Ready {
        port: u16,
    },
    Failed(EngineError),
    /// A newer generation, or quit, took over. Nothing is reported for this one.
    Cancelled,
}

/// One supervisor pass: what was probed, if anything, and the slot as observed after it.
pub struct Tick {
    pub probed: Option<(u64, HealthObservation)>,
    pub snapshot: Option<EngineSnapshot>,
}

/// What a shutdown found in the slot and reaped.
#[derive(Debug)]
pub struct Reaped {
    pub pid: u32,
    pub status: Option<ExitStatus>,
    pub was_pending: bool,
}

#[derive(Default)]
enum EngineSlot {
    #[default]
    Empty,
    Pending {
        epoch: u64,
        launched: Launched,
    },
    Ready {
        epoch: u64,
        engine: Engine,
    },
}

/// The one place a sidecar lives, from the moment it is spawned. A sidecar still inside its
/// 40 s health gate is in the slot too, so every shutdown can reap it; before this, a quit
/// during a restart left the half-started engine running on macOS.
#[derive(Default)]
pub struct EngineHost {
    slot: Mutex<EngineSlot>,
    /// Bumped by every start, restart and quit. Work tagged with an older epoch is stale.
    epoch: AtomicU64,
    quitting: AtomicBool,
}

impl EngineHost {
    fn lock_slot(&self) -> MutexGuard<'_, EngineSlot> {
        self.slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Private: the app opens a generation only through `begin_start` or `begin_restart`,
    /// which move the tray's store to it in the same step.
    fn begin_epoch(&self) -> u64 {
        self.epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Opens the first engine generation and publishes Starting for it in the same step, so
    /// the tray's store follows the generation every later status is tagged with.
    pub fn begin_start(&self, store: &mut StatusStore) -> u64 {
        let epoch = self.begin_epoch();
        store.begin(epoch, EngineStatus::Starting);
        epoch
    }

    /// Opens the generation that replaces the current engine: publishes Restarting for it,
    /// then reaps the old engine, pending or ready. The caller holds the tray's store for the
    /// whole call, so nothing reads the old port as running in between, and the new engine's
    /// statuses land in the generation the store now follows. `None` once quit has begun.
    pub fn begin_restart(&self, store: &mut StatusStore) -> Option<u64> {
        if self.is_quitting() {
            return None;
        }
        let epoch = self.begin_epoch();
        restart_transition(store, epoch);
        self.reap();
        Some(epoch)
    }

    pub fn is_current(&self, epoch: u64) -> bool {
        !self.quitting.load(Ordering::SeqCst) && self.epoch.load(Ordering::SeqCst) == epoch
    }

    pub fn is_quitting(&self) -> bool {
        self.quitting.load(Ordering::SeqCst)
    }

    /// Resolve nothing, assume nothing: spawn on a fresh ephemeral port and health-gate.
    pub fn start(
        &self,
        epoch: u64,
        engine_path: &Path,
        models_dir: Option<&Path>,
        features: &EngineFeatures,
    ) -> StartOutcome {
        self.start_with(
            epoch,
            |port| build_serve_command(engine_path, port, models_dir, features),
            HEALTH_TIMEOUT,
            HEALTH_POLL_INTERVAL,
        )
    }

    /// `start` with the command and the gate's budget supplied, so tests drive the path the
    /// app takes with a stand-in engine. The sidecar enters the slot as it is spawned, not
    /// after its gate passes, so a shutdown during the gate reaps it.
    fn start_with(
        &self,
        epoch: u64,
        command_for: impl FnOnce(u16) -> Command,
        timeout: Duration,
        poll_interval: Duration,
    ) -> StartOutcome {
        let port = match pick_ephemeral_port() {
            Ok(port) => port,
            Err(err) => return StartOutcome::Failed(err),
        };
        match self.launch(epoch, command_for(port)) {
            Ok(true) => {}
            Ok(false) => return StartOutcome::Cancelled,
            Err(err) => return StartOutcome::Failed(err),
        }
        self.gate_pending(epoch, port, timeout, poll_interval)
    }

    /// Spawns while holding the slot lock, after checking that this generation is still
    /// wanted, so a quit can never slip in between the spawn and the child being reapable.
    /// `Ok(false)` means the launch was superseded and nothing was spawned.
    fn launch(&self, epoch: u64, command: Command) -> Result<bool, EngineError> {
        let mut slot = self.lock_slot();
        if !self.is_current(epoch) {
            return Ok(false);
        }
        let launched = launch_contained(command)?;
        let previous = std::mem::replace(&mut *slot, EngineSlot::Pending { epoch, launched });
        drop(slot);
        reap_slot(previous);
        Ok(true)
    }

    /// The health gate for the pending sidecar of `epoch`. The slot is locked only for each
    /// `try_wait`, never across the HTTP probe, so a shutdown is never blocked by the gate.
    fn gate_pending(
        &self,
        epoch: u64,
        port: u16,
        timeout: Duration,
        poll_interval: Duration,
    ) -> StartOutcome {
        let gate = health_gate(port, timeout, poll_interval, || {
            let mut slot = self.lock_slot();
            match &mut *slot {
                EngineSlot::Pending {
                    epoch: pending,
                    launched,
                } if *pending == epoch => match launched.child.try_wait() {
                    Ok(Some(status)) => ChildPoll::Exited(status),
                    _ => ChildPoll::Alive,
                },
                _ => ChildPoll::Cancelled,
            }
        });

        let mut slot = self.lock_slot();
        let mut launched = match std::mem::take(&mut *slot) {
            EngineSlot::Pending {
                epoch: pending,
                launched,
            } if pending == epoch => launched,
            other => {
                *slot = other;
                return StartOutcome::Cancelled;
            }
        };
        match gate {
            Ok(()) => {
                *slot = EngineSlot::Ready {
                    epoch,
                    engine: Engine::adopt(launched, port),
                };
                StartOutcome::Ready { port }
            }
            Err(GateStop::Failed(err)) => {
                drop(slot);
                StartOutcome::Failed(finish_startup_failure(&mut launched.child, err))
            }
            Err(GateStop::Cancelled) => {
                drop(slot);
                reap_launched(launched);
                StartOutcome::Cancelled
            }
        }
    }

    /// The running engine, observed without blocking. A sidecar still in its health gate is
    /// not reported: the start path owns its status until it passes or fails.
    pub fn observe(&self) -> Option<EngineSnapshot> {
        let mut slot = self.lock_slot();
        match &mut *slot {
            EngineSlot::Ready { epoch, engine } => Some(EngineSnapshot {
                epoch: *epoch,
                port: engine.port,
                pid: engine.pid(),
                exit: engine.poll_exit().map(|status| exit_summary(&status)),
                last_stderr_line: engine.stderr_tail(),
            }),
            _ => None,
        }
    }

    /// Probes `/v1/health` when `due` says so, then observes the slot. The snapshot returned
    /// is taken after the probe, never before it: a probe can take seconds, and an engine
    /// that died during it must not be published as the live engine it was when it began.
    pub fn supervisor_tick(
        &self,
        due: impl FnOnce(&EngineSnapshot) -> bool,
        probe: impl FnOnce(u16) -> HealthObservation,
    ) -> Tick {
        let before = self.observe();
        let probed = before
            .as_ref()
            .filter(|engine| engine.exit.is_none() && due(engine))
            .map(|engine| (engine.epoch, probe(engine.port)));
        let snapshot = if probed.is_some() {
            self.observe()
        } else {
            before
        };
        Tick { probed, snapshot }
    }

    pub fn containment(&self) -> Option<Containment> {
        match &*self.lock_slot() {
            EngineSlot::Empty => None,
            EngineSlot::Pending { launched, .. } => Some(launched.containment),
            EngineSlot::Ready { engine, .. } => Some(engine.containment),
        }
    }

    /// Kills and waits on whatever the slot holds, pending or ready. Private: the app reaps
    /// only through `begin_restart` and `shutdown_for_exit`.
    fn reap(&self) -> Option<Reaped> {
        let taken = std::mem::take(&mut *self.lock_slot());
        reap_slot(taken)
    }

    /// The exit path. After this no engine can start, and any gate still running is
    /// cancelled.
    pub fn shutdown_for_exit(&self) -> Option<Reaped> {
        self.quitting.store(true, Ordering::SeqCst);
        self.epoch.fetch_add(1, Ordering::SeqCst);
        self.reap()
    }
}

fn reap_launched(mut launched: Launched) -> Reaped {
    let pid = launched.child.id();
    let _ = launched.child.kill();
    let status = launched.child.wait().ok();
    Reaped {
        pid,
        status,
        was_pending: true,
    }
}

fn reap_slot(slot: EngineSlot) -> Option<Reaped> {
    match slot {
        EngineSlot::Empty => None,
        EngineSlot::Pending { launched, .. } => Some(reap_launched(launched)),
        EngineSlot::Ready { mut engine, .. } => {
            let pid = engine.pid();
            let status = engine.stop();
            Some(Reaped {
                pid,
                status,
                was_pending: false,
            })
        }
    }
}

/// Recognize only explicit OS bind diagnostics captured from the sidecar.
///
/// The ephemeral reservation must be released before the sidecar binds. A listener observed
/// *after* a generic sidecar exit is not proof that it caused the exit, so it must not control
/// the user-facing diagnosis. The engine's own bind failure is the authority instead.
fn stderr_is_bind_failure(stderr: Option<&str>) -> bool {
    let Some(stderr) = stderr else {
        return false;
    };
    let stderr = stderr.to_ascii_lowercase();
    stderr.contains("address already in use")
        || stderr.contains("only one usage of each socket address")
        || stderr.contains("os error 98")
        || stderr.contains("os error 10048")
}

/// Dependency-free loopback HTTP/1.1 GET of `/v1/health`; true iff the status line is 200.
fn http_health_ok(port: u16) -> bool {
    let addr: SocketAddr = match format!("127.0.0.1:{port}").parse() {
        Ok(a) => a,
        Err(_) => return false,
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(750)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(2000)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(2000)));
    let req =
        format!("GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 256];
    match stream.read(&mut buf) {
        Ok(n) if n > 0 => {
            let head = String::from_utf8_lossy(&buf[..n]);
            head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
        }
        _ => false,
    }
}

/// The tray's probe of a running engine: a 200 with its body read as a model observation,
/// or a failure. Any non-200, timeout or I/O error is a failure, never an answer.
pub fn fetch_health(port: u16) -> HealthObservation {
    let result = match http_get_health(port) {
        Some((200, body)) => ProbeResult::Answered(parse_health_body(&body)),
        _ => ProbeResult::Failed,
    };
    HealthObservation {
        at: Instant::now(),
        result,
    }
}

fn http_get_health(port: u16) -> Option<(u16, String)> {
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().ok()?;
    let mut stream = TcpStream::connect_timeout(&addr, PROBE_TIMEOUT).ok()?;
    stream.set_read_timeout(Some(PROBE_TIMEOUT)).ok()?;
    stream.set_write_timeout(Some(PROBE_TIMEOUT)).ok()?;
    let request =
        format!("GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = Vec::new();
    stream
        .take(HEALTH_RESPONSE_LIMIT)
        .read_to_end(&mut response)
        .ok()?;
    let text = String::from_utf8_lossy(&response);
    let (head, body) = text.split_once("\r\n\r\n")?;
    let code = head.split_whitespace().nth(1)?.parse().ok()?;
    Some((code, body.to_string()))
}

/// Best-effort read of engine stderr after the child has been reaped.
fn drain_stderr(child: &mut Child) -> Option<String> {
    let mut stderr = child.stderr.take()?;
    let mut buf = String::new();
    let _ = stderr.read_to_string(&mut buf);
    if buf.trim().is_empty() {
        None
    } else {
        Some(buf)
    }
}

fn terminate_and_collect_stderr(child: &mut Child) -> Option<String> {
    let _ = child.kill();
    let _ = child.wait();
    drain_stderr(child)
}

fn finish_startup_failure(child: &mut Child, error: EngineError) -> EngineError {
    // A timed-out child still owns the stderr pipe. Reap it before a blocking read so the
    // startup worker cannot get stuck and leave the splash spinner visible indefinitely.
    let stderr = terminate_and_collect_stderr(child);
    let EngineError {
        mut kind,
        message,
        stderr: prior_stderr,
    } = error;
    let stderr = stderr.or(prior_stderr);
    if kind == EngineErrorKind::StartupFailed && stderr_is_bind_failure(stderr.as_deref()) {
        kind = EngineErrorKind::PortUnavailable;
    }
    EngineError::with_stderr(kind, message, stderr)
}

/// Suppress the spawned engine's console window on Windows (CREATE_NO_WINDOW).
#[cfg(windows)]
fn no_console_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn no_console_window(_command: &mut Command) {}

// ---------------------------------------------------------------------------------------
// Windows Job Object: kill-on-close backstop so a desktop crash can't orphan the sidecar.
// ---------------------------------------------------------------------------------------
#[cfg(windows)]
struct JobObject {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

// The job handle is just a kernel object handle; closing it (the only thing we do, on Drop)
// is valid from any thread. Sending it across threads is sound, so the engine can live in
// Tauri's `Send + Sync` managed state.
#[cfg(windows)]
unsafe impl Send for JobObject {}

#[cfg(windows)]
impl JobObject {
    fn assign(child: &Child) -> Result<Self, std::io::Error> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };

        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                let err = std::io::Error::last_os_error();
                windows_sys::Win32::Foundation::CloseHandle(job);
                return Err(err);
            }
            let proc_handle = child.as_raw_handle() as HANDLE;
            if AssignProcessToJobObject(job, proc_handle) == 0 {
                let err = std::io::Error::last_os_error();
                windows_sys::Win32::Foundation::CloseHandle(job);
                return Err(err);
            }
            Ok(JobObject { handle: job })
        }
    }
}

#[cfg(windows)]
impl Drop for JobObject {
    fn drop(&mut self) {
        // Closing the last handle to the job triggers KILL_ON_JOB_CLOSE for the sidecar.
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        finish_startup_failure, sidecar_models_dir, spawn, stderr_is_bind_failure,
        terminate_and_collect_stderr, wait_for_health_with_timeout, EngineError, EngineErrorKind,
    };
    use std::net::TcpListener;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    #[test]
    fn splash_failure_contract_is_actionable() {
        let cases = [
            (
                EngineErrorKind::MissingBinary,
                "Camelid engine is missing",
                "restore its bundled Camelid engine",
            ),
            (
                EngineErrorKind::PortUnavailable,
                "Sidecar port unavailable",
                "selected loopback port",
            ),
            (
                EngineErrorKind::StartupTimeout,
                "Engine startup timed out",
                "40-second health check",
            ),
            (
                EngineErrorKind::StartupFailed,
                "Engine startup failed",
                "stopped before it became healthy",
            ),
        ];

        for (kind, title, guidance_fragment) in cases {
            let error = EngineError::new(kind, "technical detail");
            assert_eq!(error.splash_title(), title);
            assert!(error.splash_guidance().contains(guidance_fragment));
        }
    }

    #[test]
    fn missing_sidecar_is_classified_at_spawn() {
        let missing = std::env::temp_dir().join(format!(
            "camelid-desktop-missing-sidecar-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&missing);

        let error = match spawn(&missing, None) {
            Ok(_) => panic!("a missing sidecar unexpectedly launched"),
            Err(error) => error,
        };

        assert_eq!(error.kind, EngineErrorKind::MissingBinary);
        assert_eq!(error.splash_title(), "Camelid engine is missing");
    }

    #[test]
    fn captured_bind_diagnostic_is_required_for_port_classification() {
        assert!(stderr_is_bind_failure(Some(
            "failed to bind 127.0.0.1:1234: Address already in use (os error 98)"
        )));
        assert!(stderr_is_bind_failure(Some(
            "Only one usage of each socket address is normally permitted. (os error 10048)"
        )));
        assert!(!stderr_is_bind_failure(Some(
            "model metadata is invalid; sidecar stopped before binding"
        )));
        assert!(!stderr_is_bind_failure(None));
    }

    #[test]
    fn unhealthy_live_child_is_reaped_before_stderr_is_drained() {
        let mut command = long_lived_stderr_command("sidecar remains alive");
        command.stdout(Stdio::null()).stderr(Stdio::piped());
        let mut child = command.spawn().expect("start test sidecar");
        assert!(child.try_wait().expect("query child state").is_none());

        let started = Instant::now();
        let error = wait_for_health_with_timeout(
            unused_loopback_port(),
            &mut child,
            Duration::from_millis(100),
            Duration::from_millis(5),
        )
        .expect_err("live child without health endpoint must time out");
        assert_eq!(error.kind, EngineErrorKind::StartupTimeout);

        let stderr = terminate_and_collect_stderr(&mut child).expect("captured test stderr");
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(stderr.contains("sidecar remains alive"));
        assert!(child.try_wait().expect("query reaped child").is_some());
    }

    #[test]
    fn only_captured_bind_diagnostics_change_the_final_error_kind() {
        let mut bind_command = long_lived_stderr_command("address already in use");
        bind_command.stdout(Stdio::null()).stderr(Stdio::piped());
        let mut bind_child = bind_command.spawn().expect("start bind-failure test child");
        std::thread::sleep(Duration::from_millis(100));

        let bind_error = finish_startup_failure(
            &mut bind_child,
            EngineError::new(EngineErrorKind::StartupFailed, "sidecar exited"),
        );
        assert_eq!(
            bind_error.kind,
            EngineErrorKind::PortUnavailable,
            "unexpected captured stderr: {:?}",
            bind_error.stderr
        );

        let mut generic_command = long_lived_stderr_command("model metadata is invalid");
        generic_command.stdout(Stdio::null()).stderr(Stdio::piped());
        let mut generic_child = generic_command
            .spawn()
            .expect("start generic-failure test child");
        std::thread::sleep(Duration::from_millis(100));

        let generic_error = finish_startup_failure(
            &mut generic_child,
            EngineError::new(EngineErrorKind::StartupFailed, "sidecar exited"),
        );
        assert_eq!(generic_error.kind, EngineErrorKind::StartupFailed);
    }

    fn unused_loopback_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
        let port = listener.local_addr().expect("read reserved port").port();
        drop(listener);
        port
    }

    #[cfg(windows)]
    fn long_lived_stderr_command(message: &str) -> Command {
        let mut command = Command::new("cmd.exe");
        let script = format!("echo {message} 1>&2 & ping -n 3 127.0.0.1 >nul 2>nul");
        command.args(["/D", "/C", script.as_str()]);
        command
    }

    #[cfg(not(windows))]
    fn long_lived_stderr_command(message: &str) -> Command {
        let mut command = Command::new("sh");
        command.args([
            "-c",
            &format!("printf '%s\\n' '{message}' >&2; exec sleep 30"),
        ]);
        command
    }

    #[test]
    fn models_dir_sits_beside_an_absolute_engine_path() {
        let engine = if cfg!(windows) {
            PathBuf::from(r"C:\Apps\Camelid\camelid.exe")
        } else {
            PathBuf::from("/opt/camelid/camelid")
        };
        let dir = sidecar_models_dir(&engine).expect("absolute engine path yields a models dir");
        assert!(dir.is_absolute());
        assert_eq!(dir, engine.parent().unwrap().join("models"));
    }

    #[test]
    fn bare_path_engine_name_yields_no_models_dir() {
        // The PATH-resolution fallback: no directory to anchor to, so the engine's own
        // exe-relative default must stay in charge.
        assert_eq!(sidecar_models_dir(Path::new("camelid.exe")), None);
    }

    #[cfg(windows)]
    #[test]
    fn verbatim_prefix_is_stripped_so_the_models_dir_stays_a_normal_path() {
        use super::simplify_extended_path;
        // A `\\?\` drive path (what Tauri's resource_dir can hand back) is normalized...
        let engine = simplify_extended_path(PathBuf::from(
            r"\\?\C:\Apps\Camelid Desktop\sidecar\camelid.exe",
        ));
        assert_eq!(
            engine,
            PathBuf::from(r"C:\Apps\Camelid Desktop\sidecar\camelid.exe")
        );
        // ...so the derived models dir carries no verbatim prefix. A later
        // `<models_dir>/<file>` join by the web UI is then a normal (separator-
        // normalized) path, not the os-error-123 verbatim+slash mix.
        let models = sidecar_models_dir(&engine).expect("absolute engine path yields a models dir");
        assert!(!models.to_string_lossy().contains(r"\\?\"));
        assert_eq!(
            models,
            PathBuf::from(r"C:\Apps\Camelid Desktop\sidecar\models")
        );
    }

    #[cfg(windows)]
    #[test]
    fn non_verbatim_paths_pass_through_unchanged() {
        use super::simplify_extended_path;
        let plain = PathBuf::from(r"C:\Apps\Camelid\camelid.exe");
        assert_eq!(simplify_extended_path(plain.clone()), plain);
    }
}

/// Background-lifetime tests (P7). A separate module so the splash-contract tests above stay
/// exactly as they were.
#[cfg(test)]
mod lifetime_tests {
    use super::*;

    const HELPER_ENV: &str = "CAMELID_DESKTOP_ENGINE_TEST_HELPER";

    /// Re-runs this test binary as a stand-in engine. `mode` picks the behaviour; see
    /// `stand_in_engine_process`.
    fn stand_in_engine(mode: &str) -> Command {
        let mut command = Command::new(std::env::current_exe().expect("test binary path"));
        command
            .args([
                "--exact",
                "engine::lifetime_tests::stand_in_engine_process",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(HELPER_ENV, mode)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        command
    }

    /// Not a test: the body of the stand-in engine. Returns at once unless re-executed by
    /// `stand_in_engine`.
    #[test]
    fn stand_in_engine_process() {
        let Ok(mode) = std::env::var(HELPER_ENV) else {
            return;
        };
        if let Some(code) = mode.strip_prefix("exit:") {
            std::process::exit(code.parse().expect("exit code"));
        }
        if let Some(marker) = mode.strip_prefix("chatty:") {
            let mut stderr = std::io::stderr().lock();
            let line = [b'x'; 1023];
            for _ in 0..1024 {
                stderr.write_all(&line).unwrap();
                stderr.write_all(b"\n").unwrap();
            }
            stderr.write_all(b"stand-in engine final line\n").unwrap();
            stderr.flush().unwrap();
            std::fs::write(marker, b"wrote 1 MiB").unwrap();
        }
        if let Some(marker) = mode.strip_prefix("touch:") {
            std::fs::write(marker, b"started").unwrap();
        }
        if let Some(port) = mode.strip_prefix("healthy:") {
            serve_health(port, None);
        }
        if let Some(spec) = mode.strip_prefix("answers:") {
            let (count, port) = spec.split_once(':').expect("answers:<count>:<port>");
            serve_health(port, Some(count.parse().expect("answer count")));
        }
        std::thread::sleep(Duration::from_secs(30));
        std::process::exit(0);
    }

    /// Answers `/v1/health` like an engine with no model ready. With `limit`, exits with code
    /// 3 straight after the last answer: a crash the supervisor has to notice.
    fn serve_health(port: &str, limit: Option<usize>) {
        let listener = TcpListener::bind(format!("127.0.0.1:{port}")).expect("bind");
        let mut served = 0;
        for mut stream in listener.incoming().flatten() {
            let mut request = [0u8; 1024];
            let _ = stream.read(&mut request);
            let body = r#"{"generation_ready":false,"active_model_id":null}"#;
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            drop(stream);
            served += 1;
            if limit == Some(served) {
                std::process::exit(3);
            }
        }
    }

    fn unused_loopback_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
        listener.local_addr().expect("read reserved port").port()
    }

    fn wait_until(budget: Duration, mut done: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + budget;
        while Instant::now() < deadline {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        done()
    }

    #[test]
    fn a_chatty_sidecar_never_blocks_on_a_full_stderr_pipe() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("wrote-everything");
        let launched = launch_contained(stand_in_engine(&format!("chatty:{}", marker.display())))
            .expect("launch the stand-in engine");
        let mut engine = Engine::adopt(launched, 0);

        let finished_writing = wait_until(Duration::from_secs(5), || marker.exists());
        let tail_caught_up = wait_until(Duration::from_secs(2), || {
            engine.stderr_tail().as_deref() == Some("stand-in engine final line")
        });
        let tail = engine.stderr_tail();
        engine.shutdown();

        assert!(
            finished_writing,
            "the engine blocked writing 1 MiB of stderr: nothing drains the pipe"
        );
        assert!(tail_caught_up, "unexpected stderr tail: {tail:?}");
    }

    #[test]
    fn an_exited_sidecar_is_observed_by_poll_exit() {
        let launched = launch_contained(stand_in_engine("exit:7")).expect("launch");
        let mut engine = Engine::adopt(launched, 0);
        let mut status = None;
        wait_until(Duration::from_secs(3), || {
            status = engine.poll_exit();
            status.is_some()
        });
        let status = status.expect("the exit was never observed");
        assert_eq!(status.code(), Some(7));
        assert_eq!(exit_summary(&status), ExitSummary::Code(7));
    }

    /// Through `start_with`, the path `EngineHost::start` takes, with a stand-in engine that
    /// never becomes healthy: a quit mid-gate must find the sidecar in the slot and reap it.
    #[test]
    fn shutdown_during_the_health_gate_reaps_the_child() {
        let host = Arc::new(EngineHost::default());
        let epoch = host.begin_start(&mut StatusStore::default());

        let gate_host = Arc::clone(&host);
        let gate = std::thread::spawn(move || {
            let started = Instant::now();
            let outcome = gate_host.start_with(
                epoch,
                |_| stand_in_engine("never-healthy"),
                Duration::from_secs(20),
                Duration::from_millis(20),
            );
            (outcome, started.elapsed())
        });
        let in_slot = wait_until(Duration::from_secs(5), || host.containment().is_some());

        let reaped = host.shutdown_for_exit();
        assert!(
            in_slot,
            "the starting sidecar never entered the engine slot, so a quit cannot reach it"
        );
        let reaped = reaped.expect("a sidecar inside its health gate must be reapable");
        assert!(reaped.was_pending);
        assert!(reaped.status.is_some(), "the child was not waited on");

        let (outcome, elapsed) = gate.join().expect("gate thread");
        assert!(
            matches!(outcome, StartOutcome::Cancelled),
            "a cancelled gate must not surface an error: {outcome:?}"
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "the gate outlived the quit"
        );
        assert!(host.observe().is_none());
        assert!(host.reap().is_none());
    }

    /// The first start and every restart open the generation the tray's store follows, and a
    /// restart reaps the old engine without the store ever showing its port again.
    #[test]
    fn a_restart_moves_the_tray_to_the_new_engine_and_reaps_the_old_one() {
        let host = EngineHost::default();
        let mut store = StatusStore::default();
        let first = host.begin_start(&mut store);
        assert_eq!(
            (store.epoch(), store.status()),
            (first, &EngineStatus::Starting),
            "the first start must open the generation the tray follows"
        );

        let outcome = host.start_with(
            first,
            |port| stand_in_engine(&format!("healthy:{port}")),
            Duration::from_secs(20),
            Duration::from_millis(50),
        );
        let port = match outcome {
            StartOutcome::Ready { port } => port,
            other => panic!("{other:?}"),
        };
        let old = host.observe().expect("a ready engine is observed");
        let old_running = EngineStatus::Running { port, pid: old.pid };
        assert!(store.apply(first, old_running.clone()));

        let second = host.begin_restart(&mut store).expect("not quitting");
        assert!(second > first);
        assert_eq!(
            store.epoch(),
            second,
            "the tray still follows the replaced engine, so every status of the new one is dropped"
        );
        assert_eq!(store.status(), &EngineStatus::Restarting);
        assert!(
            host.observe().is_none(),
            "the old engine is still in the slot"
        );
        assert_eq!(
            fetch_health(port).result,
            ProbeResult::Failed,
            "the old engine still answers after the restart began"
        );
        // A late answer from the old engine is dropped; the new engine's statuses land.
        assert!(!store.apply(first, old_running));
        assert_eq!(store.status(), &EngineStatus::Restarting);
        assert!(store.apply(second, EngineStatus::Starting));

        host.shutdown_for_exit();
        assert_eq!(host.begin_restart(&mut store), None, "a restart after quit");
        assert_eq!(store.status(), &EngineStatus::Starting);
    }

    /// A probe can take seconds. An engine that died during it must be reported from what was
    /// observed after the probe, not from what was observed before it.
    #[test]
    fn a_supervisor_tick_reports_an_exit_that_happened_during_its_probe() {
        use crate::lifetime::{classify, ModelObservation, ProbeHistory};

        let host = EngineHost::default();
        let epoch = host.begin_start(&mut StatusStore::default());
        // The health gate takes the first answer; the stand-in exits right after the second.
        let outcome = host.start_with(
            epoch,
            |port| stand_in_engine(&format!("answers:2:{port}")),
            Duration::from_secs(20),
            Duration::from_millis(50),
        );
        let port = match outcome {
            StartOutcome::Ready { port } => port,
            other => panic!("{other:?}"),
        };

        let tick = host.supervisor_tick(
            |_| true,
            |probe_port| {
                assert_eq!(probe_port, port);
                let observation = fetch_health(probe_port);
                assert!(
                    wait_until(Duration::from_secs(5), || host
                        .observe()
                        .is_some_and(|engine| engine.exit.is_some())),
                    "the stand-in engine never exited after its last answer"
                );
                observation
            },
        );
        let (probed_epoch, observation) = tick.probed.expect("a due probe ran");
        assert_eq!(probed_epoch, epoch);
        assert_eq!(
            observation.result,
            ProbeResult::Answered(ModelObservation::NotReady)
        );
        let snapshot = tick
            .snapshot
            .expect("the exited engine is still in the slot");
        assert_eq!(
            snapshot.exit,
            Some(ExitSummary::Code(3)),
            "the tick returned the snapshot taken before its probe"
        );
        let mut history = ProbeHistory::default();
        history.record(epoch, observation);
        assert_eq!(
            classify(Some(&snapshot), &history, Instant::now()),
            Some(EngineStatus::Stopped {
                exit: ExitSummary::Code(3)
            })
        );

        // An engine already seen to exit is not probed again.
        let next = host.supervisor_tick(|_| true, |_| panic!("an exited engine was probed"));
        assert!(next.probed.is_none());
        assert_eq!(
            next.snapshot.and_then(|engine| engine.exit),
            Some(ExitSummary::Code(3))
        );
        host.shutdown_for_exit();
    }

    #[test]
    fn quit_reaps_a_running_engine_and_refuses_new_starts() {
        use crate::lifetime::ModelObservation;

        let host = EngineHost::default();
        let epoch = host.begin_epoch();
        let port = unused_loopback_port();
        assert!(host
            .launch(epoch, stand_in_engine(&format!("healthy:{port}")))
            .expect("launch the stand-in engine"));
        let outcome = host.gate_pending(
            epoch,
            port,
            Duration::from_secs(20),
            Duration::from_millis(50),
        );
        assert!(
            matches!(outcome, StartOutcome::Ready { port: ready } if ready == port),
            "{outcome:?}"
        );

        let snapshot = host.observe().expect("a ready engine is observed");
        assert_eq!(
            (snapshot.epoch, snapshot.port, snapshot.exit),
            (epoch, port, None)
        );
        assert_eq!(
            fetch_health(port).result,
            ProbeResult::Answered(ModelObservation::NotReady)
        );

        let reaped = host
            .shutdown_for_exit()
            .expect("quit must stop a running engine");
        assert!(!reaped.was_pending);
        assert!(reaped.status.is_some(), "the engine was not waited on");
        assert_eq!(reaped.pid, snapshot.pid);
        assert!(host.observe().is_none());
        assert_eq!(
            fetch_health(port).result,
            ProbeResult::Failed,
            "something still answers after quit"
        );
        // After quit nothing may start, not even for the epoch that was current.
        let latest = host.epoch.load(Ordering::SeqCst);
        assert!(!host
            .launch(latest, stand_in_engine("never-healthy"))
            .expect("a refused launch is not an error"));
    }

    #[test]
    fn a_launch_racing_quit_never_leaves_a_child() {
        let dir = tempfile::tempdir().unwrap();

        // Quit already under way: nothing may spawn.
        let marker = dir.path().join("spawned-after-quit");
        let host = EngineHost::default();
        let epoch = host.begin_epoch();
        host.quitting.store(true, Ordering::SeqCst);
        let launched = host
            .launch(
                epoch,
                stand_in_engine(&format!("touch:{}", marker.display())),
            )
            .expect("a refused launch is not an error");
        assert!(!launched, "a launch after quit must be refused");
        assert!(host.containment().is_none());

        // A newer generation superseded this one: nothing may spawn either.
        let stale_marker = dir.path().join("spawned-for-a-stale-epoch");
        let host = EngineHost::default();
        let stale = host.begin_epoch();
        host.begin_epoch();
        assert!(!host
            .launch(
                stale,
                stand_in_engine(&format!("touch:{}", stale_marker.display()))
            )
            .expect("a refused launch is not an error"));
        assert!(host.containment().is_none());

        std::thread::sleep(Duration::from_millis(1500));
        assert!(!marker.exists(), "a process was spawned after quit");
        assert!(
            !stale_marker.exists(),
            "a process was spawned for a stale epoch"
        );
    }

    #[test]
    fn optional_flags_are_passed_only_when_advertised() {
        let engine = if cfg!(windows) {
            PathBuf::from(r"C:\Apps\Camelid\camelid.exe")
        } else {
            PathBuf::from("/opt/camelid/camelid")
        };
        let args = |features: &EngineFeatures| -> Vec<String> {
            build_serve_command(&engine, 5000, None, features)
                .get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect()
        };

        let old_help = "Usage: camelid serve [OPTIONS]\n      --no-open\n  -h, --help";
        let old = parse_advertised_flags(old_help);
        assert_eq!(old, EngineFeatures::default());
        assert!(!old.pipes_stdin());
        let plain = args(&old);
        assert_eq!(
            &plain[..4],
            ["serve", "--addr", "127.0.0.1:5000", "--no-open"]
        );
        assert!(!plain.iter().any(|arg| arg == EXIT_WHEN_STDIN_CLOSES_FLAG));

        // A longer flag that merely starts with the literal is not the flag.
        assert!(
            !parse_advertised_flags("      --exit-when-stdin-closes-later").exit_when_stdin_closes
        );

        let new_help = "      --no-open\n      --exit-when-stdin-closes\n          Exit as soon";
        let new = parse_advertised_flags(new_help);
        assert!(new.exit_when_stdin_closes);
        assert!(new.pipes_stdin());
        let flagged = args(&new);
        assert_eq!(flagged[2], "127.0.0.1:5000");
        assert!(flagged.iter().any(|arg| arg == EXIT_WHEN_STDIN_CLOSES_FLAG));

        // A failed or slow probe passes nothing, and says why.
        let missing = std::env::temp_dir().join(format!(
            "camelid-desktop-missing-engine-{}",
            std::process::id()
        ));
        let reason = probe_engine_features(&missing).expect_err("no engine to probe");
        let (features, log) = features_or_none(Err(reason.clone()));
        assert_eq!(features, EngineFeatures::default());
        assert!(log.contains(&reason) && log.contains("without optional flags"));

        let timed_out =
            probe_features_with(stand_in_engine("never-exits"), Duration::from_millis(300))
                .expect_err("a probe that never finishes must fail");
        assert!(timed_out.contains("did not finish"), "{timed_out}");
        let (features, _) = features_or_none(Err(timed_out));
        assert_eq!(features, EngineFeatures::default());
    }

    #[cfg(windows)]
    #[test]
    fn job_object_kills_the_sidecar_when_its_last_handle_closes() {
        let mut command = Command::new("ping");
        command
            .args(["-n", "30", "127.0.0.1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut launched = launch_contained(command).expect("launch ping");
        assert_eq!(launched.containment, Containment::JobObject);

        // What the OS does to the handle when the desktop process dies.
        drop(launched.job.take());

        let exited = wait_until(Duration::from_secs(3), || {
            matches!(launched.child.try_wait(), Ok(Some(_)))
        });
        if !exited {
            let _ = launched.child.kill();
        }
        assert!(
            exited,
            "closing the job's last handle did not kill the sidecar"
        );
    }
}
