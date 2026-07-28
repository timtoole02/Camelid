// Sidecar lifecycle for the camelid engine.
//
// The desktop app does NOT reimplement any inference, tokenization, or HTTP surface. It
// spawns the shipped `camelid` server binary as a loopback-only sidecar, health-gates it,
// then points the WebView at the sidecar's already-embedded UI. Behavior is therefore
// byte-identical to running `camelid serve` + the web UI manually. See DECISIONS.md D11.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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

/// A running sidecar plus the loopback port it bound. Dropping/`shutdown` kills the child.
pub struct Engine {
    child: Child,
    port: u16,
    #[cfg(windows)]
    _job: Option<JobObject>,
}

impl Engine {
    /// The loopback base URL the WebView should navigate to. UI and API are same-origin.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }

    /// Graceful-ish shutdown. On Windows there is no SIGTERM; the child is loopback-only and
    /// holds no external state, so `TerminateProcess` (via `Child::kill`) is the clean stop.
    /// The kill-on-close job object (set in `spawn`) is the backstop if the parent crashes.
    pub fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
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

/// Spawn `camelid serve --addr 127.0.0.1:<port> --no-open --models-dir <abs>`, bound to
/// loopback only, and health-gate it. An explicit models directory lets packaged platforms
/// keep mutable model data outside the signed application bundle; Windows retains its
/// existing engine-adjacent directory when no override is supplied.
pub fn spawn(engine_path: &Path, models_dir: Option<&Path>) -> Result<Engine, EngineError> {
    let port = pick_ephemeral_port()?;
    spawn_on_port(engine_path, models_dir, port)
}

fn spawn_on_port(
    engine_path: &Path,
    models_dir: Option<&Path>,
    port: u16,
) -> Result<Engine, EngineError> {
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
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    no_console_window(&mut command);

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
                engine_path.display()
            ),
        )
    })?;

    // Tie the child's lifetime to ours: if the desktop process dies (even crashes), the OS
    // terminates the sidecar too, so no orphaned `camelid` process can survive.
    #[cfg(windows)]
    let job = JobObject::assign(&child).ok();

    match wait_for_health(port, &mut child) {
        Ok(()) => Ok(Engine {
            child,
            port,
            #[cfg(windows)]
            _job: job,
        }),
        Err(err) => Err(finish_startup_failure(&mut child, err)),
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

/// Poll `/v1/health` until it returns 200, the engine exits, or the budget elapses.
fn wait_for_health(port: u16, child: &mut Child) -> Result<(), EngineError> {
    wait_for_health_with_timeout(port, child, HEALTH_TIMEOUT, HEALTH_POLL_INTERVAL)
}

fn wait_for_health_with_timeout(
    port: u16,
    child: &mut Child,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<(), EngineError> {
    let deadline = Instant::now() + timeout;
    loop {
        // If the engine already exited, fail immediately with its status (stderr added later).
        if let Ok(Some(status)) = child.try_wait() {
            return Err(EngineError::new(
                EngineErrorKind::StartupFailed,
                format!("the camelid engine exited before becoming healthy (status: {status})"),
            ));
        }
        if http_health_ok(port) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(EngineError::new(
                EngineErrorKind::StartupTimeout,
                format!(
                    "the camelid engine did not report healthy on 127.0.0.1:{port} within {}s",
                    HEALTH_TIMEOUT.as_secs()
                ),
            ));
        }
        std::thread::sleep(poll_interval);
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
    fn assign(child: &Child) -> Result<Self, ()> {
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
                return Err(());
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
                windows_sys::Win32::Foundation::CloseHandle(job);
                return Err(());
            }
            let proc_handle = child.as_raw_handle() as HANDLE;
            if AssignProcessToJobObject(job, proc_handle) == 0 {
                windows_sys::Win32::Foundation::CloseHandle(job);
                return Err(());
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
