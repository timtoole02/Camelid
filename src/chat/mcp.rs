//! Minimal MCP (Model Context Protocol) client — stdio transport, v1.
//!
//! Lets a user extend agent mode with third-party tools (git, databases,
//! issue trackers, …) without any of them being compiled in. Servers are
//! declared in a `camelid.mcp.json` at the workspace root; each is spawned as a
//! child process speaking JSON-RPC 2.0 over stdin/stdout, and each tool it
//! advertises is adapted into a [`ToolSpec`] that flows through the *same*
//! `validate` → tier → `Approver` → execute path as a native tool.
//!
//! # Posture
//!
//! An MCP server is untrusted third-party code speaking a protocol. Three
//! consequences, all enforced here rather than described to the model:
//!
//! 1. **Its tool descriptions and its output are data.** A server that claims
//!    its tools need no approval, or whose output says the user pre-authorised
//!    something, is describing — not deciding. Output comes back as a normal
//!    tool result and is fenced as untrusted like any other.
//! 2. **Off unless explicitly trusted.** Disabled by default, and refused
//!    outright under `CAMELID_PRODUCTION`. `--allow-mcp` only enables the
//!    feature; every workspace-declared command also needs a matching
//!    `--trust-mcp-server <name>` supplied on the command line. A trusted
//!    command is executed immediately during agent startup.
//! 3. **Never able to impersonate a native tool.** Every tool is namespaced
//!    `mcp__<server>__<tool>`, and a server whose name would collide with an
//!    existing native tool is rejected at load.
//!
//! MCP tools are classified [`Risk::Exec`]: an MCP tool can do anything the
//! server can do, so it is always gated, and `--auto-approve` does *not*
//! promote it (only `--yolo` does, which production already refuses).

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::{Risk, Sandbox, ToolSpec};

/// The config file, read from the workspace root only.
pub const CONFIG_FILE: &str = "camelid.mcp.json";

/// Namespace every MCP tool carries, so one can never shadow a native tool.
pub const PREFIX: &str = "mcp__";

/// How long to wait for a server to answer one request.
const CALL_TIMEOUT: Duration = Duration::from_secs(30);
/// Handshake and tool listing should be quick; a server that is not ready in
/// this long is treated as unusable rather than hanging the session.
const INIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Cap on a single MCP tool result, mirroring the native output cap.
const MAX_RESULT_BYTES: usize = 16 * 1024;
const MAX_CONFIG_BYTES: usize = 64 * 1024;
const MAX_SERVERS: usize = 8;
const MAX_SERVER_ARGS: usize = 64;
const MAX_SERVER_ENV: usize = 32;
const MAX_CONFIG_VALUE_BYTES: usize = 4 * 1024;
const MAX_PROTOCOL_FRAME_BYTES: usize = 64 * 1024;
const PROTOCOL_QUEUE_CAPACITY: usize = 32;
const MAX_TOOLS_PER_SERVER: usize = 64;
const MAX_TOOL_NAME_BYTES: usize = 128;
const MAX_TOOL_DESCRIPTION_BYTES: usize = 2 * 1024;
const MAX_TOOL_SCHEMA_BYTES: usize = 32 * 1024;
/// Cancellation is cooperative while a JSON-RPC request is in flight. Keep the
/// polling interval small enough that Ctrl-C and shutdown feel immediate.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(25);

// --- config -----------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    /// Executable to spawn.
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Explicit environment for the child. The parent environment is scrubbed
    /// except for the small process-launch/runtime allowlist below.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: BTreeMap<String, ServerConfig>,
}

/// Read `camelid.mcp.json` from the workspace root.
///
/// Resolved through the sandbox like any other path. A missing file is not an
/// error — it is the normal case.
pub fn load_config(sandbox: &Sandbox) -> Result<Option<McpConfig>, String> {
    let Ok(path) = sandbox.resolve(CONFIG_FILE, true) else {
        return Ok(None);
    };
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|e| format!("cannot inspect {CONFIG_FILE}: {e}"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{CONFIG_FILE} must be a regular file"));
    }
    if metadata.len() > MAX_CONFIG_BYTES as u64 {
        return Err(format!(
            "{CONFIG_FILE} exceeds the {MAX_CONFIG_BYTES}-byte safety limit"
        ));
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(&path)
        .map_err(|e| format!("cannot open {CONFIG_FILE}: {e}"))?;
    let opened = file
        .metadata()
        .map_err(|e| format!("cannot inspect opened {CONFIG_FILE}: {e}"))?;
    if !opened.is_file() || opened.len() > MAX_CONFIG_BYTES as u64 {
        return Err(format!(
            "{CONFIG_FILE} changed or is not a bounded regular file"
        ));
    }
    let mut raw = String::new();
    file.take((MAX_CONFIG_BYTES + 1) as u64)
        .read_to_string(&mut raw)
        .map_err(|e| format!("cannot read {CONFIG_FILE}: {e}"))?;
    if raw.len() > MAX_CONFIG_BYTES {
        return Err(format!(
            "{CONFIG_FILE} exceeds the {MAX_CONFIG_BYTES}-byte safety limit"
        ));
    }
    let cfg: McpConfig =
        serde_json::from_str(&raw).map_err(|e| format!("{CONFIG_FILE} is not valid JSON: {e}"))?;
    if cfg.servers.len() > MAX_SERVERS {
        return Err(format!(
            "{CONFIG_FILE} declares {} servers; at most {MAX_SERVERS} are allowed",
            cfg.servers.len()
        ));
    }
    for (name, server) in &cfg.servers {
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            || name.is_empty()
        {
            return Err(format!(
                "{CONFIG_FILE}: server name {name:?} must be alphanumeric, '-' or '_'"
            ));
        }
        if server.command.is_empty() || server.command.len() > MAX_CONFIG_VALUE_BYTES {
            return Err(format!(
                "{CONFIG_FILE}: server {name:?} has an invalid command"
            ));
        }
        if server.args.len() > MAX_SERVER_ARGS
            || server
                .args
                .iter()
                .any(|arg| arg.len() > MAX_CONFIG_VALUE_BYTES)
        {
            return Err(format!(
                "{CONFIG_FILE}: server {name:?} exceeds the argument limits"
            ));
        }
        if server.env.len() > MAX_SERVER_ENV
            || server.env.iter().any(|(key, value)| {
                key.is_empty()
                    || key.len() > 128
                    || value.len() > MAX_CONFIG_VALUE_BYTES
                    || key.bytes().any(|byte| byte == b'=' || byte == 0)
                    || value.contains('\0')
            })
        {
            return Err(format!(
                "{CONFIG_FILE}: server {name:?} exceeds the environment limits"
            ));
        }
    }
    Ok(Some(cfg))
}

// --- a single stdio server ---------------------------------------------------

fn executable_candidate(path: &Path) -> Option<PathBuf> {
    let resolved = std::fs::canonicalize(path).ok()?;
    let metadata = std::fs::metadata(&resolved).ok()?;
    if !metadata.is_file() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return None;
        }
    }
    Some(resolved)
}

fn executable_names(command: &OsStr, path_ext: Option<&OsStr>) -> Vec<std::ffi::OsString> {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut names = vec![command.to_os_string()];
    #[cfg(windows)]
    if Path::new(command).extension().is_none() {
        let extensions = path_ext
            .and_then(OsStr::to_str)
            .unwrap_or(".COM;.EXE;.BAT;.CMD");
        for extension in extensions.split(';').filter(|value| !value.is_empty()) {
            let extension = if extension.starts_with('.') {
                extension.to_string()
            } else {
                format!(".{extension}")
            };
            let mut name = command.to_os_string();
            name.push(extension);
            names.push(name);
        }
    }
    #[cfg(not(windows))]
    let _ = path_ext;
    names
}

fn resolve_executable_from_path(
    command: &OsStr,
    search_path: Option<&OsStr>,
    path_ext: Option<&OsStr>,
) -> Result<PathBuf, String> {
    let requested = Path::new(command);
    if requested.is_absolute() {
        return executable_candidate(requested).ok_or_else(|| {
            format!(
                "MCP executable {} is not an accessible regular executable",
                requested.display()
            )
        });
    }
    if requested.components().count() != 1 {
        return Err(format!(
            "MCP executable {} must be absolute or a bare name resolved through trusted PATH",
            requested.display()
        ));
    }
    let names = executable_names(command, path_ext);
    if let Some(search_path) = search_path {
        for directory in std::env::split_paths(search_path).filter(|path| path.is_absolute()) {
            for name in &names {
                if let Some(executable) = executable_candidate(&directory.join(name)) {
                    return Ok(executable);
                }
            }
        }
    }
    Err(format!(
        "MCP executable {} was not found in an absolute trusted PATH entry",
        requested.display()
    ))
}

fn resolve_config_executable(command: &str) -> Result<PathBuf, String> {
    resolve_executable_from_path(
        OsStr::new(command),
        std::env::var_os("PATH").as_deref(),
        std::env::var_os("PATHEXT").as_deref(),
    )
}

#[cfg(windows)]
fn is_windows_command_script(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("cmd") || extension.eq_ignore_ascii_case("bat")
        })
}

/// One spawned MCP server. Reads run on a helper thread so a server that stops
/// talking cannot wedge the agent — every receive is bounded by a timeout.
struct Server {
    name: String,
    child: Child,
    /// Stdin has its own writer thread. A malicious server that stops reading
    /// cannot wedge the agent inside `write(2)` before cancellation can be
    /// observed; killing the process tree closes the pipe and releases that
    /// helper thread.
    writes: SyncSender<Vec<u8>>,
    lines: Receiver<Result<String, String>>,
    next_id: u64,
    terminated: bool,
    /// Windows: ties the server's whole process tree to this handle, so
    /// killing the direct child cannot orphan grandchildren (an `npx` shim
    /// spawns the real server as its own child). Mirrors the subagent spawner.
    #[cfg(windows)]
    _job: super::win_job::JobObject,
}

impl Server {
    fn isolate_process_tree(command: &mut Command) {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            use windows_sys::Win32::System::Threading::CREATE_SUSPENDED;
            // Assignment must precede the first instruction of workspace-
            // declared code; JobObject::contain_suspended resumes it later.
            command.creation_flags(CREATE_SUSPENDED);
        }
        #[cfg(not(unix))]
        let _ = command;
    }

    fn apply_sanitized_environment(command: &mut Command, extra: &BTreeMap<String, String>) {
        command.env_clear();
        // Preserve only process-launch/runtime basics. Provider tokens, Camelid
        // credentials, cloud secrets, and the rest of the operator environment
        // never reach workspace-declared MCP code implicitly.
        for key in [
            "PATH",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "TEMP",
            "TMP",
            "SystemRoot",
            "WINDIR",
            "PATHEXT",
            "ComSpec",
        ] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command.envs(extra);
    }

    fn spawn(name: &str, cfg: &ServerConfig, cwd: &Path) -> Result<Self, String> {
        // Resolve before setting the workspace cwd. Passing a bare name to
        // CreateProcess on Windows can otherwise select an attacker-controlled
        // workspace `node.exe`/`python.exe` ahead of PATH.
        let executable = resolve_config_executable(&cfg.command)?;
        #[cfg(windows)]
        let mut cmd = if is_windows_command_script(&executable) {
            let mut command = Command::new(super::tools::system32("cmd.exe"));
            command.args(["/D", "/S", "/C"]).arg(&executable);
            command
        } else {
            Command::new(&executable)
        };
        #[cfg(not(windows))]
        let mut cmd = Command::new(&executable);
        cmd.args(&cfg.args)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // The protocol is on stdout; a server's logging on stderr is not
            // ours to interleave into the terminal.
            .stderr(Stdio::null());
        Self::apply_sanitized_environment(&mut cmd, &cfg.env);
        Self::isolate_process_tree(&mut cmd);
        let spawned = cmd.spawn();
        let mut child = spawned.map_err(|e| {
            format!(
                "could not start MCP server '{name}' ({}): {e}",
                executable.display()
            )
        })?;

        #[cfg(windows)]
        let job = match super::win_job::JobObject::contain_suspended(&mut child) {
            Ok(job) => job,
            Err(error) => {
                return Err(format!(
                    "could not contain MCP server '{name}' process tree: {error}"
                ));
            }
        };

        let mut stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let (tx, lines) = mpsc::sync_channel(PROTOCOL_QUEUE_CAPACITY);
        let (write_tx, write_rx) = mpsc::sync_channel::<Vec<u8>>(PROTOCOL_QUEUE_CAPACITY);
        let write_errors = tx.clone();
        std::thread::spawn(move || {
            while let Ok(frame) = write_rx.recv() {
                if let Err(error) = stdin.write_all(&frame).and_then(|_| stdin.flush()) {
                    let _ = write_errors.send(Err(format!("MCP stdin write failed: {error}")));
                    break;
                }
            }
        });
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut frame = Vec::new();
                let read = match reader
                    .by_ref()
                    .take((MAX_PROTOCOL_FRAME_BYTES + 1) as u64)
                    .read_until(b'\n', &mut frame)
                {
                    Ok(read) => read,
                    Err(error) => {
                        let _ = tx.send(Err(format!("MCP stdout read failed: {error}")));
                        break;
                    }
                };
                if read == 0 {
                    break;
                }
                if frame.len() > MAX_PROTOCOL_FRAME_BYTES || !frame.ends_with(b"\n") {
                    let _ = tx.send(Err(format!(
                        "MCP protocol frame exceeded {MAX_PROTOCOL_FRAME_BYTES} bytes"
                    )));
                    break;
                }
                while matches!(frame.last(), Some(b'\n' | b'\r')) {
                    frame.pop();
                }
                let line = match String::from_utf8(frame) {
                    Ok(line) => line,
                    Err(_) => {
                        let _ = tx.send(Err("MCP protocol frame was not UTF-8".to_string()));
                        break;
                    }
                };
                if tx.send(Ok(line)).is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            name: name.to_string(),
            child,
            writes: write_tx,
            lines,
            next_id: 1,
            terminated: false,
            #[cfg(windows)]
            _job: job,
        })
    }

    fn queue_json(&self, message: &Value) -> Result<(), String> {
        let mut encoded = serde_json::to_vec(message)
            .map_err(|error| format!("{}: request encoding failed: {error}", self.name))?;
        if encoded.len() + 1 > MAX_PROTOCOL_FRAME_BYTES {
            return Err(format!(
                "{}: outbound MCP frame exceeds {MAX_PROTOCOL_FRAME_BYTES} bytes",
                self.name
            ));
        }
        encoded.push(b'\n');
        match self.writes.try_send(encoded) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(format!(
                "{}: MCP protocol write queue is full; server is not consuming input",
                self.name
            )),
            Err(TrySendError::Disconnected(_)) => {
                Err(format!("{}: MCP protocol writer is gone", self.name))
            }
        }
    }

    /// Answer a server-initiated request while one of our own requests is in
    /// flight. Notifications have no id and are deliberately ignored. MCP's
    /// `ping` request gets an empty success result; every other request gets the
    /// JSON-RPC method-not-found response instead of being silently swallowed.
    fn handle_server_message(&self, message: &Value) -> Result<bool, String> {
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            return Ok(false);
        };
        let Some(id) = message.get("id").cloned() else {
            return Ok(true); // notification
        };
        let response = if method == "ping" {
            json!({"jsonrpc":"2.0","id":id,"result":{}})
        } else {
            json!({
                "jsonrpc":"2.0",
                "id":id,
                "error":{"code":-32601,"message":"Method not found"}
            })
        };
        self.queue_json(&response)?;
        Ok(true)
    }

    /// One JSON-RPC round trip. Notifications and unrelated messages are
    /// skipped until the matching id arrives, cancellation is requested, or
    /// the deadline passes.
    fn request(
        &mut self,
        method: &str,
        params: Value,
        timeout: Duration,
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
    ) -> Result<Value, String> {
        if self.terminated {
            return Err(format!("{}: server is no longer running", self.name));
        }
        if cancel.load(Ordering::Acquire) || shutdown.load(Ordering::Acquire) {
            self.terminate();
            return Err(format!("{}: {method} cancelled", self.name));
        }
        let id = self.next_id;
        self.next_id += 1;
        let req = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        if let Err(error) = self.queue_json(&req) {
            self.terminate();
            return Err(error);
        }

        let deadline = std::time::Instant::now() + timeout;
        loop {
            if cancel.load(Ordering::Acquire) || shutdown.load(Ordering::Acquire) {
                self.terminate();
                return Err(format!("{}: {method} cancelled", self.name));
            }
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                self.terminate();
                return Err(format!("{}: no response to {method} in time", self.name));
            }
            let line = match self.lines.recv_timeout(left.min(CANCEL_POLL_INTERVAL)) {
                Ok(Ok(line)) => line,
                Ok(Err(error)) => {
                    self.terminate();
                    return Err(format!("{}: {error}", self.name));
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    self.terminate();
                    return Err(format!("{}: MCP server closed its output", self.name));
                }
            };
            let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                continue; // not JSON — a stray log line; ignore
            };
            // A message carrying `method` is a server-initiated request or
            // notification, NOT a response — even when its id collides with
            // ours (both sides count from 1, and e.g. `ping` is exempt from
            // capability negotiation). Matching on id alone would consume it
            // as our answer and hand the model a null "success".
            match self.handle_server_message(&msg) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(error) => {
                    self.terminate();
                    return Err(error);
                }
            }
            if msg.get("id").and_then(Value::as_u64) != Some(id) {
                continue; // another response
            }
            if let Some(err) = msg.get("error") {
                let m = err
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                return Err(format!("{}: {m}", self.name));
            }
            return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), String> {
        let msg = json!({"jsonrpc":"2.0","method":method,"params":params});
        self.queue_json(&msg)
    }

    fn initialize(&mut self, cancel: &AtomicBool, shutdown: &AtomicBool) -> Result<(), String> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "camelid", "version": env!("CARGO_PKG_VERSION")},
            }),
            INIT_TIMEOUT,
            cancel,
            shutdown,
        )?;
        self.notify("notifications/initialized", json!({}))
    }

    fn list_tools(
        &mut self,
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
    ) -> Result<Vec<(String, String, Value)>, String> {
        let res = self.request("tools/list", json!({}), INIT_TIMEOUT, cancel, shutdown)?;
        let items = res
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if items.len() > MAX_TOOLS_PER_SERVER {
            return Err(format!(
                "{}: advertised {} tools; at most {MAX_TOOLS_PER_SERVER} are allowed",
                self.name,
                items.len()
            ));
        }
        let mut out = Vec::new();
        for t in items {
            let Some(name) = t.get("name").and_then(Value::as_str) else {
                continue;
            };
            if name.is_empty()
                || name.len() > MAX_TOOL_NAME_BYTES
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
            {
                return Err(format!("{}: advertised an invalid tool name", self.name));
            }
            let desc = t
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("(no description)")
                .to_string();
            if desc.len() > MAX_TOOL_DESCRIPTION_BYTES {
                return Err(format!(
                    "{}: tool {name:?} description exceeds {MAX_TOOL_DESCRIPTION_BYTES} bytes",
                    self.name
                ));
            }
            let schema = t
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type":"object"}));
            if serde_json::to_vec(&schema)
                .map(|encoded| encoded.len() > MAX_TOOL_SCHEMA_BYTES)
                .unwrap_or(true)
            {
                return Err(format!(
                    "{}: tool {name:?} schema exceeds {MAX_TOOL_SCHEMA_BYTES} bytes",
                    self.name
                ));
            }
            out.push((name.to_string(), desc, schema));
        }
        Ok(out)
    }

    fn call(
        &mut self,
        tool: &str,
        args: &Value,
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
    ) -> Result<String, String> {
        let res = self.request(
            "tools/call",
            json!({"name": tool, "arguments": args}),
            CALL_TIMEOUT,
            cancel,
            shutdown,
        )?;
        // MCP returns content blocks; flatten the text ones. A server may also
        // signal a tool-level failure via isError while still returning 200.
        let mut text = String::new();
        if let Some(blocks) = res.get("content").and_then(Value::as_array) {
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                            text.push('\n');
                        }
                    }
                    Some(other) => text.push_str(&format!("[{other} content omitted]\n")),
                    None => {}
                }
            }
        }
        if text.is_empty() {
            text = res.to_string();
        }
        if text.len() > MAX_RESULT_BYTES {
            let mut end = MAX_RESULT_BYTES;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
            text.push_str("\n…[truncated]");
        }
        if res.get("isError").and_then(Value::as_bool) == Some(true) {
            return Err(text);
        }
        Ok(text)
    }

    fn terminate(&mut self) {
        if self.terminated {
            return;
        }
        self.terminated = true;
        #[cfg(windows)]
        self._job.terminate();
        #[cfg(unix)]
        unsafe {
            // The child was placed in its own process group before spawn.
            // Killing the group prevents node/python launchers from orphaning
            // their real MCP server process on timeout or shutdown.
            libc::kill(-(self.child.id() as i32), libc::SIGKILL);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.terminate();
    }
}

// --- the process-wide registry ----------------------------------------------

/// One adapted tool: the namespaced name the model sees, plus where it came from.
struct Entry {
    /// `mcp__<server>__<tool>`
    public: String,
    server: String,
    tool: String,
    description: String,
    schema: Value,
}

#[derive(Default)]
struct Registry {
    servers: BTreeMap<String, Arc<ManagedServer>>,
    tools: Vec<Entry>,
}

/// Per-server synchronization is intentionally separate from the registry
/// lock. A 30-second JSON-RPC round trip must not block `specs`, `has_tool`, or
/// (most importantly) shutdown from finding and cancelling every server.
struct ManagedServer {
    stop: AtomicBool,
    inner: Mutex<Server>,
}

fn registry() -> &'static Mutex<Option<Registry>> {
    static R: OnceLock<Mutex<Option<Registry>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

/// Whether MCP tools should be advertised and callable right now.
pub fn is_enabled() -> bool {
    registry()
        .lock()
        .map(|r| r.as_ref().is_some_and(|r| !r.tools.is_empty()))
        .unwrap_or(false)
}

/// Start every configured server and adopt its tools.
///
/// Returns the number of tools adopted. Errors here are reported to the caller
/// and are never fatal to the session: a broken MCP config should cost you MCP,
/// not your agent.
pub fn configure(
    sandbox: &Sandbox,
    allow_mcp: bool,
    production: bool,
    native_tool_names: &[String],
    trusted_server_names: &[String],
    cancel: &AtomicBool,
) -> Result<usize, String> {
    // Configuration is per agent session. Tear down the preceding session
    // before validating this one; the candidate below remains unpublished
    // until every trusted server has initialized successfully.
    shutdown();
    if !allow_mcp {
        return Ok(0);
    }
    // Detect a poisoned publication lock before starting any candidate child.
    drop(
        registry()
            .lock()
            .map_err(|_| "mcp registry poisoned".to_string())?,
    );
    if production {
        return Err(
            "MCP is refused under CAMELID_PRODUCTION: it would expose third-party tools to an \
             unattended agent. Unset CAMELID_PRODUCTION to use --allow-mcp."
                .into(),
        );
    }
    let Some(cfg) = load_config(sandbox)? else {
        publish_registry(None)?;
        return Ok(0);
    };

    let trusted = trusted_server_names.iter().collect::<BTreeSet<_>>();
    if trusted.is_empty() && !cfg.servers.is_empty() {
        let configured = cfg.servers.keys().cloned().collect::<Vec<_>>().join(", ");
        return Err(format!(
            "--allow-mcp does not execute workspace commands by itself. Review {CONFIG_FILE}, then add --trust-mcp-server <NAME> once for each server you trust. Trusted commands start immediately during agent startup. Configured servers: {configured}"
        ));
    }
    let unknown = trusted
        .iter()
        .filter(|name| !cfg.servers.contains_key(name.as_str()))
        .map(|name| name.as_str())
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(format!(
            "--trust-mcp-server named server(s) not present in {CONFIG_FILE}: {}. No MCP servers were started",
            unknown.join(", ")
        ));
    }

    let mut reg = Registry::default();
    let mut problems: Vec<String> = Vec::new();

    for (name, sc) in &cfg.servers {
        if !trusted.contains(name) {
            continue;
        }
        let mut server = match Server::spawn(name, sc, sandbox.root()) {
            Ok(s) => s,
            Err(e) => {
                problems.push(e);
                continue;
            }
        };
        let server_stop = AtomicBool::new(false);
        if let Err(e) = server.initialize(cancel, &server_stop) {
            problems.push(format!("MCP server '{name}' failed to initialize: {e}"));
            continue;
        }
        let listed = match server.list_tools(cancel, &server_stop) {
            Ok(t) => t,
            Err(e) => {
                problems.push(format!("MCP server '{name}' would not list tools: {e}"));
                continue;
            }
        };
        for (tool, description, schema) in listed {
            let public = format!("{PREFIX}{name}__{tool}");
            // Belt and braces: the prefix already makes collision impossible,
            // but assert it rather than assume it.
            if native_tool_names.contains(&public) {
                problems.push(format!(
                    "MCP tool '{public}' collides with a native tool; skipped"
                ));
                continue;
            }
            reg.tools.push(Entry {
                public,
                server: name.clone(),
                tool,
                description,
                schema,
            });
        }
        reg.servers.insert(
            name.clone(),
            Arc::new(ManagedServer {
                stop: server_stop,
                inner: Mutex::new(server),
            }),
        );
    }

    let adopted = reg.tools.len();
    if !problems.is_empty() {
        // Configuration is transactional. Dropping the unpublished registry
        // terminates every successfully started server, while the process-wide
        // registry remains None from shutdown() above.
        drop(reg);
        return Err(problems.join("; "));
    }
    publish_registry(Some(reg))?;
    Ok(adopted)
}

fn terminate_registry(old: Registry) {
    // Signal every in-flight request before waiting for any one server lock.
    // Requests poll at CANCEL_POLL_INTERVAL and terminate their own process
    // tree before releasing the per-server lock.
    for server in old.servers.values() {
        server.stop.store(true, Ordering::Release);
    }
    for server in old.servers.values() {
        let mut inner = server
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        inner.terminate();
    }
}

/// Atomically publish a fully initialized registry, then tear down any prior
/// one outside the global lock. Candidate construction errors never call this,
/// so no partial candidate becomes visible.
fn publish_registry(next: Option<Registry>) -> Result<(), String> {
    let old = {
        let mut current = registry()
            .lock()
            .map_err(|_| "mcp registry poisoned".to_string())?;
        std::mem::replace(&mut *current, next)
    };
    let Some(old) = old else {
        return Ok(());
    };
    terminate_registry(old);
    Ok(())
}

/// Stop every server and forget its tools.
pub fn shutdown() {
    let old = registry()
        .lock()
        .ok()
        .and_then(|mut registry| registry.take());
    if let Some(old) = old {
        terminate_registry(old);
    }
}

/// The adapted tool specs to advertise alongside the native ones.
pub fn specs() -> Vec<ToolSpec> {
    let Ok(guard) = registry().lock() else {
        return Vec::new();
    };
    let Some(reg) = guard.as_ref() else {
        return Vec::new();
    };
    reg.tools
        .iter()
        .map(|e| ToolSpec {
            name: e.public.clone(),
            description: format!("[MCP: {}] {}", e.server, e.description),
            // An MCP tool can do whatever its server can. Always gated, and not
            // promoted by --auto-approve.
            risk: Risk::Exec,
            params: e.schema.clone(),
        })
        .collect()
}

/// Whether a namespaced name is one this registry currently serves.
pub fn has_tool(public: &str) -> bool {
    let Ok(guard) = registry().lock() else {
        return false;
    };
    guard
        .as_ref()
        .is_some_and(|r| r.tools.iter().any(|e| e.public == public))
}

/// Invoke a namespaced MCP tool. The returned text is untrusted data and is
/// surfaced to the model through the same fenced tool-result path as any other.
#[cfg(test)]
pub fn call(public: &str, args: &Value) -> Result<String, String> {
    call_with_cancel(public, args, &super::session::CANCEL)
}

/// Cancellation-aware invocation used by the agent loop. Only registry lookup
/// happens under the process-wide mutex; the potentially long round trip is
/// serialized by the selected server's own mutex.
pub fn call_with_cancel(public: &str, args: &Value, cancel: &AtomicBool) -> Result<String, String> {
    let guard = registry()
        .lock()
        .map_err(|_| "mcp registry poisoned".to_string())?;
    let reg = guard.as_ref().ok_or("MCP is not enabled")?;
    let (server, server_name, tool) = reg
        .tools
        .iter()
        .find(|e| e.public == public)
        .and_then(|entry| {
            reg.servers
                .get(&entry.server)
                .map(|server| (Arc::clone(server), entry.server.clone(), entry.tool.clone()))
        })
        .ok_or_else(|| format!("unknown MCP tool '{public}'"))?;
    drop(guard);

    if server.stop.load(Ordering::Acquire) {
        return Err(format!("MCP server '{server_name}' is shutting down"));
    }
    let mut inner = server
        .inner
        .lock()
        .map_err(|_| format!("MCP server '{server_name}' lock poisoned"))?;
    inner.call(&tool, args, cancel, &server.stop)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The registry is process-wide, so tests that touch it must not run
    /// concurrently with each other or with the tool-set pins in `tools`.
    pub(crate) fn registry_lock() -> std::sync::MutexGuard<'static, ()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn sandbox(root: &std::path::Path) -> Sandbox {
        Sandbox::new(root, false, Duration::from_secs(5)).unwrap()
    }

    #[test]
    fn missing_config_is_not_an_error() {
        let d = tempfile::tempdir().unwrap();
        assert!(load_config(&sandbox(d.path())).unwrap().is_none());
    }

    #[test]
    fn config_parses_servers() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"git":{"command":"mcp-git","args":["--repo","."]}}}"#,
        )
        .unwrap();
        let cfg = load_config(&sandbox(d.path())).unwrap().unwrap();
        assert_eq!(cfg.servers.len(), 1);
        let git = &cfg.servers["git"];
        assert_eq!(git.command, "mcp-git");
        assert_eq!(git.args, vec!["--repo", "."]);
    }

    #[test]
    fn malformed_config_is_a_clean_error() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(CONFIG_FILE), "{not json").unwrap();
        assert!(load_config(&sandbox(d.path())).is_err());
    }

    #[test]
    fn config_is_bounded_and_must_be_regular() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(CONFIG_FILE), vec![b' '; MAX_CONFIG_BYTES + 1]).unwrap();
        let error = load_config(&sandbox(d.path())).unwrap_err();
        assert!(error.contains("safety limit"), "{error}");

        std::fs::remove_file(d.path().join(CONFIG_FILE)).unwrap();
        std::fs::create_dir(d.path().join(CONFIG_FILE)).unwrap();
        let error = load_config(&sandbox(d.path())).unwrap_err();
        assert!(error.contains("regular file"), "{error}");
    }

    #[test]
    fn config_rejects_environment_and_server_count_abuse() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"x":{"command":"ok","env":{"BAD=KEY":"value"}}}}"#,
        )
        .unwrap();
        assert!(load_config(&sandbox(d.path())).is_err());

        let servers = (0..=MAX_SERVERS)
            .map(|index| format!(r#""s{index}":{{"command":"ok"}}"#))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            d.path().join(CONFIG_FILE),
            format!(r#"{{"servers":{{{servers}}}}}"#),
        )
        .unwrap();
        assert!(load_config(&sandbox(d.path())).is_err());
    }

    /// A server name is spliced into every tool name the model sees, so it must
    /// not be able to carry separators or escape the namespace.
    #[test]
    fn hostile_server_names_are_rejected() {
        let d = tempfile::tempdir().unwrap();
        for bad in ["../evil", "a b", "read_file/../x", ""] {
            std::fs::write(
                d.path().join(CONFIG_FILE),
                format!(r#"{{"servers":{{"{bad}":{{"command":"x"}}}}}}"#),
            )
            .unwrap();
            assert!(
                load_config(&sandbox(d.path())).is_err(),
                "server name {bad:?} should be refused"
            );
        }
    }

    #[test]
    fn executable_resolution_ignores_workspace_and_relative_path_entries() {
        let workspace = tempfile::tempdir().unwrap();
        let trusted = tempfile::tempdir().unwrap();
        #[cfg(windows)]
        let filename = "camelid-mcp-resolution-test.CMD";
        #[cfg(not(windows))]
        let filename = "camelid-mcp-resolution-test";
        let shadow = workspace.path().join(filename);
        let expected = trusted.path().join(filename);
        std::fs::write(&shadow, "shadow").unwrap();
        std::fs::write(&expected, "trusted").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&shadow, std::fs::Permissions::from_mode(0o700)).unwrap();
            std::fs::set_permissions(&expected, std::fs::Permissions::from_mode(0o700)).unwrap();
        }

        let trusted_path = std::env::join_paths([trusted.path()]).unwrap();
        let resolved = resolve_executable_from_path(
            OsStr::new("camelid-mcp-resolution-test"),
            Some(&trusted_path),
            Some(OsStr::new(".CMD;.EXE")),
        )
        .unwrap();
        assert_eq!(resolved, std::fs::canonicalize(&expected).unwrap());
        assert_ne!(resolved, std::fs::canonicalize(&shadow).unwrap());

        let relative_path = std::env::join_paths([Path::new(".")]).unwrap();
        assert!(resolve_executable_from_path(
            OsStr::new("camelid-mcp-resolution-test"),
            Some(&relative_path),
            Some(OsStr::new(".CMD;.EXE")),
        )
        .is_err());
        assert!(resolve_executable_from_path(
            OsStr::new("./camelid-mcp-resolution-test"),
            Some(&trusted_path),
            Some(OsStr::new(".CMD;.EXE")),
        )
        .is_err());
    }

    #[test]
    fn disabled_by_default() {
        let _guard = registry_lock();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"x":{"command":"true"}}}"#,
        )
        .unwrap();
        // allow_mcp = false → nothing is spawned, nothing is adopted.
        assert_eq!(
            configure(
                &sandbox(d.path()),
                false,
                false,
                &[],
                &[],
                &AtomicBool::new(false),
            )
            .unwrap(),
            0
        );
        assert!(!is_enabled());
        assert!(specs().is_empty());
    }

    #[test]
    fn refused_under_production() {
        let _guard = registry_lock();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"x":{"command":"true"}}}"#,
        )
        .unwrap();
        let err = configure(
            &sandbox(d.path()),
            true,
            true,
            &[],
            &["x".into()],
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(err.contains("CAMELID_PRODUCTION"), "{err}");
        assert!(!is_enabled());
    }

    #[test]
    fn calling_without_a_registry_is_an_error_not_a_panic() {
        let _guard = registry_lock();
        shutdown();
        assert!(!has_tool("mcp__x__y"));
        assert!(call("mcp__x__y", &json!({})).is_err());
    }

    #[cfg(unix)]
    fn python3_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    // --- end-to-end against a stub stdio server ---
    //
    // These share the process-wide registry, so they run under one #[test] to
    // keep cargo's thread-per-test from interleaving configure/shutdown.

    /// A tiny MCP server: initialize, tools/list, tools/call over stdio.
    #[cfg(unix)]
    const STUB: &str = r#"
import sys, json, os
def send(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
marker = os.environ.get("MCP_START_MARKER")
if marker:
    open(marker, "w").write("started")
sys.stderr.write("a log line that is not JSON\n")
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    msg = json.loads(line)
    m, i = msg.get("method"), msg.get("id")
    if m == "initialize":
        send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2024-11-05"}})
    elif m == "tools/list":
        send({"jsonrpc":"2.0","id":i,"result":{"tools":[
            {"name":"echo","description":"Echo a value.",
             "inputSchema":{"type":"object","properties":{"v":{"type":"string"}}}},
            {"name":"boom","description":"Always fails."}
        ]}})
    elif m == "tools/call":
        # A notification is ignored. Requests are different: ping must receive
        # an empty success response and unknown methods must receive -32601.
        send({"jsonrpc":"2.0","method":"notifications/progress","params":{}})
        send({"jsonrpc":"2.0","id":"server-ping","method":"ping","params":{}})
        pong = json.loads(sys.stdin.readline())
        if pong.get("id") != "server-ping" or pong.get("result") != {}:
            raise SystemExit("client did not answer ping")
        send({"jsonrpc":"2.0","id":"server-unknown","method":"server/private","params":{}})
        missing = json.loads(sys.stdin.readline())
        if (missing.get("id") != "server-unknown" or
                missing.get("error", {}).get("code") != -32601):
            raise SystemExit("client did not reject unknown request")
        p = msg.get("params", {})
        if p.get("name") == "boom":
            send({"jsonrpc":"2.0","id":i,"result":{"isError":True,
                  "content":[{"type":"text","text":"it broke"}]}})
        else:
            v = p.get("arguments", {}).get("v", "")
            send({"jsonrpc":"2.0","id":i,"result":{"content":[
                {"type":"text","text":"echo:" + str(v)}]}})
"#;

    #[cfg(unix)]
    fn write_stub_config(dir: &std::path::Path) {
        let script = dir.join("stub_server.py");
        std::fs::write(&script, STUB).unwrap();
        std::fs::write(
            dir.join(CONFIG_FILE),
            format!(
                r#"{{"servers":{{"stub":{{"command":"python3","args":["{}"]}}}}}}"#,
                script.display()
            ),
        )
        .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn failed_multi_server_configuration_publishes_no_partial_registry() {
        let _guard = registry_lock();
        if !python3_available() {
            eprintln!("skipping: no python3");
            return;
        }

        let d = tempfile::tempdir().unwrap();
        let script = d.path().join("stub_server.py");
        let marker = d.path().join("good-started");
        std::fs::write(&script, STUB).unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            serde_json::to_vec(&json!({
                "servers": {
                    "good": {
                        "command": "python3",
                        "args": [script.to_string_lossy()],
                        "env": {"MCP_START_MARKER": marker.to_string_lossy()},
                    },
                    "zzz-missing": {
                        "command": d.path().join("does-not-exist").to_string_lossy(),
                    },
                }
            }))
            .unwrap(),
        )
        .unwrap();

        shutdown();
        let error = configure(
            &sandbox(d.path()),
            true,
            false,
            &[],
            &["good".into(), "zzz-missing".into()],
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(error.contains("does-not-exist"), "{error}");
        assert!(marker.exists(), "the successful first server did not start");
        assert!(!is_enabled(), "failed configure leaked a partial registry");
        assert!(specs().is_empty());
    }

    /// `--allow-mcp` is feature enablement, not consent to execute commands
    /// planted in a workspace. Only CLI-named servers may produce a startup
    /// side effect, and a missing trust list explains exactly how to proceed.
    #[cfg(unix)]
    #[test]
    fn startup_requires_explicit_per_server_trust() {
        let _guard = registry_lock();
        if !python3_available() {
            eprintln!("skipping: no python3");
            return;
        }

        let d = tempfile::tempdir().unwrap();
        let script = d.path().join("stub_server.py");
        let trusted_marker = d.path().join("trusted-started");
        let untrusted_marker = d.path().join("untrusted-started");
        std::fs::write(&script, STUB).unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            serde_json::to_vec(&json!({
                "servers": {
                    "trusted": {
                        "command": "python3",
                        "args": [script.to_string_lossy()],
                        "env": {"MCP_START_MARKER": trusted_marker.to_string_lossy()},
                    },
                    "untrusted": {
                        "command": "python3",
                        "args": [script.to_string_lossy()],
                        "env": {"MCP_START_MARKER": untrusted_marker.to_string_lossy()},
                    },
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let sb = sandbox(d.path());

        let error = configure(&sb, true, false, &[], &[], &AtomicBool::new(false)).unwrap_err();
        assert!(error.contains("--trust-mcp-server <NAME>"), "{error}");
        assert!(error.contains("start immediately"), "{error}");
        assert!(!trusted_marker.exists());
        assert!(!untrusted_marker.exists());

        let adopted = configure(
            &sb,
            true,
            false,
            &[],
            &["trusted".into()],
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(adopted, 2);
        assert!(trusted_marker.exists());
        assert!(!untrusted_marker.exists());
        shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn stub_server_end_to_end() {
        let _guard = registry_lock();
        if !python3_available() {
            eprintln!("skipping: no python3");
            return;
        }

        let d = tempfile::tempdir().unwrap();
        write_stub_config(d.path());
        let sb = sandbox(d.path());

        shutdown();
        let n = configure(
            &sb,
            true,
            false,
            &[],
            &["stub".into()],
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(n, 2, "both stub tools should be adopted");
        assert!(is_enabled());

        // Namespaced, and carrying the server's own description.
        let adopted = specs();
        let names: Vec<&str> = adopted.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"mcp__stub__echo"));
        assert!(names.contains(&"mcp__stub__boom"));
        assert!(names.iter().all(|n| n.starts_with(PREFIX)));

        // Exec tier: always gated, and --auto-approve does not promote it.
        let echo = adopted.iter().find(|s| s.name.ends_with("echo")).unwrap();
        assert_eq!(echo.risk, Risk::Exec);
        assert!(echo.risk.needs_approval());
        assert!(echo.description.contains("Echo a value."));

        // A real call round-trips, tolerating the server's non-JSON stderr noise.
        let out = call("mcp__stub__echo", &json!({"v":"hi"})).unwrap();
        assert!(out.contains("echo:hi"), "{out}");

        // A tool-level failure comes back as an error, not a silent success.
        let err = call("mcp__stub__boom", &json!({})).unwrap_err();
        assert!(err.contains("it broke"), "{err}");

        // Unknown names are refused.
        assert!(call("mcp__stub__nope", &json!({})).is_err());

        shutdown();
        assert!(!is_enabled());
        assert!(specs().is_empty());
    }

    #[cfg(unix)]
    const HUNG_STUB: &str = r#"
import sys, json, os, subprocess, time
def send(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    msg = json.loads(line)
    m, i = msg.get("method"), msg.get("id")
    if m == "initialize":
        send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2024-11-05"}})
    elif m == "tools/list":
        send({"jsonrpc":"2.0","id":i,"result":{"tools":[
            {"name":"hang","description":"Never returns."}
        ]}})
    elif m == "tools/call":
        late = os.environ["MCP_LATE_MARKER"]
        subprocess.Popen([sys.executable, "-c",
            "import sys,time;time.sleep(.75);open(sys.argv[1],'w').write('orphan')", late])
        open(os.environ["MCP_CALL_MARKER"], "w").write("called")
        while True: time.sleep(10)
"#;

    #[cfg(unix)]
    fn write_hung_config(
        dir: &std::path::Path,
        call_marker: &std::path::Path,
        late_marker: &std::path::Path,
    ) {
        let script = dir.join("hung_server.py");
        std::fs::write(&script, HUNG_STUB).unwrap();
        std::fs::write(
            dir.join(CONFIG_FILE),
            serde_json::to_vec(&json!({
                "servers": {
                    "hung": {
                        "command": "python3",
                        "args": [script.to_string_lossy()],
                        "env": {
                            "MCP_CALL_MARKER": call_marker.to_string_lossy(),
                            "MCP_LATE_MARKER": late_marker.to_string_lossy(),
                        },
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[cfg(unix)]
    fn wait_for_file(path: &std::path::Path, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if path.exists() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    /// Both the agent's Ctrl-C flag and process-wide shutdown interrupt a hung
    /// JSON-RPC call promptly. The global registry remains available while the
    /// call waits, and killing the isolated process group prevents a spawned
    /// grandchild from surviving to produce its delayed side effect.
    #[cfg(unix)]
    #[test]
    fn hung_calls_cancel_and_shutdown_promptly_with_tree_teardown() {
        let _guard = registry_lock();
        if !python3_available() {
            eprintln!("skipping: no python3");
            return;
        }

        let d = tempfile::tempdir().unwrap();
        let called = d.path().join("called");
        let orphaned = d.path().join("orphaned");
        write_hung_config(d.path(), &called, &orphaned);
        let sb = sandbox(d.path());
        let configure_once = || {
            configure(
                &sb,
                true,
                false,
                &[],
                &["hung".into()],
                &AtomicBool::new(false),
            )
            .unwrap()
        };

        assert_eq!(configure_once(), 1);
        let cancel = Arc::new(AtomicBool::new(false));
        let call_cancel = Arc::clone(&cancel);
        let call_thread = std::thread::spawn(move || {
            call_with_cancel("mcp__hung__hang", &json!({}), call_cancel.as_ref())
        });
        assert!(wait_for_file(&called, Duration::from_secs(2)));
        let started = std::time::Instant::now();
        cancel.store(true, Ordering::Release);
        let error = call_thread.join().unwrap().unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(error.contains("cancelled"), "{error}");
        std::thread::sleep(Duration::from_secs(1));
        assert!(!orphaned.exists(), "MCP grandchild survived cancellation");

        std::fs::remove_file(&called).unwrap();
        assert_eq!(configure_once(), 1);
        let call_thread = std::thread::spawn(move || {
            call_with_cancel("mcp__hung__hang", &json!({}), &AtomicBool::new(false))
        });
        assert!(wait_for_file(&called, Duration::from_secs(2)));
        let lookup_started = std::time::Instant::now();
        assert_eq!(specs().len(), 1);
        assert!(lookup_started.elapsed() < Duration::from_millis(250));
        let shutdown_started = std::time::Instant::now();
        shutdown();
        assert!(shutdown_started.elapsed() < Duration::from_secs(2));
        let error = call_thread.join().unwrap().unwrap_err();
        assert!(error.contains("cancelled"), "{error}");
        std::thread::sleep(Duration::from_secs(1));
        assert!(!orphaned.exists(), "MCP grandchild survived shutdown");
    }

    /// A process that is not an MCP server at all must not become one. `cat`
    /// echoes our own request back — id and all. Before the method-skip fix
    /// that echo "passed" the handshake with a null result; now the echoed
    /// request is recognised as a request (it carries `method`) and skipped,
    /// so initialize times out and the server is reported, not adopted.
    #[cfg(unix)]
    #[test]
    fn an_echo_server_is_refused_not_adopted() {
        let _guard = registry_lock();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"echo":{"command":"cat"}}}"#,
        )
        .unwrap();
        shutdown();
        let err = configure(
            &sandbox(d.path()),
            true,
            false,
            &[],
            &["echo".into()],
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(err.contains("initialize"), "{err}");
        assert!(!is_enabled());
        assert!(specs().is_empty());
        shutdown();
    }

    /// A server that never answers must not wedge the agent: the handshake is
    /// bounded by INIT_TIMEOUT, not by the server's goodwill.
    #[cfg(unix)]
    #[test]
    fn a_silent_server_times_out_instead_of_hanging() {
        let _guard = registry_lock();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join(CONFIG_FILE),
            r#"{"servers":{"mute":{"command":"sleep","args":["120"]}}}"#,
        )
        .unwrap();
        shutdown();
        let started = std::time::Instant::now();
        let res = configure(
            &sandbox(d.path()),
            true,
            false,
            &[],
            &["mute".into()],
            &AtomicBool::new(false),
        );
        let elapsed = started.elapsed();
        assert!(res.is_err(), "a mute server should be reported");
        assert!(res.unwrap_err().contains("in time"));
        assert!(elapsed >= INIT_TIMEOUT, "returned too early: {elapsed:?}");
        assert!(elapsed < INIT_TIMEOUT * 3, "took {elapsed:?}");
        assert!(!is_enabled());
        shutdown();
    }
}
