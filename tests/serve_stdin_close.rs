//! `camelid serve --exit-when-stdin-closes`, against the real binary.
//!
//! Camelid Desktop holds the write end of its sidecar's stdin, so when the desktop dies,
//! even by a crash, the OS closes the pipe. These tests prove the flag turns that into an
//! exit, and that without the flag a closed stdin changes nothing: launchd, systemd and the
//! 0.7.x desktop all start `serve` with stdin already closed.
//!
//! Every server gets `--models-dir <empty tempdir>`, so no GGUF is ever auto-loaded, and a
//! private diagnostics directory, so the runs leave nothing in the user's journal.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// A debug build on a loaded CI runner can take a while to reach its first answer.
const STARTUP_BUDGET: Duration = Duration::from_secs(120);

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve a loopback port");
    listener
        .local_addr()
        .expect("read the reserved port")
        .port()
}

fn serve_command(port: u16, scratch: &Path) -> Command {
    let models = scratch.join("models");
    std::fs::create_dir_all(&models).expect("create the empty models directory");
    let mut command = Command::new(env!("CARGO_BIN_EXE_camelid"));
    command
        .arg("serve")
        .arg("--addr")
        .arg(format!("127.0.0.1:{port}"))
        .arg("--no-open")
        .arg("--models-dir")
        .arg(&models);
    for inherited in [
        "CAMELID_MODEL",
        "CAMELID_ADDR",
        "CAMELID_MODELS_DIR",
        "CAMELID_GEMMA4_GHOST_CGHOST",
    ] {
        command.env_remove(inherited);
    }
    command
        .env("LOCALAPPDATA", scratch)
        .env("XDG_STATE_HOME", scratch)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

fn health_ok(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().unwrap(),
        Duration::from_millis(500),
    ) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let request =
        format!("GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut head = [0u8; 32];
    match stream.read(&mut head) {
        Ok(n) => String::from_utf8_lossy(&head[..n]).starts_with("HTTP/1.1 200"),
        Err(_) => false,
    }
}

fn wait_for_health(server: &mut Server, port: u16) {
    let deadline = Instant::now() + STARTUP_BUDGET;
    loop {
        if let Some(status) = server.0.try_wait().expect("query the server") {
            panic!("the server exited before it answered /v1/health: {status}");
        }
        if health_ok(port) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the server did not answer /v1/health within {STARTUP_BUDGET:?}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn wait_for_exit(server: &mut Server, budget: Duration) -> Option<ExitStatus> {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if let Some(status) = server.0.try_wait().expect("query the server") {
            return Some(status);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    None
}

#[test]
fn serve_exits_when_its_stdin_closes_under_the_flag() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let port = free_port();
    let mut command = serve_command(port, scratch.path());
    command
        .arg("--exit-when-stdin-closes")
        .stdin(Stdio::piped());
    let mut server = Server(command.spawn().expect("start camelid serve"));
    wait_for_health(&mut server, port);

    // What the OS does to the pipe when the desktop process dies.
    drop(server.0.stdin.take());

    let status = wait_for_exit(&mut server, Duration::from_secs(5))
        .expect("the server kept running after its stdin closed");
    assert!(status.success(), "expected a clean exit, got {status}");
    assert!(!health_ok(port), "something still answers on the port");
}

#[test]
fn serve_without_the_flag_survives_a_closed_stdin() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let port = free_port();
    let mut command = serve_command(port, scratch.path());
    // Immediate end-of-file, exactly what a service manager hands a server.
    command.stdin(Stdio::null());
    let mut server = Server(command.spawn().expect("start camelid serve"));
    wait_for_health(&mut server, port);

    std::thread::sleep(Duration::from_secs(3));
    assert!(
        server.0.try_wait().expect("query the server").is_none(),
        "plain serve exited because its stdin was closed"
    );
    assert!(health_ok(port), "plain serve stopped answering");
}
