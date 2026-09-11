//! A stand-in engine for the Windows desktop smoke, `scripts/desktop-lifetime-smoke.ps1`.
//!
//! Copied beside a debug `camelid-desktop.exe` as `camelid.exe`, it lets the smoke drive the
//! real window messages, job object, tray and single-instance IPC without building the
//! engine. It is never shipped.
//!
//! - `serve --help` lists only the flags named in `FAKE_SIDECAR_ADVERTISE` (comma
//!   separated). The default is none, so the crash check isolates the job object.
//! - `serve --addr A ...` answers `GET /v1/health` with the idle health shape and anything
//!   else with a small page.
//! - `--exit-when-stdin-closes` is honoured when passed; other flags are ignored.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

const HEALTH: &str = r#"{"ok":true,"engine":"camelid","loaded_now":false,"generation_ready":false,"active_model_id":null}"#;
const PAGE: &str = "<!doctype html><title>Camelid Desktop</title><p>fake_sidecar</p>";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("serve") {
        eprintln!("fake_sidecar: only `serve` is implemented");
        std::process::exit(2);
    }
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("Usage: camelid serve [OPTIONS]");
        println!("      --addr <ADDR>");
        println!("      --no-open");
        println!("      --models-dir <MODELS_DIR>");
        if let Ok(advertised) = std::env::var("FAKE_SIDECAR_ADVERTISE") {
            for flag in advertised
                .split(',')
                .map(str::trim)
                .filter(|f| !f.is_empty())
            {
                println!("      {flag}");
            }
        }
        return;
    }

    if args.iter().any(|arg| arg == "--exit-when-stdin-closes") {
        std::thread::spawn(|| {
            let mut stdin = std::io::stdin();
            let mut buf = [0u8; 64];
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) | Err(_) => std::process::exit(0),
                    Ok(_) => {}
                }
            }
        });
    }

    let addr = args
        .iter()
        .position(|arg| arg == "--addr")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:8181".to_string());
    let listener = match TcpListener::bind(&addr) {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("failed to bind {addr}: {err}");
            std::process::exit(1);
        }
    };
    eprintln!("fake_sidecar: serving on {addr}");
    for stream in listener.incoming().flatten() {
        std::thread::spawn(move || answer(stream));
    }
}

fn answer(mut stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut request = Vec::new();
    let mut buf = [0u8; 1024];
    while !request.windows(4).any(|w| w == b"\r\n\r\n") && request.len() < 64 * 1024 {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => request.extend_from_slice(&buf[..n]),
        }
    }
    let head = String::from_utf8_lossy(&request);
    let path = head.split_whitespace().nth(1).unwrap_or("/");
    let (content_type, body) = if path.starts_with("/v1/health") {
        ("application/json", HEALTH)
    } else {
        ("text/html; charset=utf-8", PAGE)
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
}
