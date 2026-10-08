//! Engine-owned static previews. No shell command or child web server is involved.
use super::tools::Sandbox;
use axum::{
    extract::State,
    http::{header, HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    Router,
};
use serde::Serialize;
use std::{
    fs,
    io::Read,
    net::TcpListener,
    path::{Component, Path, PathBuf},
    process::Command,
    sync::Arc,
    thread,
    time::Duration,
};
use tokio::sync::oneshot;

const MAX_ASSET: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct Status {
    pub running: bool,
    pub url: Option<String>,
    pub entry: Option<String>,
    pub chrome_opened: bool,
}

pub(crate) struct PreviewServer {
    root: PathBuf,
    status: Status,
    stop: Option<oneshot::Sender<()>>,
    worker: Option<thread::JoinHandle<()>>,
}

#[derive(Clone)]
struct Site {
    root: PathBuf,
    host: String,
    origin: String,
}

impl PreviewServer {
    pub fn start(workspace: &Path, entry: &str) -> Result<Self, String> {
        let entry = if entry.is_empty() {
            "index.html"
        } else {
            entry
        };
        validate_relative(entry)?;
        if !entry.to_ascii_lowercase().ends_with(".html") {
            return Err(
                "Choose an HTML file such as tiny-board/index.html as the preview entry.".into(),
            );
        }
        let sandbox =
            Sandbox::new(workspace, false, Duration::from_secs(1)).map_err(|e| e.to_string())?;
        let file = sandbox.resolve(entry, true)?;
        validate_relative(
            &file
                .strip_prefix(sandbox.root())
                .map_err(|_| "Preview escaped the workspace.")?
                .to_string_lossy()
                .replace('\\', "/"),
        )?;
        if !file.is_file() {
            return Err("The preview entry is not a file.".into());
        }
        let root = file
            .parent()
            .ok_or("Missing preview folder.")?
            .to_path_buf();
        read_asset(
            &root,
            file.file_name()
                .and_then(|s| s.to_str())
                .ok_or("Invalid preview filename.")?,
        )?;
        let listener =
            TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).map_err(|e| e.to_string())?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let origin = format!("http://{address}");
        let mut url = url::Url::parse(&origin).map_err(|e| e.to_string())?;
        url.path_segments_mut()
            .map_err(|_| "Invalid preview URL.")?
            .push(file.file_name().unwrap().to_str().unwrap());
        let site = Arc::new(Site {
            root: root.clone(),
            host: address.to_string(),
            origin,
        });
        let (stop, stopped) = oneshot::channel();
        // Construct the runtime before reporting success, so startup errors are visible.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        let worker = thread::Builder::new()
            .name("camelid-preview".into())
            .spawn(move || {
                runtime.block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(listener)
                        .expect("validated nonblocking listener");
                    let app = Router::new().fallback(serve_asset).with_state(site);
                    tokio::select! {
                        _ = axum::serve(listener, app).into_future() => {},
                        _ = stopped => {},
                    }
                });
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            root,
            status: Status {
                running: true,
                url: Some(url.to_string()),
                entry: Some(entry.replace('\\', "/")),
                chrome_opened: false,
            },
            stop: Some(stop),
            worker: Some(worker),
        })
    }

    pub fn status(&self) -> Status {
        let mut status = self.status.clone();
        status.running = self.worker.as_ref().is_some_and(|w| !w.is_finished());
        status
    }

    pub fn matches(&self, workspace: &Path, entry: &str) -> bool {
        let entry = if entry.is_empty() {
            "index.html"
        } else {
            entry
        };
        self.status().running
            && self.status.entry.as_deref() == Some(entry)
            && fs::canonicalize(workspace.join(entry))
                .ok()
                .and_then(|p| p.parent().map(Path::to_path_buf))
                .as_ref()
                == Some(&self.root)
    }

    pub fn open_chrome(&mut self) -> Result<Status, String> {
        if !self.status().running {
            return Err("The preview server stopped. Start it again.".into());
        }
        let executable = chrome_path().ok_or(
            "Google Chrome was not found. Install Chrome or open the preview URL in your browser.",
        )?;
        let mut command = Command::new(executable);
        command.arg(self.status.url.as_ref().unwrap());
        // Chrome is the user's browser, not part of an approved shell's kill-on-close job.
        // It is intentionally independent of preview lifetime; stopping closes the listener.
        command
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let mut child = command
            .spawn()
            .map_err(|e| format!("Could not open Chrome: {e}"))?;
        thread::spawn(move || {
            let _ = child.wait();
        });
        self.status.chrome_opened = true;
        Ok(self.status())
    }
}

impl Drop for PreviewServer {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn chrome_path() -> Option<PathBuf> {
    #[cfg(windows)]
    let candidates: Vec<PathBuf> = ["PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(|root| PathBuf::from(root).join("Google/Chrome/Application/chrome.exe"))
        .collect();
    #[cfg(not(windows))]
    let candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    .map(PathBuf::from);
    candidates
        .into_iter()
        .find(|p| p.is_absolute() && p.is_file())
}

fn validate_relative(path: &str) -> Result<(), String> {
    if path.is_empty()
        || path.len() > 2048
        || path.contains(['\\', ':', '\0'])
        || path
            .split('/')
            .any(|p| p.is_empty() || p.starts_with('.') || p.ends_with('.') || p.ends_with(' '))
        || Path::new(path)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(
            "Use a relative site file path without hidden folders or parent traversal.".into(),
        );
    }
    Ok(())
}

fn content_type(path: &str) -> Option<&'static str> {
    Some(
        match Path::new(path)
            .extension()?
            .to_str()?
            .to_ascii_lowercase()
            .as_str()
        {
            "html" | "htm" => "text/html; charset=utf-8",
            "css" => "text/css; charset=utf-8",
            "js" | "mjs" => "text/javascript; charset=utf-8",
            "json" | "map" => "application/json",
            "txt" => "text/plain; charset=utf-8",
            "svg" => "image/svg+xml",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "avif" => "image/avif",
            "ico" => "image/x-icon",
            "woff" => "font/woff",
            "woff2" => "font/woff2",
            "ttf" => "font/ttf",
            "wasm" => "application/wasm",
            _ => return None,
        },
    )
}

fn read_asset(root: &Path, relative: &str) -> Result<Vec<u8>, String> {
    validate_relative(relative)?;
    if content_type(relative).is_none() {
        return Err("Unsupported preview asset type.".into());
    }
    let sandbox = Sandbox::new(root, false, Duration::from_secs(1)).map_err(|e| e.to_string())?;
    if sandbox.root() != root {
        return Err("The preview folder changed.".into());
    }
    let path = sandbox.resolve(relative, true)?;
    validate_relative(
        &path
            .strip_prefix(root)
            .map_err(|_| "Preview escaped the site folder.")?
            .to_string_lossy()
            .replace('\\', "/"),
    )?;
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let file = options.open(&path).map_err(|e| e.to_string())?;
    if !file
        .metadata()
        .is_ok_and(|m| m.is_file() && m.len() <= MAX_ASSET)
    {
        return Err("Preview assets must be regular files no larger than 16 MB.".into());
    }
    if fs::canonicalize(&path).map_err(|e| e.to_string())? != path {
        return Err("The preview file changed.".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_ASSET + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_ASSET {
        return Err("Preview asset exceeds 16 MB.".into());
    }
    Ok(bytes)
}

fn decode_path(raw: &str) -> Result<String, ()> {
    let mut bytes = Vec::new();
    let mut input = raw.as_bytes().iter().copied();
    while let Some(byte) = input.next() {
        if byte == b'%' {
            let high = (input.next().ok_or(())? as char).to_digit(16).ok_or(())?;
            let low = (input.next().ok_or(())? as char).to_digit(16).ok_or(())?;
            bytes.push((high * 16 + low) as u8);
        } else {
            bytes.push(byte);
        }
    }
    String::from_utf8(bytes).map_err(|_| ())
}

async fn serve_asset(
    State(site): State<Arc<Site>>,
    method: Method,
    headers: HeaderMap,
    uri: Uri,
) -> Response {
    if !matches!(method, Method::GET | Method::HEAD) {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    if headers.get(header::HOST).and_then(|s| s.to_str().ok()) != Some(&site.host)
        || headers
            .get(header::ORIGIN)
            .is_some_and(|s| s.to_str().ok() != Some(&site.origin))
        || (headers
            .get("sec-fetch-site")
            .is_some_and(|s| s == "cross-site")
            && !(headers
                .get("sec-fetch-mode")
                .is_some_and(|s| s == "navigate")
                && headers
                    .get("sec-fetch-dest")
                    .is_some_and(|s| s == "document")))
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    let Ok(mut path) = decode_path(uri.path().strip_prefix('/').unwrap_or_default()) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if path.is_empty() || path.ends_with('/') {
        path.push_str("index.html");
    }
    let Some(kind) = content_type(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let root = site.root.clone();
    match tokio::task::spawn_blocking(move || read_asset(&root, &path)).await {
        Ok(Ok(bytes)) => {
            let length = bytes.len().to_string();
            let mut response = if method == Method::HEAD {
                Vec::new()
            } else {
                bytes
            }
            .into_response();
            let h = response.headers_mut();
            h.insert(header::CONTENT_TYPE, kind.parse().unwrap());
            h.insert(header::CONTENT_LENGTH, length.parse().unwrap());
            h.insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
            h.insert("x-content-type-options", "nosniff".parse().unwrap());
            h.insert("referrer-policy", "no-referrer".parse().unwrap());
            response
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

use std::future::IntoFuture;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, ErrorKind, Write};
    use std::net::SocketAddr;
    fn request(address: SocketAddr, method: &str, path: &str, host: &str) -> String {
        let mut socket = std::net::TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        write!(
            socket,
            "{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut result = String::new();
        socket.read_to_string(&mut result).unwrap();
        result
    }
    #[test]
    fn managed_preview_serves_assets_updates_and_stops() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("site")).unwrap();
        fs::write(
            dir.path().join("site/index.html"),
            "<script src='/app.js'></script>",
        )
        .unwrap();
        fs::write(dir.path().join("site/app.js"), "console.log('first')").unwrap();
        fs::write(dir.path().join("outside.txt"), "private").unwrap();
        fs::write(dir.path().join("site/.env"), "secret").unwrap();
        let server = PreviewServer::start(dir.path(), "site/index.html").unwrap();
        let parsed = url::Url::parse(server.status.url.as_ref().unwrap()).unwrap();
        let addr: SocketAddr = format!("127.0.0.1:{}", parsed.port().unwrap())
            .parse()
            .unwrap();
        let host = addr.to_string();
        assert!(request(addr, "GET", "/index.html", &host).contains("<script"));
        assert!(request(addr, "GET", "/app.js", &host).contains("text/javascript"));
        fs::write(dir.path().join("site/app.js"), "updated").unwrap();
        assert!(request(addr, "GET", "/app.js?v=2", &host).ends_with("updated"));
        assert!(!request(addr, "HEAD", "/app.js", &host).contains("updated"));
        for path in [
            "/../outside.txt",
            "/%2e%2e/outside.txt",
            "/.env",
            "/C%3A/secret.txt",
            "/%5coutside.txt",
        ] {
            assert!(
                request(addr, "GET", path, &host).starts_with("HTTP/1.1 404"),
                "{path}"
            );
        }
        assert!(request(addr, "GET", "/index.html", "example.com").starts_with("HTTP/1.1 403"));
        assert!(request(
            addr,
            "GET",
            "/index.html",
            &format!("{host}\r\nSec-Fetch-Site: cross-site")
        )
        .starts_with("HTTP/1.1 403"));
        assert!(request(addr, "GET", "/index.html", &format!("{host}\r\nSec-Fetch-Site: cross-site\r\nSec-Fetch-Mode: navigate\r\nSec-Fetch-Dest: document")).starts_with("HTTP/1.1 200"));
        assert!(request(
            addr,
            "GET",
            "/index.html",
            &format!("{host}\r\nOrigin: https://example.com")
        )
        .starts_with("HTTP/1.1 403"));
        assert!(request(addr, "POST", "/index.html", &host).starts_with("HTTP/1.1 405"));

        // Hold an identified connection to this server across shutdown. Once
        // its listener closes, another parallel test can reuse the same port.
        let mut connection = std::net::TcpStream::connect(addr).unwrap();
        connection
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        write!(
            connection,
            "HEAD /index.html HTTP/1.1\r\nHost: {host}\r\nConnection: keep-alive\r\n\r\n"
        )
        .unwrap();
        let mut connection = BufReader::new(connection);
        let mut line = String::new();
        connection.read_line(&mut line).unwrap();
        assert!(line.starts_with("HTTP/1.1 200"));
        loop {
            line.clear();
            assert_ne!(connection.read_line(&mut line).unwrap(), 0);
            if line == "\r\n" {
                break;
            }
        }
        drop(server);
        match connection.read(&mut [0]) {
            Ok(0) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted
                ) => {}
            other => panic!("preview connection remained open after shutdown: {other:?}"),
        }
    }
    #[test]
    fn managed_preview_rejects_missing_entry_and_invalid_paths() {
        let dir = tempfile::tempdir().unwrap();
        assert!(PreviewServer::start(dir.path(), "missing.html").is_err());
        for path in [
            "../index.html",
            ".git/index.html",
            "C:/index.html",
            "a\\b.html",
            "a//b.html",
        ] {
            assert!(PreviewServer::start(dir.path(), path).is_err());
        }
    }

    #[test]
    fn managed_preview_rejects_linked_assets_outside_site() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.json"), "secret").unwrap();
        let link = dir.path().join("assets");
        #[cfg(windows)]
        {
            let status = Command::new("cmd.exe")
                .args(["/C", "mklink", "/J"])
                .arg(&link)
                .arg(outside.path())
                .output()
                .unwrap();
            assert!(status.status.success());
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), &link).unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        assert!(read_asset(&root, "assets/secret.json").is_err());
        #[cfg(windows)]
        fs::remove_dir(link).unwrap();
    }
}
