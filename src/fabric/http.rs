//! Minimal blocking HTTP/1.1 client shared by the fabric's probe and forward
//! paths.
//!
//! Response parsing is pure over byte slices, so the awkward parts — chunked
//! bodies, truncated frames, a node that answers 500 — are tested without a
//! server. Only [`connect_and_send`] touches a socket.
//!
//! This deliberately does not reuse `chat::client`: that module is private to
//! `chat`, is keyed on a resolved `SocketAddr` where fabric members are named
//! hosts, and carries SSE, bearer auth and tool-call handling that neither a
//! health probe nor a request forward should depend on.
//!
//! Every loop here that can wait on a peer consults its caller's [`Cancel`]
//! beside its own deadline, and cancellation is checked first, so a request
//! nobody wants any more is reported as abandoned rather than as a node that
//! ran out of time. That is what makes a blocking exchange stoppable at all:
//! the socket is dropped on the way out, which is the only thing that tells a
//! node to stop generating.

use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::panic::AssertUnwindSafe;
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use rustls::{ClientConfig, ClientConnection, StreamOwned};
use rustls_pki_types::ServerName;

use super::cancel::Cancel;
use super::transport::NodeTransport;

/// Never spend the whole budget dialling or authenticating one address; a
/// forward budget is minutes long and a node that has not completed transport
/// setup in five seconds is not about to.
const CONNECT_ATTEMPT_CAP: Duration = Duration::from_secs(5);

/// The host resolver has no portable cancellation API. Keep it off request and
/// probe threads, and cap both workers and queued work so a broken resolver
/// cannot grow one abandoned thread per incoming request. Callers still stop at
/// their own deadline; IP literals remain usable even if every worker is stuck.
const RESOLVER_WORKERS: usize = 4;
const RESOLVER_QUEUE_CAPACITY: usize = 64;
const RESOLVER_WAIT_SLICE: Duration = Duration::from_millis(100);
const RESOLVER_QUEUE_WAIT_SLICE: Duration = Duration::from_millis(10);

/// Bounds the header block and a chunked body's size line, so a peer that never
/// terminates either cannot grow a buffer without bound.
const MAX_HEAD_BYTES: usize = 64 * 1024;
const MAX_CHUNK_SIZE_LINE: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HttpError {
    Resolve(String),
    Connect(String),
    Io(String),
    Malformed(String),
    TooLarge(usize),
    /// The request could not be written safely; refused before any socket work.
    InvalidRequest(String),
    /// Local policy refused the transport before any request bytes were sent.
    Policy(String),
    /// TLS setup or the handshake failed before any request bytes were sent.
    Tls(String),
    /// The caller stopped wanting the answer before it arrived.
    Cancelled,
}

impl HttpError {
    /// Whether the peer provably never received any part of the request.
    ///
    /// Resolution, dialling, local transport policy, and TLS setup/authentication
    /// qualify. Every other transport variant is raised at or after the first
    /// write, and `write_all` does not say how many bytes left the socket, so the
    /// request may already be running on the peer. Answering "it may have arrived"
    /// whenever that is possible is what lets a caller re-send elsewhere without
    /// risking a second execution.
    ///
    /// [`HttpError::InvalidRequest`] is deliberately excluded: nothing was sent,
    /// but the request is malformed, so another peer would refuse it identically.
    /// [`HttpError::Cancelled`] is excluded for a different reason again — it
    /// can be raised before the first write, but re-sending would produce an
    /// answer for a caller that has already stopped waiting for one.
    pub(crate) fn peer_never_received_it(&self) -> bool {
        match self {
            Self::Resolve(_) | Self::Connect(_) | Self::Policy(_) | Self::Tls(_) => true,
            Self::Io(_)
            | Self::Malformed(_)
            | Self::TooLarge(_)
            | Self::InvalidRequest(_)
            | Self::Cancelled => false,
        }
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolve(detail) => write!(f, "cannot resolve host: {detail}"),
            Self::Connect(detail) => write!(f, "cannot connect: {detail}"),
            Self::Io(detail) => write!(f, "connection failed: {detail}"),
            Self::Malformed(detail) => write!(f, "malformed HTTP response: {detail}"),
            Self::TooLarge(limit) => write!(f, "response exceeded {limit} bytes"),
            Self::InvalidRequest(detail) => write!(f, "cannot build request: {detail}"),
            Self::Policy(detail) => write!(f, "node transport refused: {detail}"),
            Self::Tls(detail) => write!(f, "node TLS failed: {detail}"),
            Self::Cancelled => write!(f, "the answer was no longer wanted"),
        }
    }
}

enum NodeConnection {
    Plain(TcpStream),
    Tls(Box<StreamOwned<ClientConnection, TcpStream>>),
}

type ResolveResult = Result<Vec<SocketAddr>, String>;
type Lookup = Box<dyn FnOnce() -> ResolveResult + Send + 'static>;

struct ResolveJob {
    lookup: Lookup,
    reply: mpsc::Sender<ResolveResult>,
    deadline: Instant,
    cancel: Cancel,
}

struct ResolverPool {
    jobs: mpsc::SyncSender<ResolveJob>,
}

impl ResolverPool {
    fn new(worker_count: usize, queue_capacity: usize) -> Result<Self, String> {
        let (jobs, receiver) = mpsc::sync_channel::<ResolveJob>(queue_capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..worker_count {
            let receiver = Arc::clone(&receiver);
            std::thread::Builder::new()
                .name(format!("camelid-node-resolver-{index}"))
                .spawn(move || loop {
                    let job = {
                        let receiver = receiver
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        match receiver.recv() {
                            Ok(job) => job,
                            Err(_) => return,
                        }
                    };
                    if job.cancel.is_cancelled() || Instant::now() >= job.deadline {
                        let _ = job.reply.send(Err(
                            "resolution was no longer wanted before it started".to_string(),
                        ));
                        continue;
                    }
                    let answer = std::panic::catch_unwind(AssertUnwindSafe(job.lookup))
                        .unwrap_or_else(|_| Err("resolver worker caught a panic".to_string()));
                    let _ = job.reply.send(answer);
                })
                .map_err(|error| format!("could not start resolver worker: {error}"))?;
        }
        Ok(Self { jobs })
    }
}

fn resolver_sender() -> Result<mpsc::SyncSender<ResolveJob>, HttpError> {
    static RESOLVER: OnceLock<Mutex<Option<ResolverPool>>> = OnceLock::new();
    let resolver = RESOLVER.get_or_init(|| Mutex::new(None));
    let mut resolver = resolver
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if resolver.is_none() {
        *resolver = Some(
            ResolverPool::new(RESOLVER_WORKERS, RESOLVER_QUEUE_CAPACITY)
                .map_err(HttpError::Resolve)?,
        );
    }
    Ok(resolver
        .as_ref()
        .expect("resolver initialized above")
        .jobs
        .clone())
}

fn resolve_with_sender(
    jobs: &mpsc::SyncSender<ResolveJob>,
    deadline: Instant,
    cancel: &Cancel,
    lookup: Lookup,
) -> Result<Vec<SocketAddr>, HttpError> {
    if cancel.is_cancelled() {
        return Err(HttpError::Cancelled);
    }
    if Instant::now() >= deadline {
        return Err(HttpError::Resolve(
            "resolution exceeded the request deadline".to_string(),
        ));
    }
    let (reply, result) = mpsc::channel();
    let mut job = ResolveJob {
        lookup,
        reply,
        deadline,
        cancel: cancel.clone(),
    };
    loop {
        if cancel.is_cancelled() {
            return Err(HttpError::Cancelled);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(HttpError::Resolve(
                "resolution exceeded the request deadline while queued".to_string(),
            ));
        }
        match jobs.try_send(job) {
            Ok(()) => break,
            Err(mpsc::TrySendError::Full(returned)) => {
                job = returned;
                std::thread::sleep(remaining.min(RESOLVER_QUEUE_WAIT_SLICE));
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                return Err(HttpError::Resolve("resolver workers stopped".to_string()))
            }
        }
    }

    loop {
        if cancel.is_cancelled() {
            return Err(HttpError::Cancelled);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(HttpError::Resolve(
                "resolution exceeded the request deadline".to_string(),
            ));
        }
        match result.recv_timeout(remaining.min(RESOLVER_WAIT_SLICE)) {
            Ok(Ok(addrs)) => return Ok(addrs),
            Ok(Err(detail)) => return Err(HttpError::Resolve(detail)),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(HttpError::Resolve(
                    "resolver worker stopped without an answer".to_string(),
                ))
            }
        }
    }
}

fn resolve_host(
    host: &str,
    port: u16,
    deadline: Instant,
    cancel: &Cancel,
) -> Result<Vec<SocketAddr>, HttpError> {
    let bare_host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = bare_host.parse::<IpAddr>() {
        return Ok(vec![SocketAddr::new(ip, port)]);
    }
    let host = bare_host.to_string();
    let lookup_host = host.clone();
    resolve_with_sender(
        &resolver_sender()?,
        deadline,
        cancel,
        Box::new(move || {
            (lookup_host.as_str(), port)
                .to_socket_addrs()
                .map(|addrs| addrs.collect())
                .map_err(|error| format!("{host}: {error}"))
        }),
    )
}

impl Read for NodeConnection {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(stream) => stream.read(buffer),
            Self::Tls(stream) => stream.read(buffer),
        }
    }
}

impl Write for NodeConnection {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(stream) => stream.write(buffer),
            Self::Tls(stream) => stream.write(buffer),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(stream) => stream.flush(),
            Self::Tls(stream) => stream.flush(),
        }
    }
}

/// Status line plus body of an HTTP/1.1 response.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct HttpResponse {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
}

/// The response headers this client acts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResponseHead {
    pub(crate) status: u16,
    pub(crate) chunked: bool,
    pub(crate) content_length: Option<usize>,
    /// Lower-cased and stripped of parameters, so `text/event-stream;
    /// charset=utf-8` compares equal to `text/event-stream`.
    pub(crate) content_type: Option<String>,
}

impl ResponseHead {
    pub(crate) fn is_event_stream(&self) -> bool {
        self.content_type.as_deref() == Some("text/event-stream")
    }
}

struct HeaderSplit {
    headers_end: usize,
    body_start: usize,
}

fn find_header_end(raw: &[u8]) -> Option<HeaderSplit> {
    raw.windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|at| HeaderSplit {
            headers_end: at,
            body_start: at + 4,
        })
}

fn parse_status_line(line: &str) -> Result<u16, HttpError> {
    if !line.starts_with("HTTP/") {
        return Err(HttpError::Malformed(format!(
            "status line does not start with HTTP/: {line}"
        )));
    }
    let code = line
        .split(' ')
        .nth(1)
        .ok_or_else(|| HttpError::Malformed("status line has no code".to_string()))?;
    code.parse::<u16>()
        .map_err(|_| HttpError::Malformed(format!("status code `{code}` is not a number")))
}

/// Decode `Transfer-Encoding: chunked`.
fn dechunk(mut rest: &[u8], max_body: usize) -> Result<Vec<u8>, HttpError> {
    let mut out = Vec::new();
    loop {
        let line_end = rest
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| HttpError::Malformed("chunk size line unterminated".to_string()))?;
        let size_text = std::str::from_utf8(&rest[..line_end])
            .map_err(|_| HttpError::Malformed("chunk size is not UTF-8".to_string()))?;
        // A chunk extension (`1a;name=value`) is legal and ignorable.
        let size_text = size_text.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16).map_err(|_| {
            HttpError::Malformed(format!("chunk size `{size_text}` is not hexadecimal"))
        })?;
        rest = &rest[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if out.len() + size > max_body {
            return Err(HttpError::TooLarge(max_body));
        }
        if rest.len() < size + 2 {
            return Err(HttpError::Malformed("chunk body truncated".to_string()));
        }
        out.extend_from_slice(&rest[..size]);
        rest = &rest[size + 2..];
    }
}

/// Where an incremental [`ChunkDecoder`] is inside a chunked body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkState {
    /// Reading the hexadecimal size line.
    Size,
    /// Owed this many more payload bytes for the current chunk.
    Data(usize),
    /// Consuming the CRLF that terminates a chunk's payload.
    Crlf,
    /// The zero-size chunk arrived; the body is complete.
    Done,
}

/// A chunked body decoded as it arrives.
///
/// [`dechunk`] needs the whole body up front. A streamed body's frames straddle
/// socket reads, so this keeps the frame state across pushes. When the response
/// is not chunked it is a pass-through, which is what a node answering with
/// `Connection: close` and no length uses.
struct ChunkDecoder {
    chunked: bool,
    /// Bytes received but not yet framed.
    raw: Vec<u8>,
    state: ChunkState,
}

impl ChunkDecoder {
    fn new(chunked: bool) -> Self {
        Self {
            chunked,
            raw: Vec::new(),
            state: ChunkState::Size,
        }
    }

    fn finished(&self) -> bool {
        self.chunked && self.state == ChunkState::Done
    }

    /// Frame everything buffered so far, appending decoded payload to `out`.
    ///
    /// `max_chunk` bounds a single chunk, so a peer claiming a huge size is
    /// refused before anything is allocated for it. The total body is not
    /// bounded here: a caller streaming to a client never holds it all.
    fn push(&mut self, bytes: &[u8], max_chunk: usize, out: &mut Vec<u8>) -> Result<(), HttpError> {
        if !self.chunked {
            out.extend_from_slice(bytes);
            return Ok(());
        }
        self.raw.extend_from_slice(bytes);
        loop {
            match self.state {
                ChunkState::Done => return Ok(()),
                ChunkState::Size => {
                    let Some(line_end) = self.raw.windows(2).position(|w| w == b"\r\n") else {
                        // A peer that never terminates the size line must not
                        // grow this buffer without bound.
                        if self.raw.len() > MAX_CHUNK_SIZE_LINE {
                            return Err(HttpError::Malformed(
                                "chunk size line unterminated".to_string(),
                            ));
                        }
                        return Ok(());
                    };
                    let size_text = std::str::from_utf8(&self.raw[..line_end])
                        .map_err(|_| HttpError::Malformed("chunk size is not UTF-8".to_string()))?;
                    // A chunk extension (`1a;name=value`) is legal and ignorable.
                    let size_text = size_text.split(';').next().unwrap_or("").trim();
                    let size = usize::from_str_radix(size_text, 16).map_err(|_| {
                        HttpError::Malformed(format!("chunk size `{size_text}` is not hexadecimal"))
                    })?;
                    if size > max_chunk {
                        return Err(HttpError::TooLarge(max_chunk));
                    }
                    self.raw.drain(..line_end + 2);
                    self.state = if size == 0 {
                        // Any trailer after the terminal chunk is ignored: the
                        // connection closes next either way.
                        ChunkState::Done
                    } else {
                        ChunkState::Data(size)
                    };
                }
                ChunkState::Data(owed) => {
                    if self.raw.is_empty() {
                        return Ok(());
                    }
                    let take = owed.min(self.raw.len());
                    out.extend_from_slice(&self.raw[..take]);
                    self.raw.drain(..take);
                    self.state = if take == owed {
                        ChunkState::Crlf
                    } else {
                        ChunkState::Data(owed - take)
                    };
                }
                ChunkState::Crlf => {
                    if self.raw.len() < 2 {
                        return Ok(());
                    }
                    if &self.raw[..2] != b"\r\n" {
                        return Err(HttpError::Malformed(
                            "chunk payload not terminated by CRLF".to_string(),
                        ));
                    }
                    self.raw.drain(..2);
                    self.state = ChunkState::Size;
                }
            }
        }
    }
}

/// Parse the status line and the headers this client acts on. Pure.
fn parse_head(head: &str) -> Result<ResponseHead, HttpError> {
    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| HttpError::Malformed("empty response".to_string()))?;
    let status = parse_status_line(status_line)?;

    let mut parsed = ResponseHead {
        status,
        chunked: false,
        content_length: None,
        content_type: None,
    };
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        match name.as_str() {
            "transfer-encoding" => {
                parsed.chunked = value.to_ascii_lowercase().contains("chunked");
            }
            "content-length" => parsed.content_length = value.parse::<usize>().ok(),
            "content-type" => {
                let media = value.split(';').next().unwrap_or("").trim();
                parsed.content_type = Some(media.to_ascii_lowercase());
            }
            _ => {}
        }
    }
    Ok(parsed)
}

/// Parse a whole HTTP/1.1 response. Pure; see the tests at the bottom.
pub(crate) fn parse_response(raw: &[u8], max_body: usize) -> Result<HttpResponse, HttpError> {
    let split = find_header_end(raw)
        .ok_or_else(|| HttpError::Malformed("no header terminator".to_string()))?;
    let head = std::str::from_utf8(&raw[..split.headers_end])
        .map_err(|_| HttpError::Malformed("headers are not UTF-8".to_string()))?;
    let head = parse_head(head)?;

    let rest = &raw[split.body_start..];
    let body = if head.chunked {
        dechunk(rest, max_body)?
    } else if let Some(len) = head.content_length {
        if len > max_body {
            return Err(HttpError::TooLarge(max_body));
        }
        if rest.len() < len {
            return Err(HttpError::Malformed(format!(
                "body truncated: expected {len} bytes, got {}",
                rest.len()
            )));
        }
        rest[..len].to_vec()
    } else {
        rest.to_vec()
    };

    if body.len() > max_body {
        return Err(HttpError::TooLarge(max_body));
    }
    Ok(HttpResponse {
        status: head.status,
        body,
    })
}

/// Connect to the first address that accepts.
///
/// A fabric member is usually named rather than numbered, and one name commonly
/// resolves to several addresses — typically an AAAA ahead of an A. Trying only
/// the first would report a healthy node as offline whenever its leading address
/// is unroutable, so every address gets a turn until the deadline runs out.
fn connect_any(
    addrs: &[SocketAddr],
    deadline: Instant,
    cancel: &Cancel,
) -> Result<TcpStream, HttpError> {
    let mut last: Option<String> = None;
    for (index, addr) in addrs.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(HttpError::Cancelled);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        // Share what is left between the addresses still untried, so one that
        // black-holes instead of refusing cannot starve the ones behind it.
        let untried = (addrs.len() - index) as u32;
        let attempt = (remaining / untried)
            .min(CONNECT_ATTEMPT_CAP)
            .max(Duration::from_millis(1));
        match TcpStream::connect_timeout(addr, attempt) {
            Ok(stream) => return Ok(stream),
            Err(error) => last = Some(format!("{addr}: {error}")),
        }
    }
    Err(HttpError::Connect(last.unwrap_or_else(|| {
        "no address accepted a connection within the timeout".to_string()
    })))
}

fn tls_server_name(host: &str) -> Result<ServerName<'static>, HttpError> {
    let host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    ServerName::try_from(host.to_string())
        .map_err(|error| HttpError::Tls(format!("`{host}` is not a valid server name: {error}")))
}

fn negotiate_tls(
    mut stream: TcpStream,
    server_name: ServerName<'static>,
    config: Arc<ClientConfig>,
    deadline: Instant,
    cancel: &Cancel,
) -> Result<NodeConnection, HttpError> {
    let mut connection = ClientConnection::new(config, server_name)
        .map_err(|error| HttpError::Tls(error.to_string()))?;

    // Keep the handshake responsive to cancellation and the shared request
    // deadline. The larger body-write timeout is restored once authentication
    // has completed.
    stream
        .set_write_timeout(Some(Duration::from_millis(100)))
        .map_err(|error| HttpError::Tls(error.to_string()))?;
    while connection.is_handshaking() {
        if cancel.is_cancelled() {
            return Err(HttpError::Cancelled);
        }
        if Instant::now() >= deadline {
            return Err(HttpError::Tls(
                "handshake exceeded the request deadline".to_string(),
            ));
        }
        match connection.complete_io(&mut stream) {
            Ok(_) => {}
            Err(error) if is_retryable(&error) => continue,
            Err(error) => return Err(HttpError::Tls(error.to_string())),
        }
    }
    Ok(NodeConnection::Tls(Box::new(StreamOwned::new(
        connection, stream,
    ))))
}

fn connect_tls_any(
    addrs: &[SocketAddr],
    host: &str,
    config: Arc<ClientConfig>,
    deadline: Instant,
    write_timeout: Duration,
    cancel: &Cancel,
) -> Result<NodeConnection, HttpError> {
    let server_name = tls_server_name(host)?;
    let mut last: Option<HttpError> = None;
    for (index, addr) in addrs.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(HttpError::Cancelled);
        }
        let now = Instant::now();
        let remaining = deadline.saturating_duration_since(now);
        if remaining.is_zero() {
            break;
        }
        // The address is not usable until its certificate authenticates. Give
        // each untried address a bounded share of the remaining budget so a
        // TCP endpoint that stalls or rejects TLS cannot hide a valid sibling.
        let untried = (addrs.len() - index) as u32;
        let attempt = (remaining / untried)
            .min(CONNECT_ATTEMPT_CAP)
            .max(Duration::from_millis(1));
        let attempt_deadline = now + attempt;
        let stream = match TcpStream::connect_timeout(addr, attempt) {
            Ok(stream) => stream,
            Err(error) => {
                last = Some(HttpError::Connect(format!("{addr}: {error}")));
                continue;
            }
        };
        if let Err(error) = stream.set_read_timeout(Some(Duration::from_millis(100))) {
            last = Some(HttpError::Tls(format!(
                "could not configure {addr} for TLS: {error}"
            )));
            continue;
        }
        match negotiate_tls(
            stream,
            server_name.clone(),
            Arc::clone(&config),
            attempt_deadline,
            cancel,
        ) {
            Ok(connection) => {
                let configured = match &connection {
                    NodeConnection::Tls(stream) => {
                        stream.sock.set_write_timeout(Some(write_timeout))
                    }
                    NodeConnection::Plain(_) => unreachable!("TLS negotiation returns TLS"),
                };
                if let Err(error) = configured {
                    last = Some(HttpError::Tls(error.to_string()));
                    continue;
                }
                return Ok(connection);
            }
            Err(HttpError::Cancelled) => return Err(HttpError::Cancelled),
            Err(error) => last = Some(error),
        }
    }
    Err(last.unwrap_or_else(|| {
        HttpError::Tls("no address authenticated within the connection budget".to_string())
    }))
}

/// Build the request head. Pure, so the header set — including whether an
/// `Authorization` line is present at all — is tested without a server.
///
/// A bearer token is written into the head verbatim, so one carrying a control
/// character could append headers of its own. That is refused rather than sent.
/// Neither the error nor anything else here repeats the token's value.
fn request_head(
    method: &str,
    path: &str,
    authority: &str,
    accept: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
) -> Result<String, HttpError> {
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {authority}\r\nAccept: {accept}\r\nConnection: close\r\n"
    );
    if let Some(token) = bearer {
        if token.is_empty() {
            return Err(HttpError::InvalidRequest(
                "bearer token is empty; pass no token rather than an empty one".to_string(),
            ));
        }
        if token.chars().any(char::is_control) {
            return Err(HttpError::InvalidRequest(
                "bearer token contains a control character".to_string(),
            ));
        }
        head.push_str(&format!("Authorization: Bearer {token}\r\n"));
    }
    if let Some(body) = body {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    Ok(head)
}

/// Media type sent as `Accept` on a normal request, and on one that expects a
/// server-sent event stream back.
pub(crate) const ACCEPT_JSON: &str = "application/json";
pub(crate) const ACCEPT_EVENT_STREAM: &str = "text/event-stream";

/// Resolve, connect, and write one request, leaving the socket ready to read.
///
/// Shared by [`request_with_transport`] and [`open_stream_with_transport`] so
/// both dial, authenticate and frame a request exactly the same way.
#[allow(clippy::too_many_arguments)]
fn connect_and_send(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    accept: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    deadline: Instant,
    write_timeout: Duration,
    cancel: &Cancel,
    transport: &NodeTransport,
) -> Result<NodeConnection, HttpError> {
    // Before resolving, which is itself a blocking call: a caller that has
    // already given up costs its node nothing at all, not even a connection.
    if cancel.is_cancelled() {
        return Err(HttpError::Cancelled);
    }

    let authority = format!("{host}:{port}");
    // Built before resolving, so a request that cannot be written safely is
    // refused without touching the network.
    let head = request_head(method, path, &authority, accept, body, bearer)?;

    let addrs = resolve_host(host, port, deadline, cancel)?;
    if addrs.is_empty() {
        return Err(HttpError::Resolve(
            "host resolved to no addresses".to_string(),
        ));
    }

    let addrs = transport
        .permitted_addresses(&addrs)
        .map_err(|error| HttpError::Policy(error.to_string()))?;
    let mut stream = match transport.tls_config() {
        Some(config) => connect_tls_any(&addrs, host, config, deadline, write_timeout, cancel)?,
        None => {
            let stream = connect_any(&addrs, deadline, cancel)?;
            // Short socket reads keep the caller's loop responsive to its own
            // deadline; one long timeout would overshoot on a stalled peer.
            stream
                .set_read_timeout(Some(Duration::from_millis(100)))
                .map_err(|error| HttpError::Io(error.to_string()))?;
            stream
                .set_write_timeout(Some(write_timeout))
                .map_err(|error| HttpError::Io(error.to_string()))?;
            NodeConnection::Plain(stream)
        }
    };

    stream
        .write_all(head.as_bytes())
        .map_err(|error| HttpError::Io(error.to_string()))?;
    if let Some(body) = body {
        stream
            .write_all(body)
            .map_err(|error| HttpError::Io(error.to_string()))?;
    }
    Ok(stream)
}

/// Perform one request/response round trip against a node.
///
/// `timeout` bounds the whole exchange, not each socket operation, so a peer
/// that dribbles bytes forever still fails on schedule.
///
/// `bearer` is sent as `Authorization: Bearer`, which is what a node started
/// with an API key requires on every route but `/v1/health`.
///
/// `cancel` ends the exchange early and drops the socket with it. Pass
/// [`Cancel::never`] where there is no client that can go away.
// One more parameter than clippy's threshold; every one of them is a distinct
// property of a single round trip, so bundling them would only move the list.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn request(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    timeout: Duration,
    max_body: usize,
    cancel: &Cancel,
) -> Result<HttpResponse, HttpError> {
    request_with_transport(
        host,
        port,
        method,
        path,
        body,
        bearer,
        timeout,
        max_body,
        cancel,
        &NodeTransport::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn request_with_transport(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    timeout: Duration,
    max_body: usize,
    cancel: &Cancel,
    transport: &NodeTransport,
) -> Result<HttpResponse, HttpError> {
    let deadline = Instant::now() + timeout;
    let mut stream = connect_and_send(
        host,
        port,
        method,
        path,
        ACCEPT_JSON,
        body,
        bearer,
        deadline,
        timeout,
        cancel,
        transport,
    )?;

    let mut raw = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        // Before the deadline, so an abandoned request is reported as abandoned
        // rather than as a node that was too slow.
        if cancel.is_cancelled() {
            return Err(HttpError::Cancelled);
        }
        if Instant::now() >= deadline {
            return Err(HttpError::Io("request exceeded its deadline".to_string()));
        }
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                raw.extend_from_slice(&chunk[..read]);
                if raw.len() > max_body {
                    return Err(HttpError::TooLarge(max_body));
                }
            }
            Err(error) if is_retryable(&error) => continue,
            // rustls reports an EOF without close_notify as UnexpectedEof.
            // HTTP framing can still prove the authenticated response whole:
            // Content-Length and the terminal chunk both make truncation
            // detectable. Never apply this to a close-delimited or incomplete
            // body, where the TLS close is the only end marker.
            Err(error)
                if error.kind() == std::io::ErrorKind::UnexpectedEof
                    && parse_response(&raw, max_body).is_ok() =>
            {
                break;
            }
            Err(error) => return Err(HttpError::Io(error.to_string())),
        }
    }

    parse_response(&raw, max_body)
}

/// A socket read that timed out rather than failed: the 100ms read timeout
/// fires constantly while a node is still thinking, and is not an error.
fn is_retryable(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
}

/// A response whose body is read incrementally rather than all at once.
///
/// The head has already been read and parsed; the body is delivered by
/// [`ResponseStream::next_chunk`] as it arrives. Nothing here interprets the
/// payload — a caller relaying server-sent events forwards the bytes verbatim,
/// so an event field this client has never heard of cannot be mangled.
pub(crate) struct ResponseStream {
    stream: NodeConnection,
    head: ResponseHead,
    decoder: ChunkDecoder,
    /// Body bytes that arrived alongside the head, not yet framed.
    pending: Vec<u8>,
    /// Decoded payload handed to the caller so far, which a declared
    /// `Content-Length` is measured against.
    delivered: usize,
    /// How long the node may send nothing at all before it counts as wedged.
    idle_timeout: Duration,
    max_chunk: usize,
    /// Carried rather than passed per call: the stream outlives the call that
    /// opened it, and every read it does afterwards belongs to the same client.
    cancel: Cancel,
}

/// A body that stopped before its framing said it would.
fn truncated_body() -> HttpError {
    HttpError::Malformed("connection closed before the response body completed".to_string())
}

/// Send a request and read only its head, leaving the body to be streamed.
///
/// `cancel` bounds the wait for that head and is kept by the returned stream,
/// so it bounds every later read too.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn open_stream(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    head_timeout: Duration,
    idle_timeout: Duration,
    max_chunk: usize,
    cancel: &Cancel,
) -> Result<ResponseStream, HttpError> {
    open_stream_with_transport(
        host,
        port,
        method,
        path,
        body,
        bearer,
        head_timeout,
        idle_timeout,
        max_chunk,
        cancel,
        &NodeTransport::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn open_stream_with_transport(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    bearer: Option<&str>,
    head_timeout: Duration,
    idle_timeout: Duration,
    max_chunk: usize,
    cancel: &Cancel,
    transport: &NodeTransport,
) -> Result<ResponseStream, HttpError> {
    let deadline = Instant::now() + head_timeout;
    let mut stream = connect_and_send(
        host,
        port,
        method,
        path,
        ACCEPT_EVENT_STREAM,
        body,
        bearer,
        deadline,
        head_timeout,
        cancel,
        transport,
    )?;

    let mut raw = Vec::new();
    let mut scratch = [0_u8; 8192];
    let split = loop {
        if let Some(split) = find_header_end(&raw) {
            break split;
        }
        if raw.len() > MAX_HEAD_BYTES {
            return Err(HttpError::TooLarge(MAX_HEAD_BYTES));
        }
        // A head can be a whole prefill away, which is the longest a client is
        // ever left with nothing to read; leaving is exactly what it does then.
        if cancel.is_cancelled() {
            return Err(HttpError::Cancelled);
        }
        if Instant::now() >= deadline {
            return Err(HttpError::Io(
                "node sent no response head before the deadline".to_string(),
            ));
        }
        match stream.read(&mut scratch) {
            Ok(0) => {
                return Err(HttpError::Malformed(
                    "connection closed before the response head completed".to_string(),
                ))
            }
            Ok(read) => raw.extend_from_slice(&scratch[..read]),
            Err(error) if is_retryable(&error) => continue,
            Err(error) => return Err(HttpError::Io(error.to_string())),
        }
    };

    let head_text = std::str::from_utf8(&raw[..split.headers_end])
        .map_err(|_| HttpError::Malformed("headers are not UTF-8".to_string()))?;
    let head = parse_head(head_text)?;

    Ok(ResponseStream {
        decoder: ChunkDecoder::new(head.chunked),
        pending: raw[split.body_start..].to_vec(),
        delivered: 0,
        head,
        stream,
        idle_timeout,
        max_chunk,
        cancel: cancel.clone(),
    })
}

impl ResponseStream {
    pub(crate) fn head(&self) -> &ResponseHead {
        &self.head
    }

    /// The next piece of decoded body, or `None` once the body is complete.
    ///
    /// A node that dies mid-body raises rather than ending the stream. The
    /// earlier bytes have already reached the client and cannot be taken back,
    /// but relaying this as a clean end would frame a half-generation as a whole
    /// one — the distinction chunked framing exists to carry.
    pub(crate) fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, HttpError> {
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.decoder.push(&pending, self.max_chunk, &mut out)?;
            if !out.is_empty() {
                self.delivered += out.len();
                return Ok(Some(out));
            }
        }
        if self.complete() == Some(true) {
            return Ok(None);
        }

        let deadline = Instant::now() + self.idle_timeout;
        let mut scratch = [0_u8; 8192];
        loop {
            if self.cancel.is_cancelled() {
                return Err(HttpError::Cancelled);
            }
            if Instant::now() >= deadline {
                return Err(HttpError::Io(
                    "node sent nothing before the idle timeout".to_string(),
                ));
            }
            match self.stream.read(&mut scratch) {
                Ok(0) if self.complete() == Some(false) => return Err(truncated_body()),
                Ok(0) => return Ok(None),
                Ok(read) => {
                    self.decoder
                        .push(&scratch[..read], self.max_chunk, &mut out)?;
                    if !out.is_empty() {
                        self.delivered += out.len();
                        return Ok(Some(out));
                    }
                    if self.complete() == Some(true) {
                        return Ok(None);
                    }
                }
                Err(error) if is_retryable(&error) => continue,
                Err(error) => return Err(HttpError::Io(error.to_string())),
            }
        }
    }

    /// Read the rest of the body into memory, for a response that is not the
    /// stream the caller asked for and is therefore small and bounded.
    pub(crate) fn into_buffered(mut self, max_body: usize) -> Result<HttpResponse, HttpError> {
        let mut body = Vec::new();
        let pending = std::mem::take(&mut self.pending);
        self.decoder.push(&pending, self.max_chunk, &mut body)?;
        self.delivered = body.len();

        let deadline = Instant::now() + self.idle_timeout;
        let mut scratch = [0_u8; 8192];
        while self.complete() != Some(true) {
            if self.cancel.is_cancelled() {
                return Err(HttpError::Cancelled);
            }
            if Instant::now() >= deadline {
                return Err(HttpError::Io(
                    "node sent nothing before the idle timeout".to_string(),
                ));
            }
            match self.stream.read(&mut scratch) {
                Ok(0) => break,
                Ok(read) => {
                    self.decoder
                        .push(&scratch[..read], self.max_chunk, &mut body)?;
                    self.delivered = body.len();
                    if body.len() > max_body {
                        return Err(HttpError::TooLarge(max_body));
                    }
                }
                Err(error) if is_retryable(&error) => continue,
                Err(error) => return Err(HttpError::Io(error.to_string())),
            }
        }

        // The one-shot path calls these same bytes malformed; the two readers
        // must not disagree about whether a half-delivered body is an answer.
        if self.complete() == Some(false) {
            return Err(truncated_body());
        }
        if body.len() > max_body {
            return Err(HttpError::TooLarge(max_body));
        }
        Ok(HttpResponse {
            status: self.head.status,
            body,
        })
    }

    /// Whether the body's own framing says it is complete.
    ///
    /// `None` when the response declares no framing at all: the connection
    /// closing is then the only thing that can end it, so an EOF there is a
    /// clean end rather than a truncation.
    fn complete(&self) -> Option<bool> {
        if self.head.chunked {
            return Some(self.decoder.finished());
        }
        self.head.content_length.map(|len| self.delivered >= len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::mpsc;

    const LIMIT: usize = 1024 * 1024;

    #[test]
    fn a_stalled_resolution_cannot_outlive_the_request_deadline() {
        let pool = ResolverPool::new(1, 1).expect("resolver pool starts");
        let (release, blocked) = mpsc::channel();
        let started = Instant::now();
        let error = resolve_with_sender(
            &pool.jobs,
            started + Duration::from_millis(150),
            &Cancel::never(),
            Box::new(move || {
                blocked.recv().expect("test releases resolver");
                Ok(vec![SocketAddr::from(([127, 0, 0, 1], 8181))])
            }),
        )
        .expect_err("a stalled resolver must not escape the request deadline");
        assert!(matches!(error, HttpError::Resolve(_)), "{error:?}");
        assert!(started.elapsed() < Duration::from_secs(1));
        release.send(()).expect("release resolver worker");
    }

    #[test]
    fn cancellation_stops_waiting_for_a_blocked_resolution() {
        let pool = Arc::new(ResolverPool::new(1, 1).expect("resolver pool starts"));
        let cancel = Cancel::new();
        let handed_to_waiter = cancel.clone();
        let (release, blocked) = mpsc::channel();
        let started = Instant::now();
        let waiter = {
            let pool = Arc::clone(&pool);
            std::thread::spawn(move || {
                resolve_with_sender(
                    &pool.jobs,
                    Instant::now() + Duration::from_secs(5),
                    &handed_to_waiter,
                    Box::new(move || {
                        blocked.recv().expect("test releases resolver");
                        Ok(vec![SocketAddr::from(([127, 0, 0, 1], 8181))])
                    }),
                )
            })
        };
        std::thread::sleep(Duration::from_millis(50));
        cancel.cancel();
        let error = waiter
            .join()
            .expect("waiter exits")
            .expect_err("cancelled resolution is abandoned");
        assert_eq!(error, HttpError::Cancelled);
        assert!(started.elapsed() < Duration::from_secs(1));
        release.send(()).expect("release resolver worker");
    }

    #[test]
    fn a_saturated_resolver_queue_backpressures_until_the_deadline() {
        let pool = ResolverPool::new(1, 1).expect("resolver pool starts");
        let first_jobs = pool.jobs.clone();
        let (first_started, wait_for_first) = mpsc::channel();
        let (release_first, first_blocked) = mpsc::channel();
        let first = std::thread::spawn(move || {
            resolve_with_sender(
                &first_jobs,
                Instant::now() + Duration::from_secs(5),
                &Cancel::never(),
                Box::new(move || {
                    first_started.send(()).expect("announce first lookup");
                    first_blocked.recv().expect("release first lookup");
                    Ok(vec![SocketAddr::from(([127, 0, 0, 1], 8181))])
                }),
            )
        });
        wait_for_first.recv().expect("first lookup started");

        let second_jobs = pool.jobs.clone();
        let (release_second, second_blocked) = mpsc::channel();
        let second = std::thread::spawn(move || {
            resolve_with_sender(
                &second_jobs,
                Instant::now() + Duration::from_secs(5),
                &Cancel::never(),
                Box::new(move || {
                    second_blocked.recv().expect("release second lookup");
                    Ok(vec![SocketAddr::from(([127, 0, 0, 1], 8182))])
                }),
            )
        });
        std::thread::sleep(Duration::from_millis(50));

        let started = Instant::now();
        let error = resolve_with_sender(
            &pool.jobs,
            started + Duration::from_millis(150),
            &Cancel::never(),
            Box::new(|| Ok(vec![SocketAddr::from(([127, 0, 0, 1], 8183))])),
        )
        .expect_err("a full queue must remain bounded by the caller deadline");
        assert!(matches!(error, HttpError::Resolve(_)), "{error:?}");
        assert!(started.elapsed() >= Duration::from_millis(100));
        assert!(started.elapsed() < Duration::from_secs(1));

        release_first.send(()).expect("release first lookup");
        release_second.send(()).expect("release second lookup");
        first
            .join()
            .expect("first waiter exits")
            .expect("first resolves");
        second
            .join()
            .expect("second waiter exits")
            .expect("second resolves");
    }

    #[test]
    fn a_panicking_lookup_does_not_retire_its_resolver_worker() {
        let pool = ResolverPool::new(1, 1).expect("resolver pool starts");
        let error = resolve_with_sender(
            &pool.jobs,
            Instant::now() + Duration::from_secs(1),
            &Cancel::never(),
            Box::new(|| panic!("synthetic resolver panic")),
        )
        .expect_err("resolver panic becomes an error");
        assert!(matches!(error, HttpError::Resolve(_)), "{error:?}");

        assert_eq!(
            resolve_with_sender(
                &pool.jobs,
                Instant::now() + Duration::from_secs(1),
                &Cancel::never(),
                Box::new(|| Ok(vec![SocketAddr::from(([127, 0, 0, 1], 8181))])),
            )
            .expect("worker still resolves"),
            vec![SocketAddr::from(([127, 0, 0, 1], 8181))]
        );
    }

    #[test]
    fn a_burst_of_resolutions_is_backpressured_not_rejected() {
        const CALLERS: usize = 128;
        let pool = Arc::new(ResolverPool::new(4, 64).expect("resolver pool starts"));
        let start = Arc::new(std::sync::Barrier::new(CALLERS + 1));
        let handles: Vec<_> = (0..CALLERS)
            .map(|_| {
                let jobs = pool.jobs.clone();
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    resolve_with_sender(
                        &jobs,
                        Instant::now() + Duration::from_secs(5),
                        &Cancel::never(),
                        Box::new(|| Ok(vec![SocketAddr::from(([127, 0, 0, 1], 8181))])),
                    )
                })
            })
            .collect();
        start.wait();

        for handle in handles {
            let addrs = handle
                .join()
                .expect("resolver caller exits")
                .expect("resolution is not rejected under a burst");
            assert!(!addrs.is_empty());
        }
    }

    fn tls_material(names: Vec<String>) -> (Arc<rustls::ServerConfig>, NodeTransport) {
        let issued = rcgen::generate_simple_self_signed(names).expect("generate certificate");
        let key = rustls_pki_types::PrivatePkcs8KeyDer::from(issued.key_pair.serialize_der());
        let server = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![issued.cert.der().clone()], key.into())
            .expect("server certificate and key agree");

        let directory = tempfile::tempdir().expect("temp dir");
        let ca_path = directory.path().join("node-ca");
        std::fs::write(&ca_path, rcgen::Certificate::pem(&issued.cert)).expect("write CA");
        let transport = NodeTransport::resolve(Some(&ca_path), false).expect("load CA");
        (Arc::new(server), transport)
    }

    fn serve_tls_once(
        server: Arc<rustls::ServerConfig>,
        response: &'static [u8],
    ) -> (u16, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind TLS node");
        let port = listener.local_addr().expect("local address").port();
        let handle = std::thread::spawn(move || {
            let (socket, _) = listener.accept().expect("accept TLS client");
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("bound reads");
            socket
                .set_write_timeout(Some(Duration::from_secs(2)))
                .expect("bound writes");
            let connection = rustls::ServerConnection::new(server).expect("server connection");
            let mut stream = rustls::StreamOwned::new(connection, socket);
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            // The declared body has to be drained as well as the head. The
            // client writes them separately, so the body can still be queued
            // unread when this thread returns, and closing a socket that holds
            // unread bytes is an abortive close (RST) rather than a FIN. The
            // client then observes ECONNRESET instead of the response that was
            // just written. This is the same completeness check `canned_node`
            // applies on the plaintext path.
            while !request_complete(&request) {
                let read = match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => return,
                    Ok(read) => read,
                };
                request.extend_from_slice(&buffer[..read]);
            }
            stream.write_all(response).expect("write TLS response");
            stream.flush().expect("flush TLS response");
        });
        (port, handle)
    }

    #[test]
    fn a_ca_authenticated_node_serves_a_real_https_request() {
        let (server, transport) = tls_material(vec!["localhost".to_string()]);
        let (port, handle) = serve_tls_once(
            server,
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
        );

        let response = request_with_transport(
            "localhost",
            port,
            "GET",
            "/v1/health",
            None,
            None,
            Duration::from_secs(2),
            LIMIT,
            &Cancel::never(),
            &transport,
        )
        .expect("trusted TLS node answers");
        assert_eq!(response.status, 200);
        assert_eq!(response.body, br#"{"ok":true}"#);
        handle.join().expect("TLS node exits");
    }

    #[test]
    fn an_ip_literal_is_verified_against_an_ip_subject_alt_name() {
        let (server, transport) = tls_material(vec!["127.0.0.1".to_string()]);
        let (port, handle) = serve_tls_once(
            server,
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
        );
        let response = request_with_transport(
            "127.0.0.1",
            port,
            "GET",
            "/v1/health",
            None,
            None,
            Duration::from_secs(2),
            LIMIT,
            &Cancel::never(),
            &transport,
        )
        .expect("IP SAN matches the node IP literal");
        assert_eq!(response.status, 200);
        handle.join().expect("IP SAN TLS node exits");
    }

    #[test]
    fn a_streaming_response_crosses_the_same_authenticated_tls_transport() {
        let (server, transport) = tls_material(vec!["localhost".to_string()]);
        let body = b"data: {\"token\":\"hi\"}\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let response = Box::leak([response.as_bytes(), body].concat().into_boxed_slice());
        let (port, handle) = serve_tls_once(server, response);

        let mut stream = open_stream_with_transport(
            "localhost",
            port,
            "POST",
            "/v1/chat/completions",
            Some(b"{}"),
            None,
            Duration::from_secs(2),
            Duration::from_secs(2),
            LIMIT,
            &Cancel::never(),
            &transport,
        )
        .expect("trusted TLS stream opens");
        assert!(stream.head().is_event_stream());
        assert_eq!(
            stream.next_chunk().expect("reads event"),
            Some(body.to_vec())
        );
        assert_eq!(stream.next_chunk().expect("stream ends"), None);
        handle.join().expect("TLS node exits");
    }

    #[test]
    fn a_tls_stream_cut_off_before_its_declared_end_is_not_a_clean_eof() {
        let (server, transport) = tls_material(vec!["localhost".to_string()]);
        let body = b"data: partial\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len() + 20
        );
        let response = Box::leak([response.as_bytes(), body].concat().into_boxed_slice());
        let (port, handle) = serve_tls_once(server, response);
        let mut stream = open_stream_with_transport(
            "localhost",
            port,
            "POST",
            "/v1/chat/completions",
            Some(b"{}"),
            None,
            Duration::from_secs(2),
            Duration::from_secs(2),
            LIMIT,
            &Cancel::never(),
            &transport,
        )
        .expect("TLS stream opens before the truncation");
        assert_eq!(
            stream.next_chunk().expect("partial event arrives"),
            Some(body.to_vec())
        );
        let error = stream
            .next_chunk()
            .expect_err("missing authenticated bytes are not a clean EOF");
        assert!(
            matches!(error, HttpError::Io(_) | HttpError::Malformed(_)),
            "{error:?}"
        );
        handle.join().expect("truncated TLS node exits");
    }

    #[test]
    fn an_untrusted_ca_and_a_wrong_server_name_are_refused_before_http() {
        let (server, _) = tls_material(vec!["localhost".to_string()]);
        let (_, wrong_ca) = tls_material(vec!["localhost".to_string()]);
        let (port, handle) = serve_tls_once(server, b"");
        let error = request_with_transport(
            "localhost",
            port,
            "GET",
            "/v1/health",
            None,
            None,
            Duration::from_secs(2),
            LIMIT,
            &Cancel::never(),
            &wrong_ca,
        )
        .expect_err("untrusted certificate is refused");
        assert!(matches!(error, HttpError::Tls(_)), "{error:?}");
        assert!(error.peer_never_received_it());
        handle.join().expect("untrusted TLS node exits");

        let (server, transport) = tls_material(vec!["node.example".to_string()]);
        let (port, handle) = serve_tls_once(server, b"");
        let error = request_with_transport(
            "localhost",
            port,
            "GET",
            "/v1/health",
            None,
            None,
            Duration::from_secs(2),
            LIMIT,
            &Cancel::never(),
            &transport,
        )
        .expect_err("name mismatch is refused");
        assert!(matches!(error, HttpError::Tls(_)), "{error:?}");
        handle.join().expect("wrong-name TLS node exits");
    }

    #[test]
    fn a_stalled_tls_address_does_not_hide_an_authenticated_sibling() {
        let (server, transport) = tls_material(vec!["localhost".to_string()]);
        let (release_impostor, wait_for_sibling) = mpsc::channel();

        let impostor = TcpListener::bind("127.0.0.1:0").expect("bind impostor");
        let impostor_addr = impostor.local_addr().expect("impostor address");
        let impostor_thread = std::thread::spawn(move || {
            let (mut socket, _) = impostor.accept().expect("accept first attempt");
            let mut client_hello = [0_u8; 1024];
            let _ = socket.read(&mut client_hello);
            // Keep the unauthenticated connection open. Only the trusted
            // sibling releases it, so reaching that sibling proves this
            // handshake was bounded rather than waited out indefinitely.
            wait_for_sibling
                .recv_timeout(Duration::from_secs(5))
                .expect("trusted sibling was attempted");
        });

        let trusted = TcpListener::bind("127.0.0.1:0").expect("bind trusted node");
        let trusted_addr = trusted.local_addr().expect("trusted address");
        let trusted_thread = std::thread::spawn(move || {
            let (mut socket, _) = trusted.accept().expect("accept second attempt");
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("bound reads");
            socket
                .set_write_timeout(Some(Duration::from_secs(2)))
                .expect("bound writes");
            let mut connection = rustls::ServerConnection::new(server).expect("create TLS server");
            while connection.is_handshaking() {
                connection
                    .complete_io(&mut socket)
                    .expect("complete TLS handshake");
            }
            release_impostor
                .send(())
                .expect("release stalled first address");
        });

        let connection = connect_tls_any(
            &[impostor_addr, trusted_addr],
            "localhost",
            transport.tls_config().expect("TLS config"),
            Instant::now() + Duration::from_secs(3),
            Duration::from_secs(2),
            &Cancel::never(),
        )
        .expect("the authenticated second address remains usable");
        assert!(matches!(connection, NodeConnection::Tls(_)));

        impostor_thread.join().expect("impostor exits");
        trusted_thread.join().expect("trusted node exits");
    }

    #[test]
    fn bearer_bytes_are_not_sent_before_the_tls_peer_is_authenticated() {
        let (_, transport) = tls_material(vec!["localhost".to_string()]);
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind impostor");
        let port = listener.local_addr().expect("local address").port();
        let (sent, received) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("accept client");
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("bound reads");
            let mut bytes = vec![0_u8; 8192];
            let read = socket.read(&mut bytes).expect("read ClientHello");
            bytes.truncate(read);
            sent.send(bytes).expect("send captured bytes");
            // Closing here makes the handshake fail immediately.
        });

        let secret = "bearer-must-not-cross-before-auth-9f72";
        let error = request_with_transport(
            "localhost",
            port,
            "GET",
            "/v1/health",
            None,
            Some(secret),
            Duration::from_secs(2),
            LIMIT,
            &Cancel::never(),
            &transport,
        )
        .expect_err("a non-TLS peer is refused");
        assert!(matches!(error, HttpError::Tls(_)), "{error:?}");
        let captured = received.recv().expect("captured ClientHello");
        assert!(
            !captured
                .windows(secret.len())
                .any(|window| window == secret.as_bytes()),
            "bearer appeared before server authentication"
        );
        handle.join().expect("impostor exits");
    }

    #[test]
    fn a_dead_leading_address_does_not_hide_a_live_one() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binds");
        let live = listener.local_addr().expect("has an address");
        // Port 1 is closed; it stands in for an unroutable AAAA ahead of the A.
        let dead = SocketAddr::from(([127, 0, 0, 1], 1));

        let stream = connect_any(
            &[dead, live],
            Instant::now() + Duration::from_secs(2),
            &Cancel::never(),
        )
        .expect("the second address accepts");
        assert_eq!(stream.peer_addr().expect("connected"), live);
    }

    #[test]
    fn every_address_failing_reports_the_last_failure() {
        let dead = SocketAddr::from(([127, 0, 0, 1], 1));
        let error = connect_any(
            &[dead, dead],
            Instant::now() + Duration::from_secs(1),
            &Cancel::never(),
        )
        .expect_err("nothing is listening");
        assert!(matches!(error, HttpError::Connect(_)), "{error:?}");
    }

    #[test]
    fn only_a_failure_before_the_first_write_counts_as_never_received() {
        // Exhaustive on purpose: a new variant must be classified deliberately,
        // because answering "never received" wrongly permits a second execution.
        for error in [
            HttpError::Resolve("no such host".to_string()),
            HttpError::Connect("refused".to_string()),
        ] {
            assert!(error.peer_never_received_it(), "{error:?}");
        }
        for error in [
            HttpError::Io("reset".to_string()),
            HttpError::Malformed("truncated".to_string()),
            HttpError::TooLarge(1),
            HttpError::InvalidRequest("bad token".to_string()),
            // Cancelling can happen before the first write, so this one is not
            // classified by what the peer saw: re-sending would spend another
            // node on an answer nobody is waiting for.
            HttpError::Cancelled,
        ] {
            assert!(!error.peer_never_received_it(), "{error:?}");
        }
        for error in [
            HttpError::Policy("cleartext refused".to_string()),
            HttpError::Tls("certificate rejected".to_string()),
        ] {
            assert!(error.peer_never_received_it(), "{error:?}");
        }
    }

    #[test]
    fn a_dial_failure_from_a_real_socket_is_classified_as_never_received() {
        // Not a hand-built variant: this is the error the dialling path actually
        // produces against a closed port, which is the case failover turns on.
        let dead = SocketAddr::from(([127, 0, 0, 1], 1));
        let error = connect_any(
            &[dead],
            Instant::now() + Duration::from_secs(1),
            &Cancel::never(),
        )
        .expect_err("nothing is listening");
        assert!(error.peer_never_received_it(), "{error:?}");
    }

    #[test]
    fn a_content_length_body_is_read_exactly() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello-trailing-garbage";
        let response = parse_response(raw, LIMIT).expect("parses");
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"hello");
    }

    #[test]
    fn a_chunked_body_is_reassembled() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        assert_eq!(
            parse_response(raw, LIMIT).expect("parses").body,
            b"hello world"
        );
    }

    #[test]
    fn chunk_extensions_are_ignored() {
        let raw =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5;name=value\r\nhello\r\n0\r\n\r\n";
        assert_eq!(parse_response(raw, LIMIT).expect("parses").body, b"hello");
    }

    #[test]
    fn a_body_without_framing_headers_is_read_to_end() {
        let raw = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{}";
        assert_eq!(parse_response(raw, LIMIT).expect("parses").body, b"{}");
    }

    #[test]
    fn header_names_are_matched_case_insensitively() {
        let raw = b"HTTP/1.1 200 OK\r\ncOnTeNt-LeNgTh: 2\r\n\r\nok";
        assert_eq!(parse_response(raw, LIMIT).expect("parses").body, b"ok");
    }

    #[test]
    fn a_non_http_greeting_is_refused_rather_than_guessed_at() {
        let raw = b"SSH-2.0-OpenSSH_9.0\r\n\r\n";
        assert!(matches!(
            parse_response(raw, LIMIT),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_response_with_no_header_terminator_is_refused() {
        assert!(matches!(
            parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n", LIMIT),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_truncated_body_is_refused_rather_than_silently_short() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\nshort";
        assert!(matches!(
            parse_response(raw, LIMIT),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_truncated_chunk_is_refused() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n9\r\nshort\r\n";
        assert!(matches!(
            parse_response(raw, LIMIT),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn an_oversized_content_length_is_refused_before_allocating() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 99999999\r\n\r\n";
        assert_eq!(parse_response(raw, LIMIT), Err(HttpError::TooLarge(LIMIT)));
    }

    #[test]
    fn an_oversized_chunked_body_is_refused_mid_stream() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
        assert_eq!(parse_response(raw, 2), Err(HttpError::TooLarge(2)));
    }

    #[test]
    fn a_non_200_is_reported_with_its_code() {
        let raw = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(parse_response(raw, LIMIT).expect("parses").status, 503);
    }

    /// Feed a body one byte at a time — the hostile case for an incremental
    /// decoder, because every frame boundary lands mid-push.
    fn decode_byte_by_byte(raw: &[u8], chunked: bool) -> Result<Vec<u8>, HttpError> {
        let mut decoder = ChunkDecoder::new(chunked);
        let mut out = Vec::new();
        for byte in raw {
            decoder.push(&[*byte], LIMIT, &mut out)?;
        }
        Ok(out)
    }

    /// Frame payloads as a chunked body. Built rather than written out so a
    /// hand-counted size can never make a fixture disagree with itself.
    fn chunked(payloads: &[&str]) -> Vec<u8> {
        let mut raw = Vec::new();
        for payload in payloads {
            raw.extend_from_slice(format!("{:x}\r\n{payload}\r\n", payload.len()).as_bytes());
        }
        raw.extend_from_slice(b"0\r\n\r\n");
        raw
    }

    #[test]
    fn an_incrementally_decoded_body_matches_the_whole_body_decoder() {
        // The same bytes through both decoders: whatever `dechunk` produces for
        // a complete body, the streaming decoder must produce for the pieces.
        let raw = chunked(&[
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: [DONE]\n\n",
        ]);
        let whole = dechunk(&raw, LIMIT).expect("whole-body decode");
        let streamed = decode_byte_by_byte(&raw, true).expect("incremental decode");
        assert_eq!(streamed, whole);
        assert_eq!(
            String::from_utf8(streamed).expect("utf8"),
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n"
        );
    }

    #[test]
    fn a_chunk_frame_split_across_reads_is_reassembled() {
        let raw = b"5\r\nhello\r\n5\r\nworld\r\n0\r\n\r\n";
        // Every possible split point, so no single boundary is special-cased.
        for at in 0..raw.len() {
            let mut decoder = ChunkDecoder::new(true);
            let mut out = Vec::new();
            decoder
                .push(&raw[..at], LIMIT, &mut out)
                .expect("first half");
            decoder
                .push(&raw[at..], LIMIT, &mut out)
                .expect("second half");
            assert_eq!(out, b"helloworld", "split at {at}");
            assert!(decoder.finished(), "split at {at}");
        }
    }

    #[test]
    fn the_terminal_chunk_ends_the_body_and_a_trailer_is_ignored() {
        let mut decoder = ChunkDecoder::new(true);
        let mut out = Vec::new();
        decoder
            .push(b"3\r\nabc\r\n0\r\nX-Trailer: 1\r\n\r\n", LIMIT, &mut out)
            .expect("decodes");
        assert_eq!(out, b"abc");
        assert!(decoder.finished());
    }

    #[test]
    fn an_unchunked_body_passes_straight_through() {
        let mut decoder = ChunkDecoder::new(false);
        let mut out = Vec::new();
        decoder
            .push(b"data: raw\n\n", LIMIT, &mut out)
            .expect("passes");
        assert_eq!(out, b"data: raw\n\n");
        // Without chunk framing there is no terminal chunk; only EOF ends it.
        assert!(!decoder.finished());
    }

    #[test]
    fn a_chunk_larger_than_the_cap_is_refused_before_it_is_allocated() {
        let mut decoder = ChunkDecoder::new(true);
        let mut out = Vec::new();
        // 0x1000000 = 16 MiB claimed against a 1 MiB cap.
        let error = decoder
            .push(b"1000000\r\n", LIMIT, &mut out)
            .expect_err("an oversized chunk is refused");
        assert!(matches!(error, HttpError::TooLarge(LIMIT)), "{error:?}");
        assert!(out.is_empty(), "nothing may be emitted for a refused chunk");
    }

    #[test]
    fn a_size_line_that_never_terminates_is_refused_rather_than_buffered() {
        let mut decoder = ChunkDecoder::new(true);
        let mut out = Vec::new();
        let filler = vec![b'a'; MAX_CHUNK_SIZE_LINE + 1];
        assert!(matches!(
            decoder.push(&filler, LIMIT, &mut out),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_chunk_not_terminated_by_crlf_is_malformed() {
        let mut decoder = ChunkDecoder::new(true);
        let mut out = Vec::new();
        assert!(matches!(
            decoder.push(b"3\r\nabcXX", LIMIT, &mut out),
            Err(HttpError::Malformed(_))
        ));
    }

    #[test]
    fn a_content_type_is_matched_without_its_parameters() {
        let head = parse_head(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n\
             Transfer-Encoding: chunked",
        )
        .expect("parses");
        assert!(head.is_event_stream());
        assert!(head.chunked);

        let json = parse_head("HTTP/1.1 200 OK\r\nContent-Type: application/json").expect("parses");
        assert!(!json.is_event_stream());
    }

    #[test]
    fn a_bearer_token_becomes_an_authorization_header() {
        let head = request_head(
            "POST",
            "/v1/chat/completions",
            "node:8181",
            ACCEPT_JSON,
            Some(b"{}"),
            Some("s3cret"),
        )
        .expect("a well-formed token is sendable");
        assert!(
            head.contains("\r\nAuthorization: Bearer s3cret\r\n"),
            "{head}"
        );
        // The rest of the head must survive the new line unchanged.
        assert!(
            head.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"),
            "{head}"
        );
        assert!(head.contains("\r\nHost: node:8181\r\n"), "{head}");
        assert!(head.contains("\r\nContent-Length: 2\r\n"), "{head}");
        assert!(head.ends_with("\r\n\r\n"), "{head}");
    }

    #[test]
    fn no_token_means_no_authorization_header_at_all() {
        // Not an empty one: an unauthenticated node must see the same request it
        // saw before the fabric learned about bearer tokens.
        let head = request_head("GET", "/v1/health", "node:8181", ACCEPT_JSON, None, None)
            .expect("builds");
        assert!(
            !head.to_ascii_lowercase().contains("authorization"),
            "{head}"
        );
        assert_eq!(
            head,
            "GET /v1/health HTTP/1.1\r\nHost: node:8181\r\nAccept: application/json\r\n\
             Connection: close\r\n\r\n"
        );
    }

    #[test]
    fn a_token_carrying_a_control_character_is_refused_rather_than_written() {
        // Writing this verbatim would let the token inject headers of its own.
        // `s3cret` appears in no refusal wording, so finding it in the message
        // would mean the value leaked rather than merely the word "token".
        for token in ["s3cret\r\nX-Injected: 1", "s3cret\n", "s3cret\0"] {
            let error = request_head(
                "GET",
                "/v1/health",
                "node:8181",
                ACCEPT_JSON,
                None,
                Some(token),
            )
            .expect_err("a control character is not sendable");
            match &error {
                HttpError::InvalidRequest(detail) => {
                    assert!(
                        !detail.contains("s3cret"),
                        "the token must not be echoed: {detail}"
                    )
                }
                other => panic!("expected InvalidRequest, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_empty_token_is_refused_rather_than_sent_as_a_bare_bearer() {
        // `Authorization: Bearer ` is rejected by the server anyway; failing here
        // says why instead of arriving as an unexplained 401.
        assert!(matches!(
            request_head(
                "GET",
                "/v1/health",
                "node:8181",
                ACCEPT_JSON,
                None,
                Some("")
            ),
            Err(HttpError::InvalidRequest(_))
        ));
    }

    #[test]
    fn an_unsendable_token_is_refused_before_a_socket_is_opened() {
        // Port 1 is closed, so a connect error here would prove we dialled first.
        let error = request(
            "127.0.0.1",
            1,
            "GET",
            "/v1/health",
            None,
            Some("bad\r\ntoken"),
            Duration::from_millis(500),
            LIMIT,
            &Cancel::never(),
        )
        .expect_err("refused");
        assert!(
            matches!(error, HttpError::InvalidRequest(_)),
            "must refuse before dialling, got {error:?}"
        );
    }

    #[test]
    fn a_closed_port_reports_connect_failure_not_a_panic() {
        let error = request(
            "127.0.0.1",
            1,
            "GET",
            "/v1/health",
            None,
            None,
            Duration::from_millis(500),
            LIMIT,
            &Cancel::never(),
        )
        .expect_err("port 1 is closed");
        assert!(
            matches!(error, HttpError::Connect(_) | HttpError::Io(_)),
            "unexpected: {error:?}"
        );
    }

    #[test]
    fn an_unresolvable_host_reports_a_resolve_failure() {
        let error = request(
            "camelid-fabric-host-that-does-not-exist.invalid",
            8181,
            "GET",
            "/v1/health",
            None,
            None,
            Duration::from_millis(500),
            LIMIT,
            &Cancel::never(),
        )
        .expect_err("`.invalid` never resolves");
        assert!(
            matches!(error, HttpError::Resolve(_)),
            "unexpected: {error:?}"
        );
    }

    /// Whether a request head and its declared body have both arrived.
    ///
    /// One read is not enough: [`request_head`] writes the head and the body
    /// separately, so they can land in separate segments.
    fn request_complete(raw: &[u8]) -> bool {
        let Some(split) = find_header_end(raw) else {
            return false;
        };
        let head = String::from_utf8_lossy(&raw[..split.headers_end]);
        let declared = head
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-length"))
            .and_then(|(_, value)| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        raw.len() >= split.body_start + declared
    }

    /// Answer one connection with canned bytes and hang up, so a frame that
    /// stops early can be exercised without a stub node.
    fn canned_node(raw: &'static [u8]) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binds");
        let port = listener.local_addr().expect("has an address").port();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            // Every byte of the request has to be consumed before answering:
            // closing a socket that still holds unread bytes is an abortive
            // close on Windows, and it discards the reply along with them.
            let mut request = Vec::new();
            let mut scratch = [0_u8; 1024];
            while !request_complete(&request) {
                match stream.read(&mut scratch) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => request.extend_from_slice(&scratch[..read]),
                }
            }
            let _ = stream.write_all(raw);
        });
        port
    }

    fn stream_from(port: u16) -> ResponseStream {
        open_stream(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            Some(b"{}"),
            None,
            Duration::from_secs(10),
            Duration::from_secs(10),
            LIMIT,
            &Cancel::never(),
        )
        .expect("the canned node answers")
    }

    /// Accept one connection, consume the whole request, then answer `prelude`
    /// and say nothing more — a node that has taken the work and is still doing
    /// it. The connection is held open, so a reader that gives up early can only
    /// have done so because it chose to.
    fn working_node(prelude: &'static [u8]) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binds");
        let port = listener.local_addr().expect("has an address").port();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut request = Vec::new();
            let mut scratch = [0_u8; 1024];
            while !request_complete(&request) {
                match stream.read(&mut scratch) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => request.extend_from_slice(&scratch[..read]),
                }
            }
            if !prelude.is_empty() {
                let _ = stream.write_all(prelude);
            }
            // Far longer than any deadline these tests give a reader, so a
            // reader returning early is never this node giving up.
            std::thread::sleep(Duration::from_secs(30));
        });
        port
    }

    /// Fire `cancel` shortly, the way a client hanging up mid-request does.
    fn cancel_shortly(cancel: &Cancel) {
        let cancel = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            cancel.cancel();
        });
    }

    /// The whole point of the mechanism: a node's answer can legitimately be
    /// minutes away, so waiting out the deadline is not a bounded cost.
    #[test]
    fn a_cancelled_request_stops_reading_rather_than_waiting_out_its_deadline() {
        let port = working_node(b"");
        let cancel = Cancel::new();
        cancel_shortly(&cancel);

        let started = Instant::now();
        let error = request(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            Some(b"{}"),
            None,
            Duration::from_secs(30),
            LIMIT,
            &cancel,
        )
        .expect_err("the node never answers");
        let waited = started.elapsed();

        assert_eq!(error, HttpError::Cancelled);
        assert!(
            waited < Duration::from_secs(2),
            "waited {waited:?} of a 30s budget after the client had gone"
        );
    }

    /// A caller that has already gone must not cost its node even a connection,
    /// let alone the generation slot accepting one would commit.
    #[test]
    fn a_request_already_given_up_on_never_opens_a_socket() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binds");
        let port = listener.local_addr().expect("has an address").port();
        let cancel = Cancel::new();
        cancel.cancel();

        let error = request(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            Some(b"{}"),
            None,
            Duration::from_secs(30),
            LIMIT,
            &cancel,
        )
        .expect_err("the caller had already gone");

        assert_eq!(error, HttpError::Cancelled);
        listener
            .set_nonblocking(true)
            .expect("the listener can be polled");
        assert!(
            listener.accept().is_err(),
            "a connection was opened on behalf of a caller that had gone"
        );
    }

    /// The head of a streaming answer is a whole prefill away, which is the
    /// longest a client is ever left with nothing at all to read.
    #[test]
    fn a_head_that_has_not_arrived_is_given_up_on_with_its_client() {
        let port = working_node(b"");
        let cancel = Cancel::new();
        cancel_shortly(&cancel);

        let started = Instant::now();
        let opened = open_stream(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            Some(b"{}"),
            None,
            Duration::from_secs(30),
            Duration::from_secs(30),
            LIMIT,
            &cancel,
        );
        let waited = started.elapsed();

        // `ResponseStream` owns a socket rather than describing one, so it has
        // no `Debug` for `expect_err` to print.
        let Err(error) = opened else {
            panic!("the node never sent a head, so no stream can have opened");
        };
        assert_eq!(error, HttpError::Cancelled);
        assert!(
            waited < Duration::from_secs(2),
            "waited {waited:?} of a 30s budget after the client had gone"
        );
    }

    /// Relaying already stops when a send to the client fails, but only once
    /// the node produces something to send. Between two tokens there is nothing
    /// to fail on, and that gap is the whole idle timeout wide.
    #[test]
    fn a_stream_whose_client_has_gone_stops_between_events() {
        // `data: one\n\n` is 11 bytes = 0xb; nothing follows it on the wire.
        let port = working_node(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
              Transfer-Encoding: chunked\r\n\r\nb\r\ndata: one\n\n\r\n",
        );
        let cancel = Cancel::new();
        let mut stream = open_stream(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            Some(b"{}"),
            None,
            Duration::from_secs(30),
            Duration::from_secs(30),
            LIMIT,
            &cancel,
        )
        .expect("the head arrives");
        assert_eq!(
            stream.next_chunk().expect("the first event arrives"),
            Some(b"data: one\n\n".to_vec())
        );

        cancel.cancel();
        let started = Instant::now();
        let error = stream.next_chunk().expect_err("the client has gone");
        let waited = started.elapsed();

        assert_eq!(error, HttpError::Cancelled);
        assert!(
            waited < Duration::from_secs(2),
            "waited {waited:?} of a 30s idle budget after the client had gone"
        );
    }

    /// `parse_response` calls a chunked body that stops early malformed. The
    /// streaming reader must not disagree with it about the same bytes: ending
    /// the relay cleanly would frame a half-generation as a whole one.
    #[test]
    fn a_stream_cut_off_before_its_terminal_chunk_is_reported_not_ended_cleanly() {
        // `data: one\n\n` is 11 bytes = 0xb. No terminal chunk follows it.
        let port = canned_node(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
              Transfer-Encoding: chunked\r\n\r\nb\r\ndata: one\n\n\r\n",
        );
        let mut stream = stream_from(port);

        assert_eq!(
            stream.next_chunk().expect("the first event arrives"),
            Some(b"data: one\n\n".to_vec())
        );
        let error = stream
            .next_chunk()
            .expect_err("the node died before the body was complete");
        assert!(matches!(error, HttpError::Malformed(_)), "{error:?}");
    }

    /// The control for the test above: a stream that does reach its terminal
    /// chunk still ends without an error, so the refusal is not blanket.
    #[test]
    fn a_stream_that_reaches_its_terminal_chunk_ends_cleanly() {
        let port = canned_node(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
              Transfer-Encoding: chunked\r\n\r\nb\r\ndata: one\n\n\r\n0\r\n\r\n",
        );
        let mut stream = stream_from(port);

        assert_eq!(
            stream.next_chunk().expect("the first event arrives"),
            Some(b"data: one\n\n".to_vec())
        );
        assert_eq!(stream.next_chunk().expect("the body is complete"), None);
    }

    /// With neither chunked framing nor a declared length, the close *is* the
    /// framing, so EOF must stay a clean end.
    #[test]
    fn a_close_delimited_stream_still_ends_at_eof() {
        let port = canned_node(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
              Connection: close\r\n\r\ndata: one\n\n",
        );
        let mut stream = stream_from(port);

        assert_eq!(
            stream.next_chunk().expect("the event arrives"),
            Some(b"data: one\n\n".to_vec())
        );
        assert_eq!(stream.next_chunk().expect("the close ends it"), None);
    }

    #[test]
    fn a_buffered_body_shorter_than_its_declared_length_is_refused() {
        let port = canned_node(
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\
              Content-Length: 99\r\n\r\n{\"error\":",
        );
        let error = stream_from(port)
            .into_buffered(LIMIT)
            .expect_err("the declared body never arrived");
        assert!(matches!(error, HttpError::Malformed(_)), "{error:?}");
    }

    #[test]
    fn a_buffered_body_that_matches_its_declared_length_is_returned() {
        let port = canned_node(
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\
              Content-Length: 2\r\n\r\n{}",
        );
        let response = stream_from(port).into_buffered(LIMIT).expect("complete");
        assert_eq!(response.status, 503);
        assert_eq!(response.body, b"{}");
    }
}
