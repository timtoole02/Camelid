//! A resident HTTP front door for [`super::Fabric::dispatch`].
//!
//! `fabric run` sends exactly one request per process invocation. This keeps a
//! fabric observed and routable for as long as a real client needs it, without
//! reimplementing placement or forwarding: every request still goes through
//! [`super::Fabric::dispatch`], so the routing and forwarding guarantees are
//! identical to the CLI path.
//!
//! Streaming is relayed rather than interpreted: when a client asks for
//! `stream: true` the node's server-sent events are forwarded byte for byte, so
//! an event field this proxy has never heard of reaches the client unaltered.
//!
//! # What it serves
//!
//! The engine's stateless inference routes ([`PLACED_ROUTES`]) plus discovery
//! over the whole fabric. A route earns its place there by being answerable by
//! any node serving the model the request names — which the Responses and
//! Conversations APIs are not, so they are refused with a reason rather than
//! placed. Anything else is a 404 that names what is served.
//!
//! Unlike the CLI, this process serves many requests, so it reuses a fabric
//! observation for a bounded time instead of probing every node on every one.
//! The bound is set by whoever builds the [`Fabric`]; see
//! [`Fabric::with_max_observation_age`].
//!
//! # Two credentials, in opposite directions
//!
//! [`ClientAuth`] is what a client must present to *this* proxy. The bearer the
//! [`Fabric`] was built with is what this proxy presents to *its nodes*. They
//! are configured separately and are not interchangeable: a fabric whose nodes
//! share one key must not thereby accept that key from the network.
//!
//! The check itself is [`crate::api::authenticate`] — the engine's, not a
//! second one. A client sees the same 401 whichever of the two it is talking
//! to, and there is only one constant-time comparison in the crate to get right.
//!
//! # Limits an operator has to know about
//!
//! * Without a [`ClientAuth`], anything that can reach this address can drive
//!   every node in the fabric, so [`bind`] refuses a non-loopback address
//!   unless a key is configured or the risk is acknowledged explicitly.
//! * `--api-key` and `--api-key-file` configure one fixed key that cannot be
//!   rotated without restarting. `--client-keys` instead names clients and
//!   re-reads its file when it changes, so one client's key can be revoked or
//!   rotated without disturbing the others — but a re-read that fails keeps
//!   the previous set in force, so a revocation written into a file that no
//!   longer parses has not taken effect until it does.
//! * A [`ProxyTls`] pair encrypts what clients send. It is a separate refusal
//!   from the one above and neither stands in for the other: a key sent over
//!   cleartext is a key given away, and TLS says nothing about who may drive
//!   the fabric. What crosses the wire *to the nodes* is a different question
//!   again — that hop is whatever the node's own listener speaks.
//! * Every request is recorded through `tracing` at INFO, and a failure at
//!   WARN, which means nothing is written unless `RUST_LOG` asks for it — the
//!   same as the rest of this binary. `RUST_LOG=camelid=info` is the whole
//!   access log; `RUST_LOG=camelid=warn` narrows it to what failed. Each line
//!   carries the client it came from and an `x-request-id`, which is also
//!   answered to the client — the node does not log that id, so it correlates
//!   a complaint with this proxy's line, not with the node's own.
//! * A stop (Ctrl-C, or SIGTERM where there is one) closes the listener and
//!   lets the requests already in flight finish, bounded by `forward_timeout`.
//!   A kill still drops them: this covers being asked to stop, not being shot.
//! * `/v1/health` answers readiness, not liveness — a 503 from it means "no
//!   node is ready", never "restart me". It is also the one route a client
//!   reaches without a key, because the engine's shared `authenticate` keeps
//!   `/v1/health` public; it therefore names the fabric's nodes and models only
//!   when this proxy is bound to loopback.
//! * A node that becomes ready, or loads a different model, is routed to only
//!   once the current observation expires. A node that *stops* answering is not
//!   waited on: the request that finds it gone is placed on another node
//!   serving the same model, and the observation that named it is dropped.
//!   That second placement only happens when the first node was never reached;
//!   a node that took the request and then failed ends it, because this proxy
//!   cannot know whether it started generating. See
//!   [`super::DEFAULT_MAX_FORWARD_ATTEMPTS`].
//! * **Mixed engines.** By default this proxy places on Camelid nodes only.
//!   `--allow-mixed-engines`, fixed for the life of the process, also places
//!   on engines it otherwise only reads. Every answer then also names the
//!   model id sent and, where the node's own listing said, whether it was
//!   resident; a request carrying tools, or for a rerank route, still never
//!   lands on a node that cannot be shown to handle it; and a refusal from
//!   such a node is relayed once, never re-placed, because nothing tells it
//!   apart from a real failure. `/v1/health` reports the mode and, on
//!   loopback, what it accepts about each node.
//! * A client that hangs up takes its request's work with it. A dispatch runs
//!   on a blocking thread and cannot be aborted, so it is told instead: the
//!   frame that owns the request holds a [`Cancel`], dropping that frame is
//!   what a client leaving looks like to axum, and the dispatch stops at its
//!   next socket-operation boundary and hangs up on the node. A node that sees
//!   its caller go stops generating within one decode step, so this gives back
//!   the generation slot rather than merely a thread here. Mid-relay a stream
//!   already noticed on its own — its next send to the client fails — so what
//!   this adds there is the idle gap between two events, and everywhere else
//!   it is the only mechanism. What it does not cover is a probe round already
//!   in flight, which is bounded by `--timeout-ms` and costs a node a health
//!   read rather than its generation slot.

use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::connect_info::IntoMakeServiceWithConnectInfo;
use axum::extract::rejection::JsonRejection;
use axum::extract::{ConnectInfo, DefaultBodyLimit, Path, Request, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{middleware, Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use axum_server::Handle;
use serde_json::Value;
use tower_http::cors::CorsLayer;

use super::client_keys::Admission;
use super::policy::{
    MixedEngines, Residency, RouteDecision, RouteError, RouteMode, RouteRequest, COLD_LOAD_COST,
    MIXED_ENGINES_FLAG, UNREPORTED_LOAD_COST,
};
use crate::api::{ApiAuth, DEFAULT_MAX_REQUEST_BODY_BYTES};
use crate::fabric::{
    self as fabric, Cancel, DispatchError, Fabric, ForwardError, ModelIdentity, NodeEngine,
    StreamOutcome,
};
use crate::tls_pair::resolve_tls;

/// Optional header a client sends to request affinity to a specific node.
const STICKY_HEADER: &str = "x-camelid-fabric-sticky";

/// The header a request id arrives on, and is answered with.
const REQUEST_ID_HEADER: &str = "x-request-id";

/// Longest inbound request id this proxy will adopt as its own.
const MAX_REQUEST_ID_LEN: usize = 128;

/// What a request that lost its client is answered with.
///
/// Not an IANA code: 499 is nginx's, and it is here for the reason nginx has it
/// — every standard code would claim either that the request succeeded or that
/// something upstream failed, and neither happened. Nothing on the network
/// reads it, because the client it describes has gone, and nor does the access
/// log, whose own future is dropped with the handler's. What it protects is a
/// test, and any future caller of [`forward_error`] that is not the network,
/// from being told a healthy node failed.
fn client_closed_request() -> StatusCode {
    StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_GATEWAY)
}

/// The engine routes this proxy places on a node.
///
/// Every one of them is a pure function of its own request body and the model
/// that body names: nothing in the answer depends on which node produced it, so
/// any node serving that model may. That is what makes placing them legitimate,
/// and it is the property a route has to have to be added here.
///
/// This list is the single source of truth. The router is built from it, and so
/// is the refusal a client gets for a route that is not on it, so the two cannot
/// drift apart.
pub const PLACED_ROUTES: [&str; 5] = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/rerank",
    "/v1/reranking",
];

/// Engine routes that answer from state held by one node.
///
/// The Responses and Conversations APIs keep their items in a SQLite store on
/// the node that served the request. Placing them would appear to work and then
/// lose a conversation the moment a follow-up landed on a different node, so
/// they are refused here rather than half-supported. A client that needs them
/// can talk to a node directly, where they work exactly as documented.
const NODE_LOCAL_ROUTES: [&str; 6] = [
    "/v1/responses",
    "/v1/responses/:id",
    "/v1/conversations",
    "/v1/conversations/:id",
    "/v1/conversations/:id/items",
    "/v1/conversations/:id/items/:item_id",
];

/// The credential a client must present to this proxy.
///
/// Re-exported from [`super::client_keys`], which owns what a key set is and
/// when it is re-read; this module only decides what a refusal looks like on
/// the wire.
pub use super::client_keys::ClientAuth;

/// The certificate this proxy presents to its clients.
///
/// Holds the loaded certificate rather than the paths it came from. Reading it
/// is what tells you whether it can be served at all, and that has to happen
/// before an address is bound, announced or a node is probed — otherwise the
/// operator is told the proxy is listening on `https://` moments before it
/// exits. So `Some` here means a certificate that will actually be presented,
/// which is also what lets [`bind`] treat it as an answer to the cleartext
/// guard.
#[derive(Clone)]
pub struct ProxyTls {
    config: RustlsConfig,
}

impl fmt::Debug for ProxyTls {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProxyTls").finish_non_exhaustive()
    }
}

impl ProxyTls {
    /// Resolve a certificate/key pair and load it, refusing half a pair.
    ///
    /// The half-a-pair rule is `crate::tls_pair`'s, shared with the engine's
    /// own listener so the two front doors cannot answer it differently.
    pub async fn resolve(
        cert: Option<PathBuf>,
        key: Option<PathBuf>,
    ) -> std::io::Result<Option<Self>> {
        let Some(files) = resolve_tls(cert, key)? else {
            return Ok(None);
        };
        let config = RustlsConfig::from_pem_file(&files.cert, &files.key)
            .await
            .map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("could not load TLS certificate/key: {error}"),
                )
            })?;
        Ok(Some(Self { config }))
    }
}

/// Which browser origins may read this proxy.
///
/// The web UI is served from the engine's address or from a dev server, never
/// from the proxy's own, so without an allowance a browser withholds every
/// answer this proxy gives it. Validated and built by [`crate::cors_origins`],
/// the engine's own rules, so the two front doors refuse a wildcard, `null`, a
/// path or a query in the same words and allow the same request headers.
#[derive(Debug, Clone)]
pub struct ProxyCors {
    layer: CorsLayer,
    origins: usize,
}

impl ProxyCors {
    /// `None` when no origin is named: no allowance at all, rather than a
    /// layer that allows nothing, because even that answers every `OPTIONS`
    /// itself and would change what an unconfigured proxy says.
    pub fn resolve(origins: &[String]) -> std::io::Result<Option<Self>> {
        if origins.is_empty() {
            return Ok(None);
        }
        let allowed = origins
            .iter()
            .map(|origin| crate::cors_origins::parse_origin(origin))
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(Some(Self {
            layer: crate::cors_origins::layer(&allowed),
            origins: allowed.len(),
        }))
    }

    /// How many origins are allowed, for the startup line.
    pub fn origin_count(&self) -> usize {
        self.origins
    }
}

/// What the operator has explicitly accepted about exposing this proxy.
///
/// Two separate acknowledgements because they are two separate risks: one is
/// "anyone who can reach this drives my fabric", the other is "the credential
/// and every prompt cross the network in the clear". A single flag for both
/// would let acknowledging one silently accept the other.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoteAcknowledgements {
    /// Serve a routable address with no client credential.
    pub unauthenticated: bool,
    /// Serve a routable address without TLS.
    pub cleartext: bool,
}

/// Everything a request needs beyond which nodes exist.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Default placement mode; a client can still request affinity to a
    /// specific node via [`STICKY_HEADER`] regardless of this default.
    pub mode: RouteMode,
    /// Budget for the generation itself, which can legitimately take minutes.
    ///
    /// On a streaming request it bounds two waits that mean the same thing —
    /// the wait for the node's response head, and any later gap in which the
    /// node sends nothing. A healthy stream resets the second with every token,
    /// so a long generation is never cut short by it.
    pub forward_timeout: Duration,
    /// What a client must present to be served at all.
    pub auth: ClientAuth,
    /// The certificate this proxy presents, or `None` to serve cleartext.
    ///
    /// Set on the config rather than passed to [`serve_on_until`] so that the
    /// same value decides both what is served and whether [`bind`] considers
    /// a routable address safe to open.
    pub tls: Option<ProxyTls>,
    /// Browser origins allowed to read this proxy. `None` — the default — sends
    /// no CORS header at all.
    pub cors: Option<ProxyCors>,
    /// Whether this proxy also places on engines it otherwise only reads.
    /// Fixed for the life of the process: a request is placed under the mode
    /// the proxy started with, and there is no route that changes it.
    pub mixed: MixedEngines,
    /// The address this proxy listens on.
    ///
    /// Read only to decide whether `/v1/health` may name the fabric's members:
    /// a loopback listener cannot be reached from off the machine.
    /// [`serve_on_until`] overwrites this with the address the listener
    /// actually bound, so the value the disclosure rule sees can never be a
    /// caller's guess. It is set there rather than in [`serve_on`] because
    /// every serving path goes through it, including the injected-stop seam.
    pub bound: SocketAddr,
}

#[derive(Clone)]
struct ServerState {
    /// Shared rather than cloned: axum clones the state for every request, and
    /// a `Fabric` owns a `Vec<NodeSpec>`.
    fabric: Arc<Fabric>,
    config: ServeConfig,
}

/// Build the router without binding a socket, so tests can drive it directly.
pub fn router(fabric: Fabric, config: ServeConfig) -> Router {
    let auth = config.auth.clone();
    let cors = config.cors.clone();
    // Under mixed placement a foreign node added to the node file is placed on
    // at once, so from here on every such addition is announced.
    if config.mixed == MixedEngines::Allowed {
        fabric.announce_foreign_additions(true);
    }

    let placed = PLACED_ROUTES.iter().fold(Router::new(), |router, path| {
        let path = *path;
        router.route(
            path,
            post(
                move |state: State<ServerState>,
                      headers: HeaderMap,
                      payload: std::result::Result<Json<Value>, JsonRejection>| async move {
                    place_and_forward(state, path, headers, payload).await
                },
            ),
        )
    });
    let refusals = NODE_LOCAL_ROUTES
        .iter()
        .fold(placed, |router, path| router.route(path, any(node_local)));

    let guarded = refusals
        .route("/v1/models", get(models))
        .route("/v1/models/:model", get(model))
        .route("/v1/health", get(health))
        .route("/v1/fabric/compare", post(compare))
        // A client that reaches an address it was told is OpenAI-compatible has
        // to be able to tell "wrong route" from "wrong model", and axum's own
        // 404 carries no body at all.
        .fallback(unknown_route)
        // Without this, axum's 2 MiB default would refuse conversations the
        // node behind the proxy accepts.
        .layer(DefaultBodyLimit::max(DEFAULT_MAX_REQUEST_BODY_BYTES))
        .with_state(ServerState {
            fabric: Arc::new(fabric),
            config,
        })
        // Wraps every route: an unauthenticated request is refused before
        // anything below reads its body or observes the fabric.
        .layer(middleware::from_fn_with_state(auth, client_auth));
    // Outside the client key, so that key's refusals carry the allow-origin
    // header too: without it a page sees an opaque network failure where the
    // proxy said 401, and cannot tell its user a key is missing. The key
    // already lets a preflight through, since a browser sends no credential
    // on one; this is what answers it, instead of a route that only takes
    // POST. Absent entirely unless an origin was named, because even a layer
    // allowing nothing answers every `OPTIONS` itself, and a proxy nobody
    // configured for a browser must answer exactly as it always has.
    let reachable = match cors {
        Some(cors) => guarded.layer(cors.layer),
        None => guarded,
    };
    // Outermost, so a request refused above is still recorded.
    reachable.layer(middleware::from_fn(access_log))
}

/// Refuse a request that presents no key this proxy knows, and name the client
/// that presented one it does.
///
/// The proxy has its own layer rather than the engine's `authenticate` because
/// it now holds a *set* of keys, and a single-key middleware cannot say which
/// of them answered. Everything security-bearing is still the engine's: the
/// header parsing and constant-time comparison come from [`ApiAuth`], the
/// exemption for the health routes from [`ApiAuth::route_requires_auth`], and
/// the refusal itself from [`ApiAuth::unauthorized`], so a refusal here is
/// word-for-word the one the engine gives.
///
/// The admitted name is attached to the *response* rather than the request,
/// because the only thing that reads it — [`access_log`] — wraps this layer and
/// has already given the request away by the time an answer exists.
async fn client_auth(State(auth): State<ClientAuth>, request: Request, next: Next) -> Response {
    let exempt = request.method() == axum::http::Method::OPTIONS
        || !ApiAuth::route_requires_auth(request.uri().path());

    let admitted = if exempt {
        None
    } else {
        match auth.admit(request.headers()) {
            Admission::Refused => return ApiAuth::unauthorized(),
            Admission::Open => None,
            Admission::Client(name) => Some(name),
        }
    };

    let mut response = next.run(request).await;
    if let Some(name) = admitted {
        response.extensions_mut().insert(AdmittedClient(name));
    }
    response
}

/// The name of the client a request was admitted under, carried from the
/// authentication layer out to the access log.
#[derive(Clone)]
struct AdmittedClient(Arc<str>);

/// Record one line per request, whatever the outcome.
///
/// The proxy announced its address and then said nothing ever again, so an
/// operator could not answer "is anything reaching this", "which machine served
/// that", or "what is failing" without a packet capture.
///
/// `TraceLayer`, which the engine's router uses, is not enough on its own here:
/// it can only see HTTP, so it cannot say which node answered — the one fact
/// this process exists to decide — and it records at DEBUG, below the level an
/// operator runs.
///
/// The placement facts are read back off the response rather than threaded out
/// of dispatch, because [`tag`] already puts them there for the client, and one
/// fact with two sources drifts.
///
/// Applied outside the authentication layer deliberately: a refused request is
/// exactly the one worth having a record of, and inside the layer it would be
/// invisible.
///
/// Never recorded: request and response bodies, and every request header — one
/// of them is the client's key. The query string is dropped for the same
/// reason, since the path alone identifies the route.
///
/// On a streaming answer this measures time to the response head, not to the
/// final event: the middleware returns once the head is ready while the body is
/// still being relayed.
async fn access_log(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let client = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(peer)| peer.to_string())
        .unwrap_or_else(|| "-".to_string());
    let request_id = correlation_id(request.headers());
    let started = Instant::now();

    let mut response = next.run(request).await;

    let elapsed_ms = started.elapsed().as_millis() as u64;
    let status = response.status();
    let node = tagged(&response, "x-camelid-fabric-node");
    let reason = tagged(&response, "x-camelid-fabric-reason");
    // The name, never the key. "-" covers both an unauthenticated proxy and a
    // request that was refused, which the status already tells them apart.
    let client_name = response
        .extensions()
        .get::<AdmittedClient>()
        .map_or("-", |AdmittedClient(name)| name.as_ref())
        .to_string();

    if status.is_server_error() {
        tracing::warn!(
            %method,
            %path,
            status = status.as_u16(),
            node,
            reason,
            elapsed_ms,
            client,
            client_name,
            request_id,
            "fabric request failed"
        );
    } else {
        tracing::info!(
            %method,
            %path,
            status = status.as_u16(),
            node,
            reason,
            elapsed_ms,
            client,
            client_name,
            request_id,
            "fabric request"
        );
    }

    // Returned so a client that reports a slow or failed call can name the line
    // that recorded it, without an operator having to guess from a timestamp.
    insert(response.headers_mut(), REQUEST_ID_HEADER, &request_id);
    response
}

/// The id this request is known by, in the log and in the client's answer.
///
/// An inbound id is honoured so one assigned upstream survives the hop — but
/// only after it is checked, because it is written into a log line a client
/// does not otherwise control. HTTP framing rejects a raw CR or LF before this
/// sees it, so the reachable problems are the quieter ones: a tab or DEL that
/// makes a line hard to parse, and a length nothing bounds.
///
/// Anything that fails the check gets a fresh id rather than a cleaned-up one.
/// A sanitised id is no longer the caller's id, so it would correlate with
/// nothing while still looking as though it did.
fn correlation_id(headers: &HeaderMap) -> String {
    headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .filter(|value| is_usable_request_id(value))
        .map(str::to_string)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

/// Pure: non-empty, printable, and short enough to belong in a log line.
///
/// `is_ascii_graphic` excludes space and every control character, so a value
/// that passes cannot break the line it is written on.
fn is_usable_request_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_REQUEST_ID_LEN
        && value.bytes().all(|byte| byte.is_ascii_graphic())
}

/// One of this proxy's own response headers, or `-` when the answer never got
/// far enough to have one.
fn tagged<'a>(response: &'a Response, name: &str) -> &'a str {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("-")
}

/// Bind `addr` for the proxy, refusing an exposure the operator did not ask for.
///
/// Binding goes through here rather than through [`tokio::net::TcpListener`]
/// directly so the guard cannot be skipped by forgetting to call it.
pub async fn bind(
    addr: SocketAddr,
    auth: &ClientAuth,
    tls: Option<&ProxyTls>,
    acknowledged: RemoteAcknowledgements,
) -> std::io::Result<tokio::net::TcpListener> {
    refuse_unauthenticated_remote(addr, auth.is_enabled(), acknowledged.unauthenticated)?;
    refuse_cleartext_remote(addr, tls.is_some(), acknowledged.cleartext)?;
    tokio::net::TcpListener::bind(addr).await
}

/// Refuse an unauthenticated listener that the network can reach. Pure.
///
/// An unauthenticated routable bind exposes every node in the fabric, not just
/// this process. `crate::api::server` refuses the same shape of listener on the
/// same three conditions; this mirrors that refusal and its escape hatch.
fn refuse_unauthenticated_remote(
    addr: SocketAddr,
    authenticated: bool,
    allow_unauthenticated_remote: bool,
) -> std::io::Result<()> {
    if addr.ip().is_loopback() || authenticated || allow_unauthenticated_remote {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "refusing unauthenticated non-loopback listener {addr}; this would expose every \
             configured node to the network. Bind a loopback address, configure \
             --api-key/--api-key-file/--client-keys, or explicitly acknowledge the risk with \
             --allow-unauthenticated-remote"
        ),
    ))
}

/// Refuse a cleartext listener that the network can reach. Pure.
///
/// The key this proxy asks its clients for is sent on every request. Without
/// TLS it crosses the network in the clear, so the flag that is supposed to
/// protect a routable bind is the very thing the bind gives away — along with
/// every prompt and completion. Refusing here rather than warning keeps the
/// insecure case deliberate, which is the same call `--api-key` already makes
/// for the unauthenticated one.
fn refuse_cleartext_remote(
    addr: SocketAddr,
    tls_enabled: bool,
    allow_cleartext_remote: bool,
) -> std::io::Result<()> {
    if addr.ip().is_loopback() || tls_enabled || allow_cleartext_remote {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "refusing cleartext non-loopback listener {addr}; the client key and every \
             prompt would cross the network unencrypted. Bind a loopback address, configure \
             --tls-cert/--tls-key, or explicitly acknowledge the risk with \
             --allow-cleartext-remote"
        ),
    ))
}

/// Serve on an already-bound listener, until the operator asks it to stop.
///
/// Split from [`bind`] so a caller can read back the real port — which matters
/// when it asked for port 0 — before it starts serving.
///
/// A stop is not a kill. The listener closes so no new request is accepted,
/// and the ones already in flight are given until `forward_timeout` to finish,
/// because that is already this proxy's answer to "how long may a request
/// legitimately take".
///
/// That bound is a backstop, and it is worth knowing which requests it is for.
/// A buffered request cannot outlive it — the same value bounds the forward, so
/// the request ends first either way. A *stream* can: its idle timeout resets
/// with every event, so a client reading slowly could otherwise hold the
/// process open for as long as it kept reading. An orchestrator's own grace
/// period is usually shorter than either and simply wins.
pub async fn serve_on(
    listener: tokio::net::TcpListener,
    fabric: Fabric,
    config: ServeConfig,
) -> std::io::Result<()> {
    serve_on_until(listener, fabric, config, stop_requested()).await
}

/// [`serve_on`], with the stop supplied rather than taken from the OS.
///
/// The signal itself cannot be exercised by a test — a test that raised
/// SIGTERM would stop the test runner — so what a stop *does* is separated
/// here from what asks for one. Everything below this line is covered; only
/// [`stop_requested`] is not, and it is the part with no logic in it.
pub async fn serve_on_until(
    listener: tokio::net::TcpListener,
    fabric: Fabric,
    mut config: ServeConfig,
    stop: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    // The listener is the only authority on what was actually bound, and the
    // health disclosure rule turns on it. It is set here rather than in
    // `serve_on` so that a caller entering through this seam cannot skip it.
    config.bound = listener.local_addr()?;

    let drain = config.forward_timeout;
    let tls = config.tls.clone();
    // The peer address is only available to a service built this way, and the
    // access log is the only thing that reads it. Both serving paths below take
    // the same service, so a TLS listener records the client too.
    let service = router(fabric, config).into_make_service_with_connect_info::<SocketAddr>();

    let stopping = async move {
        stop.await;
        tracing::info!("stopping: no longer accepting requests, finishing the ones in flight");
    };

    match tls {
        Some(tls) => serve_tls(listener, service, tls, drain, stopping).await,
        None => serve_cleartext(listener, service, drain, stopping).await,
    }
}

/// Serve cleartext until `stopping` resolves, then drain within `drain`.
async fn serve_cleartext(
    listener: tokio::net::TcpListener,
    service: IntoMakeServiceWithConnectInfo<Router, SocketAddr>,
    drain: Duration,
    stopping: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let (draining, drain_started) = tokio::sync::oneshot::channel();
    let serving = axum::serve(listener, service).with_graceful_shutdown(async move {
        stopping.await;
        let _ = draining.send(());
    });

    tokio::select! {
        served = serving => served,
        // The clock starts when the stop is asked for, not when serving began.
        () = async move {
            let _ = drain_started.await;
            tokio::time::sleep(drain).await;
        } => {
            warn_drain_expired(drain);
            Ok(())
        }
    }
}

/// Serve TLS until `stopping` resolves, then drain within `drain`.
///
/// `axum::serve` cannot present a certificate, so the TLS path is `axum_server`
/// — the same crate, and the same `RustlsConfig::from_pem_file`, the engine's
/// own listener uses. That crate owns the drain itself via [`Handle`], so the
/// bound is expressed by handing it `drain` rather than by racing a sleep.
///
/// The certificate arrived loaded, so there is nothing here that can fail on
/// account of it and quietly leave the operator on the cleartext they were
/// avoiding.
async fn serve_tls(
    listener: tokio::net::TcpListener,
    service: IntoMakeServiceWithConnectInfo<Router, SocketAddr>,
    tls: ProxyTls,
    drain: Duration,
    stopping: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    // `axum_server` drives the accept loop itself, so it needs the listener
    // back as a std one. Non-blocking is set explicitly rather than inherited,
    // matching what the engine does before handing its own listener over.
    let listener = listener.into_std()?;
    listener.set_nonblocking(true)?;

    let handle = Handle::new();
    let stopper = handle.clone();
    let started = std::sync::Arc::new(std::sync::Mutex::new(None::<Instant>));
    let stamped = std::sync::Arc::clone(&started);
    tokio::spawn(async move {
        stopping.await;
        *stamped.lock().expect("drain clock") = Some(Instant::now());
        stopper.graceful_shutdown(Some(drain));
    });

    let served = axum_server::from_tcp_rustls(listener, tls.config)?
        .handle(handle)
        .serve(service)
        .await;

    // A drain that ran the full budget is one that was cut short, which is the
    // same thing the cleartext path reports when its own sleep wins the race.
    if let Some(asked) = *started.lock().expect("drain clock") {
        if asked.elapsed() >= drain {
            warn_drain_expired(drain);
        }
    }
    served
}

/// One wording for "the drain bound ran out", shared by both serving paths.
fn warn_drain_expired(drain: Duration) {
    tracing::warn!(
        drain_seconds = drain.as_secs(),
        "in-flight requests did not finish in time; stopping without them"
    );
}

/// Resolves when the operator asks this process to stop.
///
/// Ctrl-C is the interactive stop. SIGTERM is what an orchestrator sends first,
/// and it is the one that matters for a proxy holding other people's in-flight
/// requests: without a handler it is an immediate kill, and every request being
/// served at that moment dies with it. Windows has no SIGTERM, and there
/// `ctrl_c` also covers the console being closed.
async fn stop_requested() {
    let interrupt = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            // Nothing to wait for, but Ctrl-C must still work, so this arm just
            // never completes rather than resolving and stopping the server.
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = interrupt => {}
        () = terminate => {}
    }
}

/// Report whether this proxy can route a request right now.
///
/// A load balancer needs one address to ask "should I send you traffic", and
/// the proxy had none: `/v1/health` matched no route, so a probe got a bodyless
/// 404 whatever the fabric was actually doing.
///
/// **This is readiness, not liveness.** It answers 503 while no node is ready,
/// which is the right answer to "send me traffic" and the wrong one to "should
/// I restart you" — restarting a proxy cannot bring a node back. Wire it to a
/// readiness probe. Liveness is the `ok` in the body, which is true whenever
/// this code runs at all.
///
/// The answer comes from the same observation placement uses, so a health check
/// taken inside the freshness window costs the fabric no traffic at all.
/// Outside it this probes every node, exactly as any other request would: at the
/// 500 ms default a once-a-second check is always past the window and re-probes
/// every time. A deployment that polls this route therefore wants
/// `--observation-max-age-ms` at or above its polling interval, or the health
/// check itself becomes the node traffic that bound exists to remove.
///
/// That bound is also the only thing pacing this route, because it is
/// deliberately outside [`ClientAuth`]: the caller decides how often it is
/// asked, and with `--observation-max-age-ms 0` every ask probes every node.
async fn health(State(state): State<ServerState>) -> Response {
    let fabric = Arc::clone(&state.fabric);
    // Observing is blocking socket I/O against every node, same as a dispatch.
    let snapshots = match tokio::task::spawn_blocking(move || fabric.observe()).await {
        Ok(snapshots) => snapshots,
        Err(join_error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("could not observe the fabric: {join_error}"),
            )
        }
    };

    let view = PlacementView::of(&state.fabric, &snapshots, state.config.mixed);
    let (status, body) = health_report(&snapshots, disclose(&state), &view);
    (status, Json(body)).into_response()
}

/// Whether this proxy may name its nodes and their engines' versions. Only a
/// loopback listener may: this is decided by the address it is bound to, the
/// same fact for health and for refusals.
fn disclose(state: &ServerState) -> bool {
    state.config.bound.ip().is_loopback()
}

/// How this proxy places, gathered from the fabric so the report that renders
/// it stays pure.
struct PlacementView {
    mixed: MixedEngines,
    served: Vec<String>,
    served_if_mixed: Vec<String>,
    max_forward_attempts: usize,
    covers_nodes_added_later: bool,
    foreign_added: Vec<fabric::ForeignAddition>,
}

impl PlacementView {
    fn of(fabric: &Fabric, snapshots: &[fabric::NodeSnapshot], mixed: MixedEngines) -> Self {
        let ids = |mode: MixedEngines| -> Vec<String> {
            fabric
                .servable_models(snapshots, mode)
                .into_iter()
                .map(|model| model.id)
                .collect()
        };
        let served = ids(mixed);
        let served_if_mixed = ids(MixedEngines::Allowed)
            .into_iter()
            .filter(|id| !served.contains(id))
            .collect();
        Self {
            mixed,
            served,
            served_if_mixed,
            max_forward_attempts: fabric.max_forward_attempts(),
            covers_nodes_added_later: fabric.is_reloadable(),
            foreign_added: fabric.foreign_added_since_start(),
        }
    }
}

/// The sentence the Routing screen shows for what the flag means over a node
/// file that is re-read. Written here, where the behaviour is, so a page never
/// describes a proxy it is not talking to.
const STANDING_GRANT: &str = "This also applies to any node of another engine added to the nodes \
     file later, without asking again.";

/// Longest prompt this route will compare. A comparison is a diagnostic, not a
/// serving path, and every byte here is generated against twice.
const MAX_COMPARE_PROMPT_BYTES: usize = 8 * 1024;

/// Ceiling on tokens generated per run, for the same reason.
const MAX_COMPARE_MAX_TOKENS: u32 = 1024;

#[derive(Debug, serde::Deserialize)]
struct CompareRequest {
    left: String,
    right: String,
    model: String,
    #[serde(default)]
    left_model: Option<String>,
    #[serde(default)]
    right_model: Option<String>,
    prompt: String,
    #[serde(default)]
    repetitions: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    max_tokens: Option<u32>,
    /// Send each side one short unrelated request between its runs, so an
    /// answer that depends on the engine's request history shows as
    /// instability. On unless a caller turns it off, and then the answer lists
    /// request history as uncontrolled.
    #[serde(default)]
    perturb_history: Option<bool>,
}

/// Ask two named nodes the same question and report what differed.
///
/// Behind the client key like every other route on this proxy, because unlike
/// `/v1/health` it *causes work*: it makes an authenticated caller generate on
/// two machines. The bounds above, and [`fabric::MAX_COMPARE_REPETITIONS`],
/// exist for that reason rather than for correctness, and are clamped rather
/// than rejected so a caller asking for more simply gets the maximum,
/// described in the answer it gets back. A temperature is refused instead:
/// clamping it would record a plan other than the one the caller asked about.
async fn compare(
    State(state): State<ServerState>,
    payload: std::result::Result<Json<CompareRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(rejection) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("could not read the comparison request: {rejection}"),
            )
        }
    };

    if request.prompt.len() > MAX_COMPARE_PROMPT_BYTES {
        return error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            &format!(
                "prompt is {} bytes; this route compares at most {MAX_COMPARE_PROMPT_BYTES}",
                request.prompt.len()
            ),
        );
    }

    if let Some(temperature) = request.temperature {
        if let Err(reason) = fabric::check_temperature(temperature) {
            return error_response(StatusCode::BAD_REQUEST, &reason);
        }
    }

    let plan = fabric::SamplingPlan {
        temperature: request.temperature.unwrap_or(0.0),
        seed: request.seed,
        max_tokens: request
            .max_tokens
            .unwrap_or(64)
            .clamp(1, MAX_COMPARE_MAX_TOKENS),
        repetitions: request
            .repetitions
            .unwrap_or(2)
            .clamp(1, fabric::MAX_COMPARE_REPETITIONS),
        history_perturbed: request.perturb_history.unwrap_or(true),
    };

    // Every run is a generation, which is what `forward_timeout` budgets. The
    // fabric's own timeout is the probe budget, and under it a node that took
    // longer than a health read to generate was reported as having failed.
    let fabric_handle = state
        .fabric
        .as_ref()
        .clone()
        .with_generation_timeout(state.config.forward_timeout);
    // Same reason as `health`: this is blocking socket I/O, and a long one.
    let outcome = tokio::task::spawn_blocking(move || {
        // An explicit per-side id wins; otherwise the operator's declared
        // aliases apply, so this route and the CLI resolve a model the same
        // way. A UI that quietly skipped the alias table would ask a node for
        // a name it does not know and be refused for no visible reason.
        match (&request.left_model, &request.right_model) {
            (None, None) => fabric_handle.compare_model(
                &request.left,
                &request.right,
                &request.model,
                &request.prompt,
                plan,
            ),
            _ => fabric_handle.compare(
                &request.left,
                &request.right,
                request.left_model.as_deref().unwrap_or(&request.model),
                request.right_model.as_deref().unwrap_or(&request.model),
                &request.prompt,
                plan,
            ),
        }
    })
    .await;

    match outcome {
        Ok(Ok(comparison)) => (StatusCode::OK, Json(serde_json::json!(comparison))).into_response(),
        // A comparison that could not be set up or run is a failed request, not
        // a verdict. Returning 200 with an empty comparison would let a caller
        // read "no difference" out of "we never asked".
        Ok(Err(error)) => error_response(compare_status(&error), &error.to_string()),
        Err(join_error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("the comparison did not complete: {join_error}"),
        ),
    }
}

/// Whose failure a comparison that did not complete was.
///
/// A caller's own mistakes — a label this fabric does not have, one node named
/// twice, a model the node does not hold — are theirs to fix and stay 400. A
/// node that is down or failed to generate is the upstream failing; calling
/// that a bad request sends the caller to fix a request that was fine.
fn compare_status(error: &fabric::CompareError) -> StatusCode {
    use fabric::{CompareError, SampleError};
    match error {
        CompareError::NoSuchNode { .. }
        | CompareError::SameNode { .. }
        | CompareError::Side(SampleError::ModelAbsent { .. }) => StatusCode::BAD_REQUEST,
        CompareError::Side(SampleError::NotServing { .. } | SampleError::Failed { .. }) => {
            StatusCode::BAD_GATEWAY
        }
    }
}

/// Build the health answer.
///
/// Ready means at least one node is ready, which is exactly what placement
/// needs: with one, some request can be served; with none, every request is
/// refused, and saying otherwise would invite traffic this proxy cannot serve.
///
/// `disclose_detail` decides whether anything beyond that verdict is included,
/// and is false unless the listener is loopback. This route is **not** behind
/// [`ClientAuth`]: the engine's shared `authenticate` keeps `/v1/health` public
/// so a probe needs no credential, and reusing that check means inheriting the
/// rule. An anonymous caller can therefore read whatever is here, which makes
/// the contents a disclosure decision rather than a formatting one:
///
/// * the node list is every label and address in the fabric;
/// * the model list is the answer `/v1/models` gives, and that route *is*
///   behind the key, so publishing it here would hand it out through a side
///   door;
/// * off-box callers get the verdict and nothing else, which is all a load
///   balancer reads anyway.
///
/// [`crate::api`] withholds its own `executable` and `listen_addr` on the same
/// fact, for the same reason.
///
/// Deliberately not shaped like a node's health: there is no `engine` field and
/// the service name differs, so nothing can mistake a proxy for something that
/// generates. A proxy has no single `active_model_id` to report anyway.
///
/// Kept pure so both rules are covered by unit tests rather than by starting a
/// server and guessing which branch ran.
fn health_report(
    snapshots: &[fabric::NodeSnapshot],
    disclose_detail: bool,
    view: &PlacementView,
) -> (StatusCode, Value) {
    let summary = fabric::FabricSummary::of(snapshots);
    // Readiness is "a request placed now would find a node", which is not the
    // same as "some node is healthy": a fabric whose only healthy node runs an
    // engine this proxy does not place on can serve nothing, and reporting it
    // ready would keep it in a load balancer's rotation while every request 503s.
    let ready = snapshots
        .iter()
        .any(|snapshot| snapshot.is_placeable_under(view.mixed));

    let mut body = serde_json::json!({
        "ok": true,
        "service": "camelid-fabric",
        "version": env!("CARGO_PKG_VERSION"),
        "build": crate::receipt::camelid_version(),
        "ready": ready,
    });

    if disclose_detail {
        if let Some(object) = body.as_object_mut() {
            object.insert(
                "nodes".to_string(),
                serde_json::json!({
                    "total": summary.total(),
                    "ready": summary.ready,
                    "not_ready": summary.not_ready,
                    "unreachable": summary.unreachable,
                }),
            );
            object.insert("models".to_string(), serde_json::json!(view.served));
            // Configuration, not observation, so it lives with the rest of
            // what only a loopback caller is told. Every sentence in it is
            // this build's, for the reason given on `STANDING_GRANT`.
            let consequences: Vec<Value> = if view.covers_nodes_added_later {
                vec![serde_json::json!({ "key": "standing_grant", "text": STANDING_GRANT })]
            } else {
                Vec::new()
            };
            object.insert(
                "placement".to_string(),
                serde_json::json!({
                    "mixed_engines": view.mixed.as_str(),
                    "flag": MIXED_ENGINES_FLAG,
                    "unreported_load_cost": UNREPORTED_LOAD_COST,
                    "cold_load_cost": COLD_LOAD_COST,
                    "max_forward_attempts": view.max_forward_attempts,
                    "models_if_mixed": view.served_if_mixed,
                    "covers_nodes_added_later": view.covers_nodes_added_later,
                    "consequences": consequences,
                    "foreign_nodes_added_since_start": view.foreign_added,
                }),
            );
            // The shape `fabric status --json` already prints, plus two fields
            // only a fabric can answer: whether *this* proxy would place work on
            // that node, and what its engine can be asked. A node can be
            // perfectly healthy and still not be somewhere work goes, and
            // nothing on the node itself says so.
            if let Ok(serde_json::Value::Array(mut detail)) = serde_json::to_value(snapshots) {
                for (entry, snapshot) in detail.iter_mut().zip(snapshots) {
                    if let Some(entry) = entry.as_object_mut() {
                        entry.insert(
                            "placeable".to_string(),
                            serde_json::Value::Bool(snapshot.is_placeable_under(view.mixed)),
                        );
                        let capabilities = snapshot.capabilities();
                        if let Ok(value) = serde_json::to_value(capabilities) {
                            entry.insert("capabilities".to_string(), value);
                        }
                        entry.insert(
                            "placement_blockers".to_string(),
                            serde_json::json!(capabilities.placement_blockers()),
                        );
                        entry.insert(
                            "placement_blocker_detail".to_string(),
                            serde_json::json!(
                                capabilities.placement_blocker_detail(UNREPORTED_LOAD_COST)
                            ),
                        );
                        entry.insert(
                            "requirement_limits".to_string(),
                            serde_json::json!(
                                capabilities.requirement_limits(&snapshot.engine_and_version())
                            ),
                        );
                    }
                }
                object.insert("node_detail".to_string(), serde_json::Value::Array(detail));
            }
        }
    }

    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, body)
}

/// List every model the fabric can currently route to.
///
/// A client points at one address, so it has to be able to ask that address
/// what it can serve; the per-node listing is not reachable through the proxy.
/// The answer is the union across ready nodes, in the same shape a node
/// answers, minus the per-file `meta`: two nodes can serve one model id from
/// different files, and there is no honest way to merge that.
///
/// No ready node means an empty list rather than an error. "Nothing right now"
/// is the truthful answer to a discovery question, and a model picker can show
/// it without special-casing a failure.
async fn models(State(state): State<ServerState>) -> Response {
    let fabric = Arc::clone(&state.fabric);
    // Observing is blocking socket I/O against every node, same as a dispatch.
    let snapshots = match tokio::task::spawn_blocking(move || fabric.observe()).await {
        Ok(snapshots) => snapshots,
        Err(join_error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("could not observe the fabric: {join_error}"),
            )
        }
    };

    let data: Vec<Value> = state
        .fabric
        .servable_models(&snapshots, state.config.mixed)
        .into_iter()
        .map(|model| listed(&model.id, model.identity))
        .collect();

    Json(serde_json::json!({ "object": "list", "data": data })).into_response()
}

/// Answer for one model id, the way an SDK's `models.retrieve` asks.
///
/// The verdict comes from placement itself rather than from a second rule about
/// what is servable: a model is retrievable here exactly when a request naming
/// it would be placed. That also inherits the distinction placement already
/// makes — a model no ready node serves is a settled 404, but only once every
/// node has been consulted; while any is unaccounted for the refusal can still
/// clear, so it stays a retryable 503.
async fn model(Path(id): Path<String>, State(state): State<ServerState>) -> Response {
    let fabric = Arc::clone(&state.fabric);
    let snapshots = match tokio::task::spawn_blocking(move || fabric.observe()).await {
        Ok(snapshots) => snapshots,
        Err(join_error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("could not observe the fabric: {join_error}"),
            )
        }
    };

    match state.fabric.decide(
        &snapshots,
        &RouteRequest::new(RouteMode::Throughput)
            .with_model(Some(&id))
            .with_mixed_engines(state.config.mixed),
    ) {
        Ok(decision) => Json(listed(
            &id,
            decision.model_identity.unwrap_or(ModelIdentity::SameId),
        ))
        .into_response(),
        Err(error) => route_error(error, disclose(&state)),
    }
}

/// One model, in the shape a node answers with. Pure.
fn model_object(id: &str) -> Value {
    serde_json::json!({
        "id": id,
        "object": "model",
        "created": 0,
        "owned_by": "camelid",
    })
}

/// A listed model, marked when what it names rests on an operator's word.
/// Plain entries keep exactly the shape they always had.
fn listed(id: &str, identity: ModelIdentity) -> Value {
    let mut object = model_object(id);
    if identity == ModelIdentity::AssertedByOperator {
        object["x_camelid_identity"] = Value::String("asserted_by_operator".to_string());
    }
    object
}

/// Refuse a route whose answer lives on one node.
///
/// A 501 rather than a 404: the route exists and a node implements it, so
/// "not here" with the reason is more use to a client than "no such thing".
async fn node_local() -> Response {
    error_response(
        StatusCode::NOT_IMPLEMENTED,
        "the Responses and Conversations APIs keep their state on the node that served the \
         request, so this proxy does not place them: a follow-up could land on another node and \
         find nothing. Send these to a node directly. This proxy serves the stateless routes and \
         model discovery",
    )
}

/// Answer a route this proxy does not serve, naming the ones it does.
async fn unknown_route() -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        &format!(
            "this fabric proxy does not serve that route; it serves {}, and GET /v1/models, \
             GET /v1/models/{{model}} and GET /v1/health",
            PLACED_ROUTES
                .iter()
                .map(|path| format!("POST {path}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}

/// Place one request on a node and relay its answer.
///
/// `path` is the matched route, not anything taken from the request line, so
/// what reaches a node is one of [`PLACED_ROUTES`] and nothing a client can
/// steer. The body is relayed unread beyond the two fields placement needs.
async fn place_and_forward(
    State(state): State<ServerState>,
    path: &'static str,
    headers: HeaderMap,
    payload: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let body = match payload {
        Ok(Json(body)) => body,
        Err(rejection) => return rejected_body(rejection),
    };
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    // A blank value is not a node name: taking it would silently switch this
    // request to affinity and then report affinity lost to a node called "".
    let sticky = headers
        .get(STICKY_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(str::to_string);
    let request = OwnedRequest { model, sticky };

    // Asked for by the request, not by the route: `stream: true` is meaningful
    // on the generation routes and meaningless on the rest, and a node that has
    // nothing to stream answers with a body, which the streaming path relays.
    if fabric::wants_streaming(&body) {
        return stream_completion(state, path, body, request).await;
    }
    buffered_completion(state, path, body, request).await
}

/// The parts of a [`RouteRequest`] that outlive the borrow of the request body,
/// so placement can run on another thread.
struct OwnedRequest {
    model: Option<String>,
    sticky: Option<String>,
}

/// Fires a request's [`Cancel`] when the frame holding it is dropped.
///
/// That drop is the only signal this proxy gets that a client has gone: axum
/// drops the handler future when the connection does, measured here at ~16 ms
/// after the client's socket closes. Holding one of these for exactly as long
/// as a client could still read the answer therefore turns "nobody is waiting"
/// into something the blocking dispatch can see.
///
/// It also fires on the way out of a request that was *served*, which costs
/// nothing: by then the dispatch has already returned its answer.
struct CancelOnDrop(Cancel);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
impl OwnedRequest {
    /// A client that names a node is asking for affinity to it, so the header
    /// settles the mode for its own request. Without this the header would be
    /// dead under the throughput default, since placement only consults a
    /// sticky label in [`RouteMode::Affinity`].
    fn as_route(&self, default_mode: RouteMode, mixed: MixedEngines) -> RouteRequest<'_> {
        let mode = match self.sticky {
            Some(_) => RouteMode::Affinity,
            None => default_mode,
        };
        RouteRequest::new(mode)
            .with_model(self.model.as_deref())
            .with_sticky(self.sticky.as_deref())
            .with_mixed_engines(mixed)
    }
}

async fn buffered_completion(
    state: ServerState,
    path: &'static str,
    body: Value,
    request: OwnedRequest,
) -> Response {
    let cancel = Cancel::new();
    // Bound to this frame, which axum drops when the client hangs up. Nothing
    // else can stop the task below: dropping the `JoinHandle` the await holds
    // detaches a blocking task rather than aborting it. Named, because binding
    // to `_` would drop it here and cancel the request before it started.
    let _client = CancelOnDrop(cancel.clone());
    let mixed = state.config.mixed;
    let disclose = disclose(&state);

    // Fabric::dispatch is synchronous socket I/O (probes every node, then
    // forwards) and can legitimately run for the whole forward_timeout — up to
    // minutes for a real generation. Running it directly on an async worker
    // thread would starve every other in-flight request once concurrent
    // dispatches reach the runtime's worker count; spawn_blocking moves it onto
    // tokio's much larger blocking pool instead.
    let outcome = tokio::task::spawn_blocking(move || {
        state.fabric.dispatch(
            path,
            &body,
            &request.as_route(state.config.mode, state.config.mixed),
            state.config.forward_timeout,
            &cancel,
        )
    })
    .await;

    match outcome {
        Ok(Ok(dispatched)) => {
            let status =
                StatusCode::from_u16(dispatched.answer.status).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut response = (status, Json(dispatched.answer.body)).into_response();
            tag(
                response.headers_mut(),
                &dispatched.decision,
                dispatched.attempts,
                mixed,
            );
            response
        }
        Ok(Err(DispatchError::Route(error))) => route_error(error, disclose),
        Ok(Err(DispatchError::Forward { error, engine })) => forward_error(error, engine),
        Err(join_error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("dispatch task did not complete: {join_error}"),
        ),
    }
}

/// What the blocking side reports once it has the node's response head.
///
/// Sent before any body byte, so the response status and the placement headers
/// are settled before the client is committed to reading a stream.
enum StreamStart {
    Streaming {
        decision: RouteDecision,
        attempts: usize,
        status: u16,
        content_type: Option<String>,
    },
    /// The node answered outright instead of streaming.
    Buffered {
        decision: RouteDecision,
        attempts: usize,
        answer: fabric::Forwarded,
    },
    Failed(DispatchError),
}

/// How many body pieces may sit between the node and a slow client.
///
/// Bounded on purpose: once it fills, the blocking reader stops reading its
/// socket, which is what pushes back on the node rather than buffering a
/// generation in memory.
const STREAM_CHANNEL_DEPTH: usize = 32;

async fn stream_completion(
    state: ServerState,
    path: &'static str,
    body: Value,
    request: OwnedRequest,
) -> Response {
    let (start_tx, start_rx) = tokio::sync::oneshot::channel::<StreamStart>();
    let (chunk_tx, mut chunk_rx) =
        tokio::sync::mpsc::channel::<Result<Vec<u8>, String>>(STREAM_CHANNEL_DEPTH);

    let cancel = Cancel::new();
    // Held by this frame only until the head is in; the streaming arm below
    // moves it into the response body, because that is where the client can
    // still be lost from. Until then a client leaving is invisible to the
    // channel below — nothing has been sent on it yet — and the wait for a head
    // is a whole prefill long.
    let client = CancelOnDrop(cancel.clone());
    let dispatch_cancel = cancel.clone();
    let mixed = state.config.mixed;
    let disclose = disclose(&state);

    // Same reasoning as the buffered path: this is blocking socket I/O for the
    // whole life of the generation, so it belongs on the blocking pool. It also
    // owns cancellation — every send below reports whether the client is still
    // there, and the socket to the node is dropped as soon as it is not.
    tokio::task::spawn_blocking(move || {
        let outcome = state.fabric.dispatch_streaming(
            path,
            &body,
            &request.as_route(state.config.mode, state.config.mixed),
            state.config.forward_timeout,
            state.config.forward_timeout,
            &dispatch_cancel,
        );

        let (mut streaming, mut placement) = match outcome {
            Err(error) => {
                let _ = start_tx.send(StreamStart::Failed(error));
                return;
            }
            Ok(dispatched) => {
                let decision = dispatched.placement.decision().clone();
                let attempts = dispatched.attempts;
                match dispatched.outcome {
                    StreamOutcome::Buffered(answer) => {
                        let _ = start_tx.send(StreamStart::Buffered {
                            decision,
                            attempts,
                            answer,
                        });
                        return;
                    }
                    StreamOutcome::Streaming(streaming) => {
                        let start = StreamStart::Streaming {
                            decision,
                            attempts,
                            status: streaming.status,
                            content_type: streaming.content_type.clone(),
                        };
                        if start_tx.send(start).is_err() {
                            return;
                        }
                        // Bound alongside the stream so the node stays reserved
                        // until the last event is read — however this loop ends.
                        (streaming, dispatched.placement)
                    }
                }
            }
        };
        let mut client_backpressured = false;

        loop {
            match streaming.next_chunk() {
                Ok(None) => {
                    if !client_backpressured {
                        placement.record_success(streaming.elapsed());
                    }
                    return;
                }
                Ok(Some(chunk)) => {
                    // A closed channel means the client hung up. Returning drops
                    // the node's socket with it, so the generation is not left
                    // running for nobody.
                    match chunk_tx.try_send(Ok(chunk)) {
                        Ok(()) => {}
                        Err(tokio::sync::mpsc::error::TrySendError::Full(chunk)) => {
                            // Once the channel fills, elapsed wall time includes
                            // client backpressure rather than only node service.
                            // Finish relaying, but never train placement on it.
                            client_backpressured = true;
                            if chunk_tx.blocking_send(chunk).is_err() {
                                return;
                            }
                        }
                        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => return,
                    }
                }
                Err(error) => {
                    if !matches!(error, ForwardError::Cancelled { .. }) {
                        placement.invalidate_service_time();
                    }
                    let _ = chunk_tx.blocking_send(Err(error.to_string()));
                    return;
                }
            }
        }
    });

    let start = match start_rx.await {
        Ok(start) => start,
        Err(_) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "dispatch task did not complete",
            )
        }
    };

    let (decision, attempts, status, content_type) = match start {
        StreamStart::Failed(DispatchError::Route(error)) => return route_error(error, disclose),
        StreamStart::Failed(DispatchError::Forward { error, engine }) => {
            return forward_error(error, engine)
        }
        StreamStart::Buffered {
            decision,
            attempts,
            answer,
        } => {
            let status = StatusCode::from_u16(answer.status).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut response = (status, Json(answer.body)).into_response();
            tag(response.headers_mut(), &decision, attempts, mixed);
            return response;
        }
        StreamStart::Streaming {
            decision,
            attempts,
            status,
            content_type,
        } => (decision, attempts, status, content_type),
    };

    let stream = async_stream::stream! {
        // Moved in, so it lives exactly as long as the response body does. A
        // client that leaves mid-stream is a dropped body, and the guard is
        // part of the stream's captured state from the moment it is built, so
        // a body dropped before it is ever polled still fires it.
        let _client = client;
        while let Some(chunk) = chunk_rx.recv().await {
            yield chunk;
        }
    };

    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    let out = response.headers_mut();
    // Only the node headers that describe the payload are relayed. Everything
    // else — Content-Length and the hop-by-hop set — describes the node's own
    // connection, and this response is framed independently of it.
    if let Some(content_type) = &content_type {
        insert(out, CONTENT_TYPE, content_type);
    }
    insert(out, CACHE_CONTROL, "no-cache");
    tag(out, &decision, attempts, mixed);
    response
}

/// Record which node served a request and why, on any answer shape.
///
/// Three headers are sent only where they say something a Camelid-only proxy
/// never had to: the model id sent, under mixed placement or when an alias
/// changed it; that the id rests on an operator's declaration, only then; and
/// the residency the node's own listing reported, under mixed placement and
/// only when it reported one. That last one is an observation, at most the
/// proxy's observation age (500 ms by default) older than the send, and an
/// engine that unloads on its own timer can make it stale. Without the flag
/// and without an alias, none of the three is sent. Failures are tagged by
/// [`forward_error`], not here.
fn tag(headers: &mut HeaderMap, decision: &RouteDecision, attempts: usize, mixed: MixedEngines) {
    insert(headers, "x-camelid-fabric-node", &decision.label);
    // Always sent, in every routing mode, so a client never has to know which
    // engines this fabric is willing to place on to find out which one answered.
    insert(headers, "x-camelid-fabric-engine", decision.engine.as_str());
    insert(headers, "x-camelid-fabric-reason", decision.reason.as_str());
    // Always sent, so a client reads a failover off the header rather than
    // inferring one from a header that is only there sometimes.
    insert(headers, "x-camelid-fabric-attempts", &attempts.to_string());
    if let Some(previous) = &decision.affinity_lost {
        insert(headers, "x-camelid-fabric-affinity-lost", previous);
    }
    let aliased = decision.model_identity == Some(ModelIdentity::AssertedByOperator);
    if let Some(model) = decision.model.as_deref() {
        if mixed == MixedEngines::Allowed || aliased {
            insert(headers, "x-camelid-fabric-model", model);
        }
    }
    if aliased {
        insert(
            headers,
            "x-camelid-fabric-model-identity",
            "asserted_by_operator",
        );
    }
    if mixed == MixedEngines::Allowed {
        match decision.residency {
            Some(residency @ (Residency::Resident | Residency::NotResident)) => {
                insert(
                    headers,
                    "x-camelid-fabric-residency-observed",
                    residency.as_str(),
                );
            }
            Some(Residency::Unknown) | None => {}
        }
    }
}

fn insert<K>(headers: &mut HeaderMap, name: K, value: &str)
where
    K: axum::http::header::IntoHeaderName,
{
    // A label or reason string can never contain characters invalid in a
    // header value in practice, but a malformed one must not crash the
    // response — dropping the header is strictly better than losing the body.
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(name, value);
    }
}

/// Answer a body axum refused in the same shape every other answer here uses.
///
/// Mirrors `crate::api::malformed_json_error`: the default `JsonRejection`
/// response is plain text, which would leave the rejection a client is most
/// likely to hit as the only one it cannot parse.
fn rejected_body(rejection: JsonRejection) -> Response {
    let status = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    };
    error_response(status, &rejection.to_string())
}

/// A refusal, in the words `disclose` allows: an off-box listener gets
/// [`RouteError::public_message`], which names no node and no engine version.
fn route_error(error: RouteError, disclose: bool) -> Response {
    let message = if disclose {
        error.to_string()
    } else {
        error.public_message()
    };
    match &error {
        // The client asked to go back to a node that cannot say its prefix is
        // still warm. Repeating the request unchanged will never succeed, and
        // 503 would tell an SDK to keep trying; dropping the sticky header
        // will, so say that instead.
        RouteError::AffinityUnsupported { .. } => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "affinity_unsupported",
                }
            })),
        )
            .into_response(),
        // Asking again will not make the model appear, and the nodes themselves
        // answer 404 `model_not_found` for exactly this. Answering 503 told
        // clients to retry, so an SDK spent its whole retry budget on a refusal
        // that could never change.
        //
        // Only when every node answered, though: a node still unaccounted for
        // may be the one that owns the model, and that refusal can clear.
        RouteError::ModelUnavailable { unobserved: 0, .. } => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "fabric_error",
                    "code": "model_not_found",
                    "param": "model",
                }
            })),
        )
            .into_response(),
        // Every node was asked and none that holds the model can do what the
        // request carries. Sending it again unchanged cannot succeed; sending
        // it without tools, or to a node that has the route, can. A caller's
        // request to change, so 400, like `affinity_unsupported`.
        RouteError::RequirementUnmet {
            requirement,
            unobserved: 0,
            ..
        } => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "capability_unavailable",
                    "param": if *requirement == "tool_calls" { "tools" } else { "route" },
                }
            })),
        )
            .into_response(),
        RouteError::ModelUnspecified { unobserved: 0 } => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "model_required",
                    "param": "model",
                }
            })),
        )
            .into_response(),
        // These can genuinely clear on their own, so they stay retryable.
        RouteError::ModelUnavailable { .. }
        | RouteError::RequirementUnmet { .. }
        | RouteError::ModelUnspecified { .. }
        | RouteError::NoNodesConfigured
        | RouteError::AllNodesUnavailable { .. } => {
            error_response(StatusCode::SERVICE_UNAVAILABLE, &message)
        }
    }
}

/// A request that did not complete, naming the node and — when it was sent
/// anywhere — the engine it was sent to, taken from the placement so a node
/// file re-declared mid-request cannot change which engine the answer names.
///
/// In every routing mode, the default one included: before mixed placement a
/// failure carried the node alone, and a failure that does not name its
/// engine is one a client cannot attribute (I2).
fn forward_error(error: ForwardError, engine: Option<NodeEngine>) -> Response {
    let message = error.to_string();
    // The node was reachable but refused unsupported input: that is the
    // caller's mistake, not the upstream's, so it is a 400 not a 502/503.
    let status = match &error {
        ForwardError::Unsupported(_) => StatusCode::BAD_REQUEST,
        // Nobody is left to read this. It is answered rather than invented as a
        // gateway failure so that the one thing that *can* still read it — a
        // test, or a caller of this function that is not the network — is not
        // told a healthy node failed.
        ForwardError::Cancelled { .. } => client_closed_request(),
        ForwardError::Unreachable { .. }
        | ForwardError::Transport { .. }
        | ForwardError::Json { .. } => StatusCode::BAD_GATEWAY,
    };
    let label = error.label().map(str::to_string);
    let mut response = error_response(status, &message);
    // The node that failed rides on the answer for the same reason [`tag`] puts
    // the node that served there: placement is read back off the response, so a
    // failure that does not carry its node is one nobody can attribute.
    if let Some(label) = label {
        insert(response.headers_mut(), "x-camelid-fabric-node", &label);
    }
    if let Some(engine) = engine {
        insert(
            response.headers_mut(),
            "x-camelid-fabric-engine",
            engine.as_str(),
        );
    }
    response
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({ "error": { "message": message, "type": "fabric_error" } })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::node::{NodeReady, NodeSnapshot, NodeSpec, NodeStatus};
    use crate::fabric::policy::RouteReason;
    use tower::ServiceExt;

    fn request(body: Value) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::new(body.to_string()))
            .expect("valid request")
    }

    async fn read_json(response: Response) -> (StatusCode, Value) {
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("collect body");
        (status, serde_json::from_slice(&bytes).expect("json body"))
    }

    fn get_request(uri: &str) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method("GET")
            .uri(uri)
            .body(axum::body::Body::empty())
            .expect("valid request")
    }

    /// A proxy that accepts every client. Authentication has its own tests;
    /// everything else here is about what the proxy does once a request is in.
    fn open_config() -> ServeConfig {
        ServeConfig {
            mode: RouteMode::Throughput,
            forward_timeout: Duration::from_millis(200),
            auth: ClientAuth::none(),
            tls: None,
            cors: None,
            mixed: MixedEngines::default(),
            bound: "127.0.0.1:8490".parse().expect("loopback address"),
        }
    }

    #[test]
    fn an_origin_the_engine_would_refuse_is_refused_by_the_proxy_too() {
        for origin in [
            "*",
            "null",
            "https://example.test/path",
            "https://example.test/?query=1",
        ] {
            let error = ProxyCors::resolve(&[origin.to_string()]).expect_err("refused");
            assert_eq!(
                error.kind(),
                std::io::ErrorKind::InvalidInput,
                "{origin}: {error}"
            );
        }
        assert!(
            ProxyCors::resolve(&[])
                .expect("nothing to refuse")
                .is_none(),
            "naming no origin is no CORS at all, not a layer that allows nothing"
        );
        let allowed = ProxyCors::resolve(&[
            "http://127.0.0.1:5173".to_string(),
            "https://ui.example/".to_string(),
        ])
        .expect("both are explicit origins")
        .expect("two origins were named");
        assert_eq!(allowed.origin_count(), 2);
    }

    /// A proxy on an address the network can reach. Integration tests always
    /// bind loopback, so this is the only place the redacted branch is reached.
    fn exposed_config() -> ServeConfig {
        ServeConfig {
            bound: "203.0.113.7:8490".parse().expect("routable address"),
            ..open_config()
        }
    }

    fn snapshot(label: &str, status: NodeStatus) -> NodeSnapshot {
        NodeSnapshot {
            spec: NodeSpec::camelid(label, "192.0.2.10", 8181),
            status,
            latency: Some(Duration::from_millis(3)),
        }
    }

    fn ready_status(model: &str) -> NodeStatus {
        NodeStatus::Ready(NodeReady {
            engine: crate::fabric::engine::NodeEngine::Camelid,
            active_model_id: Some(model.to_string()),
            models: vec![model.to_string()],
            resident_models: Some(vec![model.to_string()]),
            backend: Some("llama".to_string()),
            version: Some("0.6.1".to_string()),
            load: Some(crate::fabric::node::NodeLoad {
                in_flight: 0,
                waiting: 0,
            }),
        })
    }

    fn proxy(fabric: Fabric) -> Router {
        router(fabric, open_config())
    }

    /// The health answer of a Camelid-only proxy over `snapshots`, which is
    /// what every test written before mixed placement existed describes.
    fn report(snapshots: &[NodeSnapshot], disclose_detail: bool) -> (StatusCode, Value) {
        health_report(
            snapshots,
            disclose_detail,
            &PlacementView::of(&Fabric::new(Vec::new()), snapshots, MixedEngines::Refused),
        )
    }

    /// Node labels and engine versions identify what runs where. Health
    /// withholds them off-box, so a refusal must not hand them out instead —
    /// while the status a client acts on stays the same either way.
    #[test]
    fn an_exposed_listener_refuses_without_naming_engine_versions() {
        let error = RouteError::RequirementUnmet {
            requirement: "tool_calls",
            model: Some("only-on-ollama:latest".to_string()),
            unmet: vec![("b-ollama".to_string(), "ollama 0.33.2".to_string())],
            capable_elsewhere: vec!["a-camelid".to_string()],
            unobserved: 0,
        };
        let exposed = route_error(error.clone(), false);
        let loopback = route_error(error, true);
        assert_eq!(exposed.status(), StatusCode::BAD_REQUEST);
        assert_eq!(exposed.status(), loopback.status());

        let text = |response: Response| async move {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("body");
            String::from_utf8(bytes.to_vec()).expect("utf-8")
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        let exposed = runtime.block_on(text(exposed));
        let loopback = runtime.block_on(text(loopback));
        assert!(!exposed.contains("0.33.2"), "{exposed}");
        assert!(!exposed.contains("b-ollama"), "{exposed}");
        assert!(!exposed.contains("a-camelid"), "{exposed}");
        assert!(exposed.contains("capability_unavailable"), "{exposed}");
        assert!(loopback.contains("0.33.2"), "{loopback}");
        assert!(loopback.contains("b-ollama"), "{loopback}");
    }

    #[test]
    fn an_off_box_listener_does_not_disclose_the_routing_mode() {
        let snapshots = vec![snapshot("a", ready_status("llama-3b"))];
        let (_, withheld) = report(&snapshots, false);
        assert!(withheld.get("placement").is_none(), "{withheld}");
        let (_, disclosed) = report(&snapshots, true);
        assert_eq!(disclosed["placement"]["mixed_engines"], "refused");
        assert_eq!(disclosed["placement"]["flag"], MIXED_ENGINES_FLAG);
    }

    /// S5: a Camelid-only answer carries exactly the headers it always did.
    #[test]
    fn a_camelid_only_answer_carries_no_new_header() {
        let decision = RouteDecision {
            label: "a".to_string(),
            engine: NodeEngine::Camelid,
            reason: RouteReason::OnlyCandidate,
            affinity_lost: None,
            model: Some("m".to_string()),
            model_identity: Some(ModelIdentity::SameId),
            residency: Some(Residency::Resident),
        };
        let mut headers = HeaderMap::new();
        tag(&mut headers, &decision, 1, MixedEngines::Refused);
        let mut names: Vec<&str> = headers.keys().map(|name| name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "x-camelid-fabric-attempts",
                "x-camelid-fabric-engine",
                "x-camelid-fabric-node",
                "x-camelid-fabric-reason",
            ]
        );
    }

    /// Collects formatted log output, so these tests assert on what was
    /// actually written rather than on whether a function was reached.
    #[derive(Clone, Default)]
    struct Captured(Arc<std::sync::Mutex<Vec<u8>>>);

    impl Captured {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().expect("log buffer")).into_owned()
        }
    }

    impl std::io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("log buffer").extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Captured {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Drive one request and return what was logged while it ran.
    ///
    /// The subscriber is thread-local, so these tests run on the default
    /// current-thread runtime: on a multi-threaded one the task can resume on a
    /// thread that never had the subscriber installed and capture nothing.
    async fn logged(app: Router, request: axum::http::Request<axum::body::Body>) -> String {
        logged_at(tracing::Level::INFO, app, request).await
    }

    async fn logged_at(
        level: tracing::Level,
        app: Router,
        request: axum::http::Request<axum::body::Body>,
    ) -> String {
        logged_answer_at(level, app, request).await.0
    }

    /// The log and the answer together, for the fields that appear in both.
    async fn logged_answer(
        app: Router,
        request: axum::http::Request<axum::body::Body>,
    ) -> (String, Response) {
        logged_answer_at(tracing::Level::INFO, app, request).await
    }

    async fn logged_answer_at(
        level: tracing::Level,
        app: Router,
        request: axum::http::Request<axum::body::Body>,
    ) -> (String, Response) {
        let captured = Captured::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(captured.clone())
            .with_max_level(level)
            .with_ansi(false)
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        let response = app.oneshot(request).await.expect("router answers");
        drop(guard);
        (captured.text(), response)
    }

    /// A router whose answer carries whatever the fabric would have tagged, so
    /// the middleware can be tested without placing anything.
    fn answering(status: StatusCode, node: Option<&'static str>) -> Router {
        Router::new()
            .route(
                "/v1/chat/completions",
                post(move || async move {
                    let mut response = (status, "body").into_response();
                    if let Some(label) = node {
                        let decision = RouteDecision {
                            label: label.to_string(),
                            engine: crate::fabric::engine::NodeEngine::Camelid,
                            reason: RouteReason::LeastLoaded,
                            affinity_lost: None,
                            model: None,
                            model_identity: None,
                            residency: None,
                        };
                        tag(response.headers_mut(), &decision, 1, MixedEngines::Refused);
                    }
                    response
                }),
            )
            .layer(middleware::from_fn(access_log))
    }

    /// The proxy announced its address and then said nothing ever again.
    #[tokio::test]
    async fn a_request_is_recorded_with_its_method_path_and_status() {
        let written = logged(proxy(Fabric::new(Vec::new())), get_request("/v1/models")).await;
        assert!(written.contains("fabric request"), "{written}");
        assert!(written.contains("method=GET"), "{written}");
        assert!(written.contains("path=/v1/models"), "{written}");
        assert!(written.contains("status=200"), "{written}");
        assert!(written.contains("elapsed_ms"), "{written}");
    }

    /// Which machine served a request is the one fact a generic HTTP layer
    /// cannot know, and the only reason this proxy exists.
    #[tokio::test]
    async fn the_node_that_served_a_request_is_recorded() {
        let written = logged(
            answering(StatusCode::OK, Some("mac-studio")),
            request(serde_json::json!({ "model": "m" })),
        )
        .await;
        assert!(written.contains("mac-studio"), "{written}");
        assert!(written.contains("LeastLoaded"), "{written}");
    }

    /// An answer that never reached a node still gets a line; it just has no
    /// node to name, and an empty field would read like a missing one.
    #[tokio::test]
    async fn an_answer_that_reached_no_node_records_a_dash() {
        let written = logged(
            answering(StatusCode::OK, None),
            request(serde_json::json!({ "model": "m" })),
        )
        .await;
        assert!(written.contains("node=\"-\""), "{written}");
        assert!(written.contains("reason=\"-\""), "{written}");
    }

    /// The log layer sits outside authentication on purpose: a refused request
    /// is exactly the one worth having a record of.
    #[tokio::test]
    async fn a_refused_request_is_still_recorded() {
        let config = ServeConfig {
            auth: ClientAuth::resolve(Some("s3cret".to_string()), None).expect("a key resolves"),
            ..open_config()
        };
        let mut refused = get_request("/v1/models");
        // Actually presented, so the assertion below is about a credential that
        // reached the middleware rather than one that was never sent.
        refused.headers_mut().insert(
            "authorization",
            HeaderValue::from_static("Bearer presented-9f3a"),
        );
        let written = logged(router(Fabric::new(Vec::new()), config), refused).await;
        assert!(written.contains("status=401"), "{written}");
        // No request header reaches the log, and one of them is the client's key.
        assert!(!written.contains("presented-9f3a"), "{written}");
        assert!(!written.contains("s3cret"), "{written}");
    }

    /// An operator narrowing to failures must still see them, and must not have
    /// every healthy request in the way.
    #[tokio::test]
    async fn a_failure_is_recorded_at_warn_and_a_success_is_not() {
        let failed = logged_at(
            tracing::Level::WARN,
            answering(StatusCode::BAD_GATEWAY, Some("node-a")),
            request(serde_json::json!({ "model": "m" })),
        )
        .await;
        assert!(failed.contains("fabric request failed"), "{failed}");
        assert!(failed.contains("status=502"), "{failed}");

        let served = logged_at(
            tracing::Level::WARN,
            answering(StatusCode::OK, Some("node-a")),
            request(serde_json::json!({ "model": "m" })),
        )
        .await;
        assert!(
            served.is_empty(),
            "a served request must not warn: {served}"
        );
    }

    /// A node that could not be reached is the failure most worth attributing,
    /// and it is the one an operator is reading WARN to find. The error already
    /// carries the label, so a line that cannot name the node is a line that
    /// sends them to a packet capture anyway.
    #[tokio::test]
    async fn a_failure_to_reach_a_node_records_which_node() {
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(|| async {
                    forward_error(
                        ForwardError::Transport {
                            label: "node-a".to_string(),
                            detail: "connection refused".to_string(),
                        },
                        None,
                    )
                }),
            )
            .layer(middleware::from_fn(access_log));

        let written = logged_at(
            tracing::Level::WARN,
            app,
            request(serde_json::json!({ "model": "m" })),
        )
        .await;
        assert!(written.contains("status=502"), "{written}");
        assert!(written.contains("node=\"node-a\""), "{written}");
    }

    /// A query string can carry anything a client puts there, and the path
    /// alone already identifies the route.
    #[tokio::test]
    async fn a_query_string_is_not_recorded() {
        let written = logged(
            proxy(Fabric::new(Vec::new())),
            get_request("/v1/models?token=s3cret&x=1"),
        )
        .await;
        assert!(written.contains("path=/v1/models"), "{written}");
        assert!(!written.contains("s3cret"), "{written}");
    }

    /// An id assigned upstream has to survive the hop, or the two sides of a
    /// load balancer describe the same request by different names.
    #[tokio::test]
    async fn an_id_the_client_supplied_is_the_one_used() {
        let mut request = get_request("/v1/models");
        request
            .headers_mut()
            .insert(REQUEST_ID_HEADER, HeaderValue::from_static("abc-123"));

        let (written, response) = logged_answer(proxy(Fabric::new(Vec::new())), request).await;
        assert!(written.contains("request_id=\"abc-123\""), "{written}");
        assert_eq!(
            response.headers().get(REQUEST_ID_HEADER).unwrap(),
            "abc-123",
            "the client has to be told which id its request was recorded under"
        );
    }

    /// Without one there is nothing to quote in a complaint, so the proxy makes
    /// one rather than logging a dash.
    #[tokio::test]
    async fn a_request_with_no_id_is_given_one() {
        let (written, response) =
            logged_answer(proxy(Fabric::new(Vec::new())), get_request("/v1/models")).await;
        let issued = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .expect("an id is always answered")
            .to_str()
            .expect("printable");
        assert!(!issued.is_empty() && issued != "-", "{issued}");
        assert!(
            written.contains(issued),
            "the answer's id must be the logged one: {written}"
        );
    }

    /// The id is written into a line the client does not otherwise control, so
    /// an unusable one is replaced outright — not trimmed into something that
    /// still looks like the caller's id but no longer is.
    #[tokio::test]
    async fn an_unusable_id_is_replaced_rather_than_cleaned_up() {
        let mut request = get_request("/v1/models");
        request.headers_mut().insert(
            REQUEST_ID_HEADER,
            HeaderValue::from_static("has space and\ttab"),
        );

        let (written, response) = logged_answer(proxy(Fabric::new(Vec::new())), request).await;
        assert!(!written.contains("has space"), "{written}");
        assert!(
            !written.contains('\t'),
            "a tab would break the line: {written}"
        );
        let issued = response.headers().get(REQUEST_ID_HEADER).expect("an id");
        assert_ne!(issued, "has space and\ttab");
    }

    #[test]
    fn what_counts_as_a_usable_id() {
        assert!(is_usable_request_id("9f3a-1b2c"));
        assert!(is_usable_request_id(&"a".repeat(MAX_REQUEST_ID_LEN)));

        assert!(!is_usable_request_id(""), "nothing to correlate on");
        assert!(!is_usable_request_id(&"a".repeat(MAX_REQUEST_ID_LEN + 1)));
        assert!(!is_usable_request_id("has space"));
        assert!(!is_usable_request_id("has\ttab"));
        assert!(!is_usable_request_id("has\u{7f}del"));
    }

    /// Who made the request is half of an access log. The service that supplies
    /// it is only built in [`serve_on_until`], so the middleware has to keep
    /// working without it — every test that drives the router directly, and
    /// every one of them would otherwise fail.
    #[tokio::test]
    async fn the_client_is_recorded_when_the_service_supplies_one() {
        let mut request = get_request("/v1/models");
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 0, 2, 10], 51234))));

        let written = logged(proxy(Fabric::new(Vec::new())), request).await;
        assert!(written.contains("client=\"192.0.2.10:51234\""), "{written}");
    }

    #[tokio::test]
    async fn a_request_with_no_client_information_still_records_a_line() {
        let written = logged(proxy(Fabric::new(Vec::new())), get_request("/v1/models")).await;
        assert!(written.contains("fabric request"), "{written}");
        assert!(written.contains("client=\"-\""), "{written}");
    }

    /// Ready means "placement can succeed", which is exactly one ready node.
    #[test]
    fn health_is_ready_when_one_node_can_serve() {
        let snapshots = vec![
            snapshot("a", ready_status("llama-3b")),
            snapshot(
                "b",
                NodeStatus::Unreachable {
                    reason: "cannot connect".to_string(),
                },
            ),
        ];
        let (status, body) = report(&snapshots, true);
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["ok"], serde_json::json!(true));
        assert_eq!(body["ready"], serde_json::json!(true));
        assert_eq!(body["nodes"]["total"], serde_json::json!(2));
        assert_eq!(body["nodes"]["ready"], serde_json::json!(1));
        assert_eq!(body["nodes"]["unreachable"], serde_json::json!(1));
        assert_eq!(body["models"], serde_json::json!(["llama-3b"]));
    }

    /// A proxy with nothing to route to must not invite traffic. `ok` stays
    /// true through it: the process is fine, its fabric is not, and restarting
    /// this process would fix nothing.
    #[test]
    fn health_refuses_traffic_when_no_node_is_ready() {
        let snapshots = vec![snapshot(
            "a",
            NodeStatus::NotReady {
                reason: "no model loaded".to_string(),
            },
        )];
        let (status, body) = report(&snapshots, true);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["ok"], serde_json::json!(true));
        assert_eq!(body["ready"], serde_json::json!(false));
        assert_eq!(body["models"], serde_json::json!([]));
    }

    /// Readiness is "a request placed now would find a node", not "some node is
    /// healthy". A load balancer that kept this proxy in rotation on the second
    /// reading would send it traffic that can only 503.
    #[test]
    fn a_fabric_whose_only_healthy_node_is_unplaceable_is_not_ready() {
        let mut spec = NodeSpec::camelid("studio", "192.0.2.10", 11434);
        spec.engine = crate::fabric::engine::NodeEngine::Ollama;
        let foreign = NodeSnapshot {
            spec,
            status: NodeStatus::Ready(NodeReady {
                engine: crate::fabric::engine::NodeEngine::Ollama,
                active_model_id: None,
                models: vec!["mistral:latest".to_string()],
                resident_models: Some(Vec::new()),
                backend: None,
                version: Some("0.33.3".to_string()),
                load: None,
            }),
            latency: Some(Duration::from_millis(9)),
        };

        let (status, body) = report(std::slice::from_ref(&foreign), true);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["ready"], serde_json::json!(false));
        // The node is still reported as the healthy machine it is...
        assert_eq!(body["nodes"]["ready"], serde_json::json!(1));
        assert_eq!(
            body["node_detail"][0]["status"]["models"],
            serde_json::json!(["mistral:latest"])
        );
        assert_eq!(
            body["node_detail"][0]["placeable"],
            serde_json::json!(false)
        );
        // ...but nothing it holds is advertised, because nothing would be routed.
        assert_eq!(body["models"], serde_json::json!([]));
        // And a load it never published stays absent rather than becoming zero.
        assert!(
            body["node_detail"][0]["status"].get("in_flight").is_none(),
            "{body}"
        );
    }

    /// The verdict itself does not depend on who is asking; only the detail does.
    #[test]
    fn the_readiness_verdict_is_the_same_off_box() {
        let snapshots = vec![snapshot("a", ready_status("llama-3b"))];
        let (disclosed, _) = report(&snapshots, true);
        let (withheld, body) = report(&snapshots, false);
        assert_eq!(disclosed, withheld);
        assert_eq!(body["ready"], serde_json::json!(true));
    }

    #[test]
    fn a_loopback_listener_may_name_the_fabric() {
        let snapshots = vec![snapshot("a", ready_status("llama-3b"))];
        let (_, body) = report(&snapshots, true);
        let detail = body["node_detail"].as_array().expect("node detail");
        assert_eq!(detail.len(), 1);
        assert_eq!(detail[0]["spec"]["label"], serde_json::json!("a"));
        assert_eq!(detail[0]["spec"]["host"], serde_json::json!("192.0.2.10"));
    }

    /// This route needs no key, because the engine's shared `authenticate`
    /// keeps `/v1/health` public. So an exposed proxy answers anyone, and what
    /// it answers must not include every node's address.
    #[test]
    fn an_exposed_listener_does_not_name_the_fabric() {
        let snapshots = vec![snapshot("a", ready_status("llama-3b"))];
        let (_, body) = report(&snapshots, false);
        assert!(body.get("node_detail").is_none(), "{body}");
        assert!(body.get("nodes").is_none(), "{body}");
        assert!(!body.to_string().contains("192.0.2.10"), "{body}");
    }

    /// `/v1/models` is behind the key and has a test saying an unauthenticated
    /// caller must not learn what is served. Health needs no key, so listing
    /// models here off-box would give that same answer away through a side door.
    #[test]
    fn an_exposed_listener_does_not_list_the_models() {
        let snapshots = vec![snapshot("a", ready_status("private-model"))];
        let (_, body) = report(&snapshots, false);
        assert!(body.get("models").is_none(), "{body}");
        assert!(!body.to_string().contains("private-model"), "{body}");
    }

    /// Proves the route reads the bound address rather than merely that the
    /// pure function can redact. Integration tests all bind loopback, so the
    /// exposed branch has no other end-to-end coverage.
    #[tokio::test]
    async fn only_a_loopback_listener_names_the_fabric_on_the_route() {
        let exposed = router(Fabric::new(Vec::new()), exposed_config())
            .oneshot(get_request("/v1/health"))
            .await
            .expect("router answers");
        let (_, exposed_body) = read_json(exposed).await;
        assert!(exposed_body.get("node_detail").is_none(), "{exposed_body}");

        let loopback = proxy(Fabric::new(Vec::new()))
            .oneshot(get_request("/v1/health"))
            .await
            .expect("router answers");
        let (_, loopback_body) = read_json(loopback).await;
        assert_eq!(loopback_body["node_detail"], serde_json::json!([]));
    }

    /// An empty fabric is not a broken proxy, but it cannot take traffic.
    #[tokio::test]
    async fn the_health_route_refuses_an_empty_fabric() {
        let response = proxy(Fabric::new(Vec::new()))
            .oneshot(get_request("/v1/health"))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["ready"], serde_json::json!(false));
        assert_eq!(body["nodes"]["total"], serde_json::json!(0));
        assert_eq!(body["service"], serde_json::json!("camelid-fabric"));
    }

    /// The 404 is built from the route table, so this can only fail if the two
    /// are ever allowed to drift — which is the whole reason the table exists.
    #[tokio::test]
    async fn an_unserved_route_names_every_route_that_is_served() {
        let response = proxy(Fabric::new(Vec::new()))
            .oneshot(get_request("/v1/no-such-thing"))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "fabric_error");
        let message = body["error"]["message"].as_str().expect("a message");
        for path in PLACED_ROUTES {
            assert!(
                message.contains(path),
                "`{path}` is missing from: {message}"
            );
        }
        assert!(message.contains("/v1/models"), "{message}");
        assert!(message.contains("/v1/health"), "{message}");
    }

    /// A route is either placed or refused as node-local, never both: the two
    /// tables are what axum builds its router from, and registering one path
    /// twice is a panic at startup rather than a test failure here.
    #[test]
    fn no_route_is_both_placed_and_node_local() {
        for placed in PLACED_ROUTES {
            assert!(
                !NODE_LOCAL_ROUTES.contains(&placed),
                "`{placed}` is in both tables"
            );
        }
    }

    #[tokio::test]
    async fn a_node_local_route_is_refused_with_its_reason_rather_than_placed() {
        // An empty fabric: a placed route would refuse with a routing error, so
        // a 501 here proves the refusal happens before any placement at all.
        let response = proxy(Fabric::new(Vec::new()))
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::new(
                        serde_json::json!({ "model": "m" }).to_string(),
                    ))
                    .expect("valid request"),
            )
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(body["error"]["type"], "fabric_error");
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(
            message.contains("state") && message.contains("node"),
            "the refusal has to say why, not just no: {message}"
        );
    }

    /// Every method, not just the one an SDK happens to use first: a GET of a
    /// stored response is as node-local as the POST that created it.
    #[tokio::test]
    async fn a_node_local_route_is_refused_on_every_method() {
        for uri in ["/v1/responses/resp_1", "/v1/conversations/conv_1/items"] {
            let response = proxy(Fabric::new(Vec::new()))
                .oneshot(get_request(uri))
                .await
                .expect("router answers");
            assert_eq!(
                response.status(),
                StatusCode::NOT_IMPLEMENTED,
                "GET {uri} should be refused as node-local"
            );
        }
    }

    /// A client points at one address, so that address has to answer what it
    /// can serve. This used to 404, which left every model picker empty.
    #[tokio::test]
    async fn the_models_route_answers_a_list_even_with_nothing_to_serve() {
        let response = proxy(Fabric::new(Vec::new()))
            .oneshot(get_request("/v1/models"))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        assert_eq!(
            body["data"].as_array().expect("a data array").len(),
            0,
            "nothing ready is an empty list, not an error"
        );
    }

    /// The empty case above passes just as well against a route that can only
    /// ever answer nothing, so this is the one that proves it observes.
    #[tokio::test]
    async fn the_models_route_lists_what_a_ready_node_is_serving() {
        let node = StubModelNode::start("llama-3b");
        let fabric = Fabric::new(vec![node.spec("only")]).with_timeout(Duration::from_secs(2));
        let response = proxy(fabric)
            .oneshot(get_request("/v1/models"))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().expect("a data array");
        assert_eq!(data.len(), 1, "{body}");
        assert_eq!(data[0]["id"], "llama-3b");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "camelid");
    }

    /// `models.retrieve` on an SDK. The answer is the same object the list
    /// carries, so a client that found a model in one can look it up in the
    /// other and get something it recognises.
    #[tokio::test]
    async fn a_served_model_can_be_retrieved_on_its_own() {
        let node = StubModelNode::start("llama-3b");
        let fabric = Fabric::new(vec![node.spec("only")]).with_timeout(Duration::from_secs(2));
        let response = proxy(fabric)
            .oneshot(get_request("/v1/models/llama-3b"))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["id"], "llama-3b");
        assert_eq!(body["object"], "model");
        assert_eq!(body["owned_by"], "camelid");
    }

    /// Retrieval and placement must not be able to disagree: a model this
    /// answers 200 for is one a request naming it would be placed on, because
    /// both ask the same question of the same observation.
    #[tokio::test]
    async fn retrieving_an_unserved_model_refuses_exactly_as_placing_it_would() {
        let node = StubModelNode::start("llama-3b");
        let fabric = Fabric::new(vec![node.spec("only")]).with_timeout(Duration::from_secs(2));
        let response = proxy(fabric)
            .oneshot(get_request("/v1/models/no-such-model"))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(message.contains("llama-3b"), "{message}");
    }

    /// And it inherits the other half of that distinction too: while a node is
    /// unaccounted for, "not found" is not settled and must stay retryable.
    #[tokio::test]
    async fn retrieving_a_model_while_a_node_is_down_stays_retryable() {
        let node = StubModelNode::start("llama-3b");
        let fabric = Fabric::new(vec![
            node.spec("up"),
            // Port 1 is closed, so this node is observed as unreachable.
            NodeSpec::camelid("down", "127.0.0.1", 1),
        ])
        .with_timeout(Duration::from_millis(500));
        let response = proxy(fabric)
            .oneshot(get_request("/v1/models/no-such-model"))
            .await
            .expect("router answers");

        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a node that could not be consulted may be the one that owns it"
        );
    }

    /// A model this fabric does not serve will not appear by asking again, and
    /// the nodes answer 404 `model_not_found` for it. Answering 503 told
    /// clients to retry: an OpenAI SDK spent its whole retry budget on a
    /// permanent refusal.
    #[tokio::test]
    async fn an_unservable_model_is_refused_as_not_found_not_as_try_again() {
        let node = StubModelNode::start("llama-3b");
        let fabric = Fabric::new(vec![node.spec("only")]).with_timeout(Duration::from_secs(2));
        let response = proxy(fabric)
            .oneshot(request(
                serde_json::json!({ "model": "no-such-model", "messages": [] }),
            ))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
        // The message still names what the fabric *can* serve, which is the
        // part an operator acts on.
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(message.contains("llama-3b"), "{message}");
    }

    /// The same refusal is *not* final when a node could not be consulted: the
    /// node that is down may be the one that owns the model, so a 404 would
    /// tell an SDK to give up on something a retry would find. This is the
    /// difference between a model that is absent and a fabric that cannot yet
    /// say.
    #[tokio::test]
    async fn a_model_missing_only_because_a_node_is_down_stays_retryable() {
        let node = StubModelNode::start("llama-3b");
        let fabric = Fabric::new(vec![
            node.spec("up"),
            // Port 1 is closed, so this node is observed as unreachable.
            NodeSpec::camelid("down", "127.0.0.1", 1),
        ])
        .with_timeout(Duration::from_millis(500));

        let response = proxy(fabric)
            .oneshot(request(
                serde_json::json!({ "model": "no-such-model", "messages": [] }),
            ))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;

        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "a node that never answered may be the one serving it: {body}"
        );
    }

    /// A fabric whose nodes are merely unreachable can recover on its own, so
    /// that refusal stays retryable.
    #[tokio::test]
    async fn an_unreachable_fabric_is_still_a_retryable_503() {
        let fabric = Fabric::new(vec![NodeSpec::camelid("dead", "127.0.0.1", 1)])
            .with_timeout(Duration::from_millis(200));
        let response = proxy(fabric)
            .oneshot(request(serde_json::json!({ "model": "m", "messages": [] })))
            .await
            .expect("router answers");
        let (status, _) = read_json(response).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A node stub answering only `/v1/health`, which is all placement reads.
    struct StubModelNode {
        port: u16,
        shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl StubModelNode {
        fn start(model: &str) -> Self {
            use std::io::{Read, Write};
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
            let port = listener.local_addr().expect("addr").port();
            let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = std::sync::Arc::clone(&shutdown);
            let body = format!(
                r#"{{"ok":true,"generation_ready":true,"active_model_id":"{model}","backend":"llama","version":"0.6.1","engine_queued_tasks":0,"engine_queue_depth":0}}"#
            );
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if flag.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut stream) = stream else { continue };
                    // Read to the end of the head before answering: closing a
                    // socket that still holds unread bytes is an abortive close
                    // on Windows, and it discards the reply along with them.
                    let mut request = Vec::new();
                    let mut scratch = [0_u8; 1024];
                    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                        match stream.read(&mut scratch) {
                            Ok(0) | Err(_) => break,
                            Ok(read) => request.extend_from_slice(&scratch[..read]),
                        }
                    }
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes());
                }
            });
            Self { port, shutdown }
        }

        fn spec(&self, label: &str) -> NodeSpec {
            NodeSpec::camelid(label, "127.0.0.1", self.port)
        }
    }

    impl Drop for StubModelNode {
        fn drop(&mut self) {
            self.shutdown
                .store(true, std::sync::atomic::Ordering::SeqCst);
            let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
        }
    }

    #[tokio::test]
    async fn an_empty_fabric_answers_503_not_a_hang() {
        let fabric = Fabric::new(Vec::new());
        let router = router(fabric, open_config());
        let response = router
            .oneshot(request(serde_json::json!({ "model": "m" })))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "fabric_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no nodes configured"));
    }

    #[tokio::test]
    async fn a_streaming_request_is_routed_rather_than_refused() {
        // The node is unreachable on purpose. A 503 — a placement failure —
        // proves the proxy tried to route the stream; the 400 this used to
        // answer would mean it had refused before looking at the fabric.
        let fabric = Fabric::new(vec![NodeSpec::camelid("dead", "127.0.0.1", 1)]);
        let router = router(fabric, open_config());
        let response = router
            .oneshot(request(serde_json::json!({ "model": "m", "stream": true })))
            .await
            .expect("router answers");
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "fabric_error");
    }

    #[tokio::test]
    async fn a_malformed_body_is_rejected_before_it_reaches_the_fabric() {
        let fabric = Fabric::new(Vec::new());
        let router = router(fabric, open_config());
        let bad = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::new("not json".to_string()))
            .expect("built request");
        let response = router.oneshot(bad).await.expect("router answers");
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        // The rejection a client is most likely to hit must not be the one
        // error it cannot parse: axum's own JsonRejection body is plain text.
        assert_eq!(body["error"]["type"], "fabric_error");
        assert!(!body["error"]["message"].as_str().unwrap().is_empty());
    }

    /// A body the node behind the proxy would accept must not be refused by the
    /// proxy. Axum's default limit is 2 MiB, eight times stricter than the
    /// node's own default.
    #[tokio::test]
    async fn a_body_over_the_axum_default_still_reaches_the_fabric() {
        let fabric = Fabric::new(Vec::new());
        let router = router(fabric, open_config());
        let body = serde_json::json!({ "model": "m", "pad": "x".repeat(4 * 1024 * 1024) });
        let response = router.oneshot(request(body)).await.expect("router answers");
        let (status, body) = read_json(response).await;
        // 503 means the body was read and placement ran; 413 would mean the
        // proxy refused a request the node would have served.
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "fabric_error");
    }

    #[tokio::test]
    async fn a_body_over_the_proxy_limit_is_refused_in_the_fabric_error_shape() {
        let fabric = Fabric::new(Vec::new());
        let router = router(fabric, open_config());
        let oversize = crate::api::DEFAULT_MAX_REQUEST_BODY_BYTES + 1024;
        let body = serde_json::json!({ "model": "m", "pad": "x".repeat(oversize) });
        let response = router.oneshot(request(body)).await.expect("router answers");
        let (status, body) = read_json(response).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(body["error"]["type"], "fabric_error");
    }

    #[test]
    fn a_loopback_listener_needs_no_acknowledgement() {
        refuse_unauthenticated_remote("127.0.0.1:8282".parse().unwrap(), false, false)
            .expect("loopback is not exposed");
        refuse_unauthenticated_remote("[::1]:8282".parse().unwrap(), false, false)
            .expect("loopback is not exposed");
    }

    #[test]
    fn an_unauthenticated_non_loopback_listener_is_refused_by_default() {
        let error = refuse_unauthenticated_remote("0.0.0.0:8282".parse().unwrap(), false, false)
            .expect_err("every node in the fabric would be exposed");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        // The refusal has to name the way out of it, or it is a dead end.
        assert!(
            error.to_string().contains("--allow-unauthenticated-remote"),
            "{error}"
        );
        // ...including both ways out that do not give up authentication.
        assert!(error.to_string().contains("--api-key"), "{error}");
        assert!(error.to_string().contains("--client-keys"), "{error}");
    }

    #[test]
    fn a_non_loopback_listener_starts_once_the_risk_is_acknowledged() {
        refuse_unauthenticated_remote("0.0.0.0:8282".parse().unwrap(), false, true)
            .expect("the operator accepted the exposure");
    }

    /// Requiring a key is what the acknowledgement is an alternative to, so a
    /// key has to satisfy the guard on its own.
    #[test]
    fn a_key_lets_a_non_loopback_listener_start_without_acknowledging_anything() {
        refuse_unauthenticated_remote("0.0.0.0:8282".parse().unwrap(), true, false)
            .expect("the listener is authenticated");
    }

    #[test]
    fn a_loopback_listener_may_serve_cleartext() {
        refuse_cleartext_remote("127.0.0.1:8282".parse().unwrap(), false, false)
            .expect("loopback never reaches the network");
        refuse_cleartext_remote("[::1]:8282".parse().unwrap(), false, false)
            .expect("loopback never reaches the network");
    }

    #[test]
    fn a_cleartext_non_loopback_listener_is_refused_by_default() {
        let error = refuse_cleartext_remote("0.0.0.0:8282".parse().unwrap(), false, false)
            .expect_err("the key and every prompt would cross the network in the clear");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        // The refusal has to name the way out of it, or it is a dead end.
        assert!(
            error.to_string().contains("--allow-cleartext-remote"),
            "{error}"
        );
        // ...including the way out that does not give up encryption.
        assert!(error.to_string().contains("--tls-cert"), "{error}");
    }

    #[test]
    fn a_certificate_lets_a_non_loopback_listener_start_without_acknowledging_anything() {
        refuse_cleartext_remote("0.0.0.0:8282".parse().unwrap(), true, false)
            .expect("the listener presents a certificate");
    }

    #[test]
    fn a_cleartext_listener_starts_once_the_risk_is_acknowledged() {
        refuse_cleartext_remote("0.0.0.0:8282".parse().unwrap(), false, true)
            .expect("the operator accepted the exposure");
    }

    /// The two guards protect against different things, so neither may stand in
    /// for the other. A key over cleartext is the case this whole change exists
    /// for: the credential is sent on every request, so the flag that is meant
    /// to protect the bind is exactly what the bind gives away.
    #[test]
    fn neither_guard_satisfies_the_other() {
        let exposed: SocketAddr = "0.0.0.0:8282".parse().unwrap();

        // Authenticated, but still cleartext.
        refuse_unauthenticated_remote(exposed, true, false).expect("a key answers the auth guard");
        refuse_cleartext_remote(exposed, false, false)
            .expect_err("a key does not encrypt itself on the wire");

        // Encrypted, but still unauthenticated.
        refuse_cleartext_remote(exposed, true, false).expect("a certificate answers the TLS guard");
        refuse_unauthenticated_remote(exposed, false, false)
            .expect_err("TLS does not decide who may drive the fabric");

        // And acknowledging one does not acknowledge the other.
        refuse_unauthenticated_remote(exposed, false, true).expect("auth risk accepted");
        refuse_cleartext_remote(exposed, false, false)
            .expect_err("accepting anonymity is not accepting cleartext");
    }

    /// Half a pair is a mistake, and serving cleartext because of it would be
    /// the one outcome the operator plainly did not ask for. Refused before
    /// either file is read, so it reads the same whether or not they exist.
    #[tokio::test]
    async fn a_certificate_needs_its_key_and_a_key_needs_its_certificate() {
        assert!(ProxyTls::resolve(None, None)
            .await
            .expect("no certificate is not an error")
            .is_none());

        for half in [
            (Some(PathBuf::from("certificate-chain")), None),
            (None, Some(PathBuf::from("private-key"))),
        ] {
            let error = ProxyTls::resolve(half.0, half.1)
                .await
                .expect_err("half a pair is refused");
            assert!(error.to_string().contains("--tls-cert"), "{error}");
            assert!(error.to_string().contains("--tls-key"), "{error}");
        }
    }

    #[test]
    fn a_resolved_key_reports_itself_as_enabled() {
        assert!(!ClientAuth::none().is_enabled());
        assert!(!ClientAuth::resolve(None, None)
            .expect("no key is not an error")
            .is_enabled());
        assert!(ClientAuth::resolve(Some("k".to_string()), None)
            .expect("a key resolves")
            .is_enabled());
    }

    /// A key that cannot be read has to stop the process, not quietly leave the
    /// proxy open to whatever can reach it.
    #[test]
    fn an_unusable_key_is_an_error_rather_than_an_open_proxy() {
        ClientAuth::resolve(Some(" ".to_string()), None).expect_err("blank is not a key");
        ClientAuth::resolve(None, Some(PathBuf::from("no-such-key-file")))
            .expect_err("an unreadable file is not a key");
        ClientAuth::resolve(Some("k".to_string()), Some(PathBuf::from("f")))
            .expect_err("two sources is ambiguous");
    }

    /// The sticky header is documented as working whatever default the proxy
    /// was started with, and placement only reads a sticky label in affinity
    /// mode — so the header has to settle the mode for its own request.
    #[test]
    fn naming_a_node_asks_for_affinity_whatever_the_default_mode_is() {
        let pinned = OwnedRequest {
            model: None,
            sticky: Some("warm".to_string()),
        };
        let route = pinned.as_route(RouteMode::Throughput, MixedEngines::Refused);
        assert_eq!(route.mode, RouteMode::Affinity);
        assert_eq!(route.sticky, Some("warm"));

        // Without one, the proxy's configured default is what stands.
        let plain = OwnedRequest {
            model: None,
            sticky: None,
        };
        assert_eq!(
            plain
                .as_route(RouteMode::Throughput, MixedEngines::Refused)
                .mode,
            RouteMode::Throughput
        );
        assert_eq!(
            plain
                .as_route(RouteMode::Affinity, MixedEngines::Refused)
                .mode,
            RouteMode::Affinity
        );
    }

    /// The guard is what turns a dropped frame into something a blocking
    /// dispatch can see, so its own contract is worth pinning: dropping it
    /// fires, and holding it does not.
    #[test]
    fn dropping_the_guard_is_what_gives_up_on_the_request() {
        let cancel = Cancel::new();
        let guard = CancelOnDrop(cancel.clone());
        assert!(!cancel.is_cancelled(), "holding it must not give up");

        drop(guard);
        assert!(cancel.is_cancelled());
    }

    /// The streaming path moves its guard into the response body, so the body's
    /// lifetime is the request's. A body can be dropped without ever being
    /// polled — a client that leaves between the head and the first read — and
    /// a guard created *inside* the generator rather than moved into it would
    /// never exist to fire. This pins the distinction, which is invisible in
    /// the source.
    #[test]
    fn a_response_body_dropped_before_it_is_read_still_gives_up_on_the_request() {
        let cancel = Cancel::new();
        let guard = CancelOnDrop(cancel.clone());
        let stream = async_stream::stream! {
            let _client = guard;
            yield Ok::<Vec<u8>, String>(Vec::new());
        };

        assert!(!cancel.is_cancelled(), "building it must not give up");
        drop(stream);
        assert!(
            cancel.is_cancelled(),
            "a body nobody read left the node generating"
        );
    }

    /// A client leaving must not be logged, reported, or acted on as a node
    /// that failed: [`forward_error`] is what the access log reads placement
    /// back off, and 502 there would blame a healthy machine.
    #[test]
    fn a_request_whose_client_left_is_not_answered_as_a_node_failure() {
        let abandoned = forward_error(
            ForwardError::Cancelled {
                label: "node-a".to_string(),
            },
            None,
        );
        assert_eq!(abandoned.status().as_u16(), 499);
        assert_eq!(
            abandoned
                .headers()
                .get("x-camelid-fabric-node")
                .and_then(|value| value.to_str().ok()),
            Some("node-a")
        );

        // The control: a node that really did fail still answers 502.
        let failed = forward_error(
            ForwardError::Transport {
                label: "node-a".to_string(),
                detail: "connection reset".to_string(),
            },
            None,
        );
        assert_eq!(failed.status(), StatusCode::BAD_GATEWAY);
    }

    fn compare_request(body: Value) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/fabric/compare")
            .header("content-type", "application/json")
            .body(axum::body::Body::new(body.to_string()))
            .expect("valid request")
    }

    /// A caller's own mistake is theirs to fix; a node failing is the
    /// upstream's. Both used to answer 400, which sent a caller whose nodes
    /// were down to go and fix a request that was fine.
    #[tokio::test]
    async fn a_failed_comparison_says_whose_failure_it_was() {
        // Port 1 is closed, so both sides are observed as unreachable.
        let dead = Fabric::new(vec![
            NodeSpec::camelid("a", "127.0.0.1", 1),
            NodeSpec::camelid("b", "127.0.0.1", 1),
        ])
        .with_timeout(Duration::from_millis(200));
        let asking = |left: &str| serde_json::json!({ "left": left, "right": "b", "model": "m", "prompt": "q" });

        let upstream = proxy(dead.clone())
            .oneshot(compare_request(asking("a")))
            .await
            .expect("router answers");
        let (status, body) = read_json(upstream).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .expect("a message")
                .contains("a is not serving"),
            "{body}"
        );

        let typo = proxy(dead)
            .oneshot(compare_request(asking("nope")))
            .await
            .expect("router answers");
        let (status, body) = read_json(typo).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .expect("a message")
                .contains("no node named nope"),
            "{body}"
        );
    }

    /// Refused rather than clamped: a clamped temperature would be recorded as
    /// a plan the caller never asked about. The fabric is empty, so a request
    /// that got as far as running would have failed naming a node instead.
    #[tokio::test]
    async fn a_temperature_outside_the_documented_range_is_refused_before_any_work() {
        for temperature in [
            serde_json::json!(2.5),
            serde_json::json!(-1.0),
            serde_json::json!(1e39),
        ] {
            let response = proxy(Fabric::new(Vec::new()))
                .oneshot(compare_request(serde_json::json!({
                    "left": "a", "right": "b", "model": "m", "prompt": "q",
                    "temperature": temperature,
                })))
                .await
                .expect("router answers");
            let (status, body) = read_json(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{temperature}: {body}");
            let message = body["error"]["message"].as_str().expect("a message");
            assert!(message.contains("temperature"), "{temperature}: {message}");
        }
    }
}
