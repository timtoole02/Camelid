use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{BackendError, Result};
use crate::inference::LlamaInferenceSession;
use crate::tensor::{CpuTensor, Q4KRepack8Cell, RuntimeDType, TensorShape};

/// Wire version of the `serve-distributed` connect handshake.
///
/// Bump on any change to [`NodeIdentity`]'s meaning. Peers require equality, so an old
/// binary meeting a new one is refused at connect rather than producing wrong numbers
/// somewhere deep in a forward pass.
pub const HANDSHAKE_VERSION: u32 = 1;

const HANDSHAKE_MAGIC: u32 = 0xCA9E_0001;
const HANDSHAKE_IO_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_DISTRIBUTED_TOKEN_BYTES: usize = 4 * 1024;

/// A handshake is a few hundred bytes. The cap exists so a hostile or confused peer
/// cannot make the other side allocate on a length it chose.
const MAX_HANDSHAKE_BYTES: u32 = 64 * 1024;

fn engine_build_identity() -> String {
    let commit = crate::receipt::camelid_commit();
    if commit == "unknown" {
        return env!("CARGO_PKG_VERSION").to_string();
    }
    if crate::receipt::camelid_version().ends_with("-dirty") {
        format!("{commit}+dirty")
    } else {
        commit
    }
}

/// What a node asserts about itself when a distributed connection opens.
///
/// The pipeline ships raw activations between two processes that each hold half a model.
/// Nothing in the activation stream identifies the model, the build, or the split, so
/// without this exchange two peers running *different* weights or *different* code
/// connect happily and produce confident, wrong output. Every field below is one way
/// that can happen.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeIdentity {
    pub wire_version: u32,
    /// Engine build. Different code can mean different math, so peers must match.
    pub engine_version: String,
    /// Full-file SHA-256 of the GGUF. This is the model's identity: file length (what a
    /// weaker handshake might compare) is equal for any two same-size quantisations.
    pub model_sha256: String,
    pub total_layers: u32,
    pub hidden_size: u32,
    /// The layer range the worker owns, `[start, end)`.
    pub worker_layer_start: u32,
    pub worker_layer_end: u32,
    /// Host platform, e.g. `windows/x86_64`. **Reported, never enforced** — a coordinator
    /// and worker on different platforms are a supported configuration and have been
    /// measured producing byte-identical output. Carried so an operator (and a future
    /// receipt) can see what actually ran.
    pub platform: String,
    /// Presented by the coordinator when the worker requires one. Never logged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

impl std::fmt::Debug for NodeIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NodeIdentity")
            .field("wire_version", &self.wire_version)
            .field("engine_version", &self.engine_version)
            .field("model_sha256", &self.model_sha256)
            .field("total_layers", &self.total_layers)
            .field("hidden_size", &self.hidden_size)
            .field("worker_layer_start", &self.worker_layer_start)
            .field("worker_layer_end", &self.worker_layer_end)
            .field("platform", &self.platform)
            .field("token_present", &self.token.is_some())
            .finish()
    }
}

impl NodeIdentity {
    /// Build the identity for a node holding `worker_layers` of `model`.
    ///
    /// Hashing is deliberately uncached because this digest authenticates the
    /// model identity used by the distributed handshake. A writable local
    /// performance cache is not authority for that decision.
    pub fn for_model(
        model: &Path,
        total_layers: u32,
        hidden_size: u32,
        worker_layers: std::ops::Range<u32>,
    ) -> Result<Self> {
        let model_sha256 = crate::receipt::sha256_file_hex(model).map_err(|err| {
            BackendError::RuntimeShapeMismatch(format!(
                "could not hash {} for the distributed handshake: {err}",
                model.display()
            ))
        })?;
        Ok(Self {
            wire_version: HANDSHAKE_VERSION,
            engine_version: engine_build_identity(),
            model_sha256,
            total_layers,
            hidden_size,
            worker_layer_start: worker_layers.start,
            worker_layer_end: worker_layers.end,
            platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
            token: None,
        })
    }

    pub fn with_token(mut self, token: Option<String>) -> Self {
        self.token = token;
        self
    }

    /// The first field on which `self` (the worker's own view) disagrees with the
    /// `peer` that just connected, named exactly.
    ///
    /// `platform` is deliberately absent: see the field's documentation.
    fn first_mismatch(&self, peer: &Self) -> Option<String> {
        fn differs<T: PartialEq + std::fmt::Display>(
            field: &str,
            worker: &T,
            coordinator: &T,
        ) -> Option<String> {
            (worker != coordinator).then(|| {
                format!("{field} mismatch: worker has {worker}, coordinator has {coordinator}")
            })
        }
        differs("wire_version", &self.wire_version, &peer.wire_version)
            .or_else(|| differs("engine_version", &self.engine_version, &peer.engine_version))
            .or_else(|| differs("model_sha256", &self.model_sha256, &peer.model_sha256))
            .or_else(|| differs("total_layers", &self.total_layers, &peer.total_layers))
            .or_else(|| differs("hidden_size", &self.hidden_size, &peer.hidden_size))
            .or_else(|| {
                differs(
                    "worker_layer_start",
                    &self.worker_layer_start,
                    &peer.worker_layer_start,
                )
            })
            .or_else(|| {
                differs(
                    "worker_layer_end",
                    &self.worker_layer_end,
                    &peer.worker_layer_end,
                )
            })
    }

    /// Whether `peer` may drive this node, and why not when it may not.
    ///
    /// `required_token` is the worker's configured secret. When set, a coordinator that
    /// presents nothing or presents the wrong value is refused; the comparison is
    /// length-independent and constant time so a rejected peer learns nothing about the
    /// secret from how long the refusal took.
    pub fn admit(
        &self,
        peer: &Self,
        required_token: Option<&str>,
    ) -> std::result::Result<(), String> {
        if let Some(expected) = required_token {
            match peer.token.as_deref() {
                Some(presented) if constant_time_eq(expected.as_bytes(), presented.as_bytes()) => {}
                Some(_) => return Err("authentication failed: token rejected".to_string()),
                None => {
                    return Err(
                        "authentication failed: this worker requires a token and the \
                         coordinator presented none"
                            .to_string(),
                    )
                }
            }
        }
        match self.first_mismatch(peer) {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }
}

pub fn validate_distributed_token(
    token: Option<String>,
) -> std::result::Result<Option<String>, String> {
    let Some(token) = token else {
        return Ok(None);
    };
    let token = token.trim().to_string();
    if token.is_empty() {
        return Err("distributed token must not be empty".to_string());
    }
    if token.len() > MAX_DISTRIBUTED_TOKEN_BYTES {
        return Err(format!(
            "distributed token exceeds the {MAX_DISTRIBUTED_TOKEN_BYTES}-byte limit"
        ));
    }
    if token.chars().any(char::is_control) {
        return Err("distributed token must not contain control characters".to_string());
    }
    Ok(Some(token))
}

pub fn resolve_distributed_token(
    token: Option<String>,
    token_file: Option<&Path>,
) -> std::result::Result<Option<String>, String> {
    if token.is_some() && token_file.is_some() {
        return Err(
            "--distributed-token and --distributed-token-file are mutually exclusive".to_string(),
        );
    }
    let token = match (token, token_file) {
        (Some(token), None) => Some(token),
        (None, Some(path)) => Some(std::fs::read_to_string(path).map_err(|err| {
            format!(
                "could not read distributed token file {}: {err}",
                path.display()
            )
        })?),
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!("conflict handled above"),
    };
    validate_distributed_token(token)
}

/// Compares without an early exit and without leaking the secret's length.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    // Fold length into the result instead of returning early on it.
    let mut diff = a.len() ^ b.len();
    for i in 0..MAX_DISTRIBUTED_TOKEN_BYTES {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= usize::from(x ^ y);
    }
    diff == 0
}

/// The worker's answer to a connect attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub accepted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<NodeIdentity>,
}

fn write_frame<W: Write, T: Serialize>(writer: &mut W, value: &T) -> std::io::Result<()> {
    let body = serde_json::to_vec(value)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
    let len = u32::try_from(body.len())
        .ok()
        .filter(|n| *n <= MAX_HANDSHAKE_BYTES);
    let Some(len) = len else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "handshake frame exceeds the maximum size",
        ));
    };
    writer.write_all(&HANDSHAKE_MAGIC.to_le_bytes())?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&body)?;
    writer.flush()
}

fn read_frame<R: Read, T: for<'de> Deserialize<'de>>(reader: &mut R) -> std::io::Result<T> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if u32::from_le_bytes(magic) != HANDSHAKE_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a Camelid distributed handshake",
        ));
    }
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes);
    if len > MAX_HANDSHAKE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "handshake frame exceeds the maximum size",
        ));
    }
    let mut body = vec![0u8; len as usize];
    reader.read_exact(&mut body)?;
    serde_json::from_slice(&body)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
}

fn set_handshake_timeouts(stream: &TcpStream, timeout: Option<Duration>) -> std::io::Result<()> {
    stream.set_read_timeout(timeout)?;
    stream.set_write_timeout(timeout)
}

/// Which whole-model tensors a node loads in the `serve-distributed` pipeline.
///
/// Ownership here is a property of the **role**, not of the layer range. A worker's
/// [`LlamaInferenceSession::forward_worker_layers`] returns a bare hidden state — it
/// applies neither `output_norm` nor the output projection — so the coordinator always
/// finalizes the forward pass and needs both ends of the model while holding the prefix
/// shard supported by this transport.
///
/// [`crate::inference::LlamaLoadedWeights::load`] instead derives ownership *positionally*,
/// for a generic pipeline whose LAST stage emits logits. Using that rule here gives a
/// `0..k` coordinator no output projection, and every request fails with
/// `runtime shape mismatch: ... requires rank-2 weight token_embd.weight, got [0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineRole {
    /// Runs the HTTP API, owns its layer shard plus the embedding and output head.
    Coordinator,
    /// Owns its layer shard only.
    Worker,
}

impl PipelineRole {
    /// `(load_embedding, load_output)` for
    /// [`crate::inference::LlamaLoadedWeights::load_distributed`].
    pub const fn tensor_ownership(self) -> (bool, bool) {
        match self {
            Self::Coordinator => (true, true),
            Self::Worker => (false, false),
        }
    }

    pub fn validate_layer_range(
        self,
        layer_start: usize,
        layer_end: usize,
        total_layers: usize,
    ) -> Result<()> {
        if total_layers < 2 || layer_start >= layer_end || layer_end > total_layers {
            return Err(BackendError::InvalidModelMetadata(format!(
                "distributed layer range {layer_start}..{layer_end} cannot partition model layers 0..{total_layers}"
            )));
        }
        match self {
            Self::Coordinator if layer_start != 0 || layer_end == total_layers => {
                Err(BackendError::InvalidModelMetadata(format!(
                    "distributed coordinator must own a non-empty prefix 0..SPLIT below layer {total_layers}, got {layer_start}..{layer_end}"
                )))
            }
            Self::Worker if layer_start == 0 || layer_end != total_layers => {
                Err(BackendError::InvalidModelMetadata(format!(
                    "distributed worker must own a non-empty suffix SPLIT..{total_layers}, got {layer_start}..{layer_end}"
                )))
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod pipeline_role_tests {
    use super::PipelineRole;

    /// The rule this transport needs is deliberately *not* the positional one. If anyone
    /// "simplifies" `tensor_ownership` back to deriving from the layer range, a `0..k`
    /// coordinator silently loses the output head again and every request 503s.
    #[test]
    fn the_prefix_coordinator_owns_both_ends() {
        assert_eq!(PipelineRole::Coordinator.tensor_ownership(), (true, true));
        assert_eq!(PipelineRole::Worker.tensor_ownership(), (false, false));

        // The positional rule `LlamaLoadedWeights::load` applies, restated here: the first
        // shard owns the embedding, the last shard owns the output head.
        let total_layers = 16;
        let head_shard = 0..8;
        let positional = (head_shard.start == 0, head_shard.end >= total_layers);
        assert_eq!(positional, (true, false));
        assert_ne!(
            PipelineRole::Coordinator.tensor_ownership(),
            positional,
            "a head-shard coordinator must not inherit the positional rule"
        );
    }

    #[test]
    fn roles_accept_only_a_complete_prefix_suffix_partition() {
        PipelineRole::Coordinator
            .validate_layer_range(0, 8, 16)
            .unwrap();
        PipelineRole::Worker
            .validate_layer_range(8, 16, 16)
            .unwrap();

        for (role, start, end) in [
            (PipelineRole::Coordinator, 4, 8),
            (PipelineRole::Coordinator, 0, 16),
            (PipelineRole::Worker, 0, 8),
            (PipelineRole::Worker, 8, 15),
            (PipelineRole::Worker, 8, 17),
        ] {
            role.validate_layer_range(start, end, 16)
                .expect_err("gapped, overlapping, or out-of-bounds partitions must fail");
        }
    }
}

#[cfg(test)]
mod handshake_tests {
    use super::*;

    fn connect_to_response(response: HandshakeResponse) -> std::io::Result<DistributedClient> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _: NodeIdentity = read_frame(&mut stream).unwrap();
            write_frame(&mut stream, &response).unwrap();
        });
        let result = DistributedClient::connect(&addr.to_string(), &identity());
        server.join().unwrap();
        result
    }

    fn identity() -> NodeIdentity {
        NodeIdentity {
            wire_version: HANDSHAKE_VERSION,
            engine_version: "0.5.2".to_string(),
            model_sha256: "a".repeat(64),
            total_layers: 16,
            hidden_size: 2048,
            worker_layer_start: 8,
            worker_layer_end: 16,
            platform: "linux/x86_64".to_string(),
            token: None,
        }
    }

    #[test]
    fn a_matching_peer_is_admitted() {
        assert!(identity().admit(&identity(), None).is_ok());
    }

    /// Each of these is a way two nodes can connect and compute different mathematics.
    /// The refusal has to name the field, or an operator is left guessing which of two
    /// machines is the wrong one.
    #[test]
    fn every_identity_field_is_refused_by_name() {
        let mut cases: Vec<(&str, NodeIdentity)> = Vec::new();
        let mut peer = identity();
        peer.wire_version += 1;
        cases.push(("wire_version", peer));
        let mut peer = identity();
        peer.engine_version = "0.5.1".to_string();
        cases.push(("engine_version", peer));
        let mut peer = identity();
        peer.model_sha256 = "b".repeat(64);
        cases.push(("model_sha256", peer));
        let mut peer = identity();
        peer.total_layers = 32;
        cases.push(("total_layers", peer));
        let mut peer = identity();
        peer.hidden_size = 4096;
        cases.push(("hidden_size", peer));
        let mut peer = identity();
        peer.worker_layer_start = 7;
        cases.push(("worker_layer_start", peer));
        let mut peer = identity();
        peer.worker_layer_end = 15;
        cases.push(("worker_layer_end", peer));

        for (field, peer) in cases {
            let err = identity()
                .admit(&peer, None)
                .expect_err("a peer differing in one field must be refused");
            assert!(
                err.contains(field),
                "refusal must name the field; {field} produced: {err}"
            );
        }
    }

    /// A same-size GGUF with different contents is the case a length comparison cannot
    /// see, and it is the one that silently produces wrong output.
    #[test]
    fn a_different_model_of_the_same_size_is_refused() {
        let mut peer = identity();
        peer.model_sha256 = "c".repeat(64);
        let err = identity().admit(&peer, None).unwrap_err();
        assert!(err.contains("model_sha256"), "{err}");
    }

    /// Platform difference is a supported configuration: a Windows x86_64 coordinator and
    /// an ARM64 macOS worker have been measured producing byte-identical output. Refusing
    /// it would break a working cluster.
    #[test]
    fn a_different_platform_is_reported_not_refused() {
        let mut peer = identity();
        peer.platform = "macos/aarch64".to_string();
        assert!(identity().admit(&peer, None).is_ok());
    }

    #[test]
    fn a_worker_with_a_token_refuses_a_coordinator_without_one() {
        let err = identity().admit(&identity(), Some("s3cret")).unwrap_err();
        assert!(err.contains("requires a token"), "{err}");
    }

    #[test]
    fn a_wrong_token_is_refused() {
        let peer = identity().with_token(Some("wrong".to_string()));
        let err = identity().admit(&peer, Some("s3cret")).unwrap_err();
        assert!(err.contains("token rejected"), "{err}");
    }

    #[test]
    fn the_right_token_is_admitted() {
        let peer = identity().with_token(Some("s3cret".to_string()));
        assert!(identity().admit(&peer, Some("s3cret")).is_ok());
    }

    #[test]
    fn distributed_tokens_are_non_empty_bounded_and_control_free() {
        assert_eq!(validate_distributed_token(None).unwrap(), None);
        assert_eq!(
            validate_distributed_token(Some("  s3cret  ".to_string())).unwrap(),
            Some("s3cret".to_string())
        );
        assert!(validate_distributed_token(Some("  ".to_string())).is_err());
        assert!(validate_distributed_token(Some("line\nbreak".to_string())).is_err());
        assert!(
            validate_distributed_token(Some("x".repeat(MAX_DISTRIBUTED_TOKEN_BYTES + 1))).is_err()
        );

        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "  from-file\n").unwrap();
        assert_eq!(
            resolve_distributed_token(None, Some(file.path())).unwrap(),
            Some("from-file".to_string())
        );
        assert!(resolve_distributed_token(Some("inline".to_string()), Some(file.path())).is_err());
    }

    #[test]
    fn node_identity_debug_redacts_the_token() {
        let identity = identity().with_token(Some("do-not-print-me".to_string()));
        let debug = format!("{identity:?}");
        assert!(!debug.contains("do-not-print-me"));
        assert!(debug.contains("token_present: true"));
    }

    #[test]
    fn model_identity_uses_embedded_build_provenance() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let identity = NodeIdentity::for_model(file.path(), 16, 2048, 8..16).unwrap();
        assert_eq!(identity.engine_version, engine_build_identity());
        let commit = crate::receipt::camelid_commit();
        if commit != "unknown" {
            assert!(identity.engine_version.starts_with(&commit));
        }
    }

    #[test]
    fn a_stalled_handshake_read_times_out() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        set_handshake_timeouts(&server, Some(Duration::from_millis(20))).unwrap();

        let err = read_frame::<_, NodeIdentity>(&mut server).unwrap_err();
        assert!(matches!(
            err.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
        ));
        drop(client);
    }

    #[test]
    fn coordinator_requires_the_accepted_workers_identity() {
        let err = match connect_to_response(HandshakeResponse {
            accepted: true,
            refusal: None,
            worker: None,
        }) {
            Ok(_) => panic!("accepted response without worker identity must fail"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        let mut worker = identity();
        worker.model_sha256 = "b".repeat(64);
        let err = match connect_to_response(HandshakeResponse {
            accepted: true,
            refusal: None,
            worker: Some(worker),
        }) {
            Ok(_) => panic!("accepted response with mismatched worker identity must fail"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("model_sha256"), "{err}");
    }

    /// A token check that ran before the identity check would let a stranger probe which
    /// models a worker holds. Identity is only compared once the peer has authenticated.
    #[test]
    fn authentication_is_decided_before_identity_is_revealed() {
        let mut peer = identity();
        peer.model_sha256 = "d".repeat(64);
        let err = identity().admit(&peer, Some("s3cret")).unwrap_err();
        assert!(
            err.contains("authentication failed"),
            "an unauthenticated peer must not learn about model identity, got: {err}"
        );
    }

    #[test]
    fn constant_time_eq_matches_only_identical_bytes() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(!constant_time_eq(b"", b"a"));
        assert!(constant_time_eq(b"", b""));

        let mut length_wrapping_candidate = b"abc".to_vec();
        length_wrapping_candidate.extend([0; 256]);
        assert!(!constant_time_eq(b"abc", &length_wrapping_candidate));
    }

    #[test]
    fn a_frame_round_trips() {
        let mut buf = Vec::new();
        write_frame(&mut buf, &identity()).unwrap();
        let back: NodeIdentity = read_frame(&mut buf.as_slice()).unwrap();
        assert_eq!(back, identity());
    }

    #[test]
    fn a_frame_without_the_magic_is_rejected() {
        let buf = vec![0u8; 8];
        let err = read_frame::<_, NodeIdentity>(&mut buf.as_slice()).unwrap_err();
        assert!(err
            .to_string()
            .contains("not a Camelid distributed handshake"));
    }

    /// The length prefix is attacker-chosen, so it is bounded before it is allocated.
    #[test]
    fn an_oversized_declared_length_is_rejected_before_allocating() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&HANDSHAKE_MAGIC.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        let err = read_frame::<_, NodeIdentity>(&mut buf.as_slice()).unwrap_err();
        assert!(err.to_string().contains("maximum size"), "{err}");
    }

    #[test]
    fn the_bind_policy_matches_the_http_servers_rule() {
        let loopback: SocketAddr = "127.0.0.1:5005".parse().unwrap();
        let public: SocketAddr = "0.0.0.0:5005".parse().unwrap();

        assert!(check_worker_bind_policy(&loopback, false, false).is_ok());
        assert!(check_worker_bind_policy(&public, true, false).is_ok());
        assert!(check_worker_bind_policy(&public, false, true).is_ok());

        let err = check_worker_bind_policy(&public, false, false).unwrap_err();
        assert!(
            err.contains("refusing unauthenticated non-loopback"),
            "{err}"
        );
        assert!(err.contains("CAMELID_DISTRIBUTED_TOKEN"), "{err}");
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DistributedHeader {
    pub magic: u32,
    pub is_prefill: u32, // 0 = decode, 1 = prefill
    pub seq_len: u32,
    pub position: u32,
}

impl DistributedHeader {
    pub const MAGIC: u32 = 0xCA9E111D;

    pub fn to_bytes(self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.is_prefill.to_le_bytes());
        buf[8..12].copy_from_slice(&self.seq_len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.position.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: [u8; 16]) -> Self {
        Self {
            magic: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            is_prefill: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            seq_len: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            position: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
        }
    }
}

pub fn serialize_tensor<W: Write>(writer: &mut W, tensor: &CpuTensor) -> std::io::Result<()> {
    let dims_len = tensor.shape.dims.len() as u32;
    writer.write_all(&dims_len.to_le_bytes())?;
    for &dim in &tensor.shape.dims {
        let dim_val = dim as u32;
        writer.write_all(&dim_val.to_le_bytes())?;
    }
    let data_len = tensor.data.len() as u32;
    writer.write_all(&data_len.to_le_bytes())?;

    // Write data as raw bytes (Apple Silicon to Apple Silicon is safe)
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            tensor.data.as_ptr() as *const u8,
            tensor.data.len() * std::mem::size_of::<f32>(),
        )
    };
    writer.write_all(byte_slice)?;
    Ok(())
}

pub fn deserialize_tensor<R: Read>(reader: &mut R, name: String) -> std::io::Result<CpuTensor> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let dims_len = u32::from_le_bytes(buf) as usize;
    let mut dims = Vec::with_capacity(dims_len);
    for _ in 0..dims_len {
        reader.read_exact(&mut buf)?;
        dims.push(u32::from_le_bytes(buf) as usize);
    }
    reader.read_exact(&mut buf)?;
    let data_len = u32::from_le_bytes(buf) as usize;
    let mut data = vec![0.0f32; data_len];

    // Read data as raw bytes
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            data_len * std::mem::size_of::<f32>(),
        )
    };
    reader.read_exact(byte_slice)?;

    Ok(CpuTensor {
        name,
        shape: TensorShape { dims },
        dtype: RuntimeDType::F32,
        source_type: None,
        q8_0_blocks: None,
        q8_0_shared_blocks: None,
        q8_0_packed_rows4_4x4: None,
        q8_0_packed_rows4_4x8: None,
        q8_0_runtime_storage: None,
        q8_0_file_backing: None,
        q8_0_wire_mmap: None,
        q8_0_wire_pages: None,
        kquant_wire_pages: None,
        q8_0_split_file_backing: None,
        q4_0_file_backing: None,
        q4_k_wire_bytes: None,
        q4_k_repack8: Q4KRepack8Cell::default(),
        q5_k_wire_bytes: None,
        q6_k_wire_bytes: None,
        q2_k_wire_bytes: None,
        q3_k_wire_bytes: None,
        tq2_0_wire_bytes: None,
        iq4_xs_wire_bytes: None,
        data,
    })
}

pub struct DistributedClient {
    stream: Mutex<TcpStream>,
    addr: String,
}

impl DistributedClient {
    /// Open a connection and complete the identity handshake.
    ///
    /// A worker that disagrees about the model, the build, or the split refuses here, so a
    /// misconfigured pair fails at startup with a named field instead of producing plausible
    /// wrong output for the life of the process.
    pub fn connect(addr: &str, identity: &NodeIdentity) -> std::io::Result<Self> {
        let mut stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        set_handshake_timeouts(&stream, Some(HANDSHAKE_IO_TIMEOUT))?;
        write_frame(&mut stream, identity)?;
        let response: HandshakeResponse = read_frame(&mut stream)?;
        if !response.accepted {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!(
                    "worker {addr} refused the distributed handshake: {}",
                    response.refusal.as_deref().unwrap_or("no reason given")
                ),
            ));
        }
        let worker = response.worker.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "worker accepted the distributed handshake without returning its identity",
            )
        })?;
        if let Some(reason) = identity.first_mismatch(worker) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("worker accepted with a mismatched identity: {reason}"),
            ));
        }
        tracing::info!(
            worker_platform = %worker.platform,
            worker_layers = format!("{}..{}", worker.worker_layer_start, worker.worker_layer_end),
            "distributed handshake accepted"
        );
        set_handshake_timeouts(&stream, None)?;
        Ok(Self {
            stream: Mutex::new(stream),
            addr: addr.to_string(),
        })
    }

    pub fn forward_to_worker(
        &self,
        hidden: &CpuTensor,
        is_prefill: bool,
        seq_len: usize,
        position: usize,
    ) -> Result<CpuTensor> {
        // Worker telemetry wraps the real TCP roundtrip: active when the
        // activation ships out, idle when the response lands, error on any
        // wire failure.
        crate::telemetry::emit(crate::telemetry::Event::WorkerNodeActive {
            node: self.addr.clone(),
            detail: Some(if is_prefill {
                format!("prefill seq_len {seq_len} @ position {position}")
            } else {
                format!("decode @ position {position}")
            }),
        });
        let result = self.forward_to_worker_inner(hidden, is_prefill, seq_len, position);
        match &result {
            Ok(_) => crate::telemetry::emit(crate::telemetry::Event::WorkerNodeIdle {
                node: self.addr.clone(),
            }),
            Err(err) => crate::telemetry::emit(crate::telemetry::Event::WorkerNodeError {
                node: self.addr.clone(),
                error: err.to_string(),
            }),
        }
        result
    }

    fn forward_to_worker_inner(
        &self,
        hidden: &CpuTensor,
        is_prefill: bool,
        seq_len: usize,
        position: usize,
    ) -> Result<CpuTensor> {
        let mut stream = self.stream.lock().map_err(|_| {
            BackendError::RuntimeShapeMismatch("Failed to lock TCP stream mutex".to_string())
        })?;

        let header = DistributedHeader {
            magic: DistributedHeader::MAGIC,
            is_prefill: if is_prefill { 1 } else { 0 },
            seq_len: seq_len as u32,
            position: position as u32,
        };

        // Send header
        stream
            .write_all(&header.to_bytes())
            .map_err(|source| BackendError::Io {
                path: PathBuf::from("distributed_tcp_client_write"),
                source,
            })?;

        // Send tensor
        serialize_tensor(&mut *stream, hidden).map_err(|source| BackendError::Io {
            path: PathBuf::from("distributed_tcp_client_write_tensor"),
            source,
        })?;

        stream.flush().map_err(|source| BackendError::Io {
            path: PathBuf::from("distributed_tcp_client_flush"),
            source,
        })?;

        // Read response tensor
        let response = deserialize_tensor(&mut *stream, "worker_response_tensor".to_string())
            .map_err(|source| BackendError::Io {
                path: PathBuf::from("distributed_tcp_client_read_response"),
                source,
            })?;

        Ok(response)
    }
}

pub static DISTRIBUTED_CLIENT: OnceLock<DistributedClient> = OnceLock::new();
pub static DISTRIBUTED_RANGE: OnceLock<(usize, usize)> = OnceLock::new();

/// Refuse an unauthenticated worker listener that faces the network.
///
/// This mirrors the rule the HTTP server already applies in [`crate::api::server`]:
/// loopback stays frictionless, and exposing the port requires either a token or an
/// explicit acknowledgement. The worker port deserves at least that much — it accepts raw
/// activations and spends this machine's CPU on them, with no request logging and no
/// per-request authorization behind it.
pub fn check_worker_bind_policy(
    addr: &SocketAddr,
    has_token: bool,
    allow_unauthenticated_remote: bool,
) -> std::result::Result<(), String> {
    if addr.ip().is_loopback() || has_token || allow_unauthenticated_remote {
        return Ok(());
    }
    Err(format!(
        "refusing unauthenticated non-loopback distributed worker listener {addr}; set \
         --distributed-token/CAMELID_DISTRIBUTED_TOKEN or explicitly acknowledge the risk \
         with --allow-unauthenticated-remote"
    ))
}

pub fn run_worker_loop(
    addr: &str,
    session: LlamaInferenceSession,
    identity: NodeIdentity,
    required_token: Option<String>,
    allow_unauthenticated_remote: bool,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    if let Err(reason) = check_worker_bind_policy(
        &bound,
        required_token.is_some(),
        allow_unauthenticated_remote,
    ) {
        anyhow::bail!(reason);
    }
    run_worker_loop_on_listener(listener, session, identity, required_token)
}

/// Serve the worker protocol on an already-bound listener.
///
/// Callers that must know the port before the loop starts — tests binding
/// `127.0.0.1:0` for an ephemeral port — bind themselves and hand the listener
/// over, so there is no window in which the address is unbound and no
/// hard-coded port to collide with.
pub fn run_worker_loop_on_listener(
    listener: TcpListener,
    mut session: LlamaInferenceSession,
    identity: NodeIdentity,
    required_token: Option<String>,
) -> anyhow::Result<()> {
    tracing::info!(addr = ?listener.local_addr(), "Distributed Worker TCP server listening");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "Worker failed to accept connection");
                continue;
            }
        };

        let _ = stream.set_nodelay(true);
        if let Err(e) = set_handshake_timeouts(&stream, Some(HANDSHAKE_IO_TIMEOUT)) {
            tracing::warn!(error = %e, "could not arm distributed handshake timeout");
            continue;
        }

        // Admit before a single activation is read: a peer that disagrees about the model,
        // the build or the split can only produce wrong numbers, and one that cannot
        // authenticate has no business spending this machine's CPU.
        let peer: NodeIdentity = match read_frame(&mut stream) {
            Ok(peer) => peer,
            Err(e) => {
                tracing::warn!(error = %e, "rejected a connection that did not open with a handshake");
                continue;
            }
        };
        if let Err(reason) = identity.admit(&peer, required_token.as_deref()) {
            tracing::warn!(%reason, "refused a distributed coordinator");
            let _ = write_frame(
                &mut stream,
                &HandshakeResponse {
                    accepted: false,
                    refusal: Some(reason),
                    worker: None,
                },
            );
            continue;
        }
        if let Err(e) = write_frame(
            &mut stream,
            &HandshakeResponse {
                accepted: true,
                refusal: None,
                // Never echo a credential back, whatever this node was configured with.
                worker: Some(NodeIdentity {
                    token: None,
                    ..identity.clone()
                }),
            },
        ) {
            tracing::error!(error = %e, "failed to acknowledge the handshake");
            continue;
        }
        if let Err(e) = set_handshake_timeouts(&stream, None) {
            tracing::error!(error = %e, "could not clear distributed handshake timeout");
            continue;
        }
        tracing::info!(
            coordinator_platform = %peer.platform,
            "Worker accepted connection from coordinator"
        );

        loop {
            let mut header_buf = [0u8; 16];
            if let Err(e) = stream.read_exact(&mut header_buf) {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    tracing::info!("Coordinator closed connection");
                } else {
                    tracing::error!(error = %e, "Error reading header from stream");
                }
                break;
            }

            let header = DistributedHeader::from_bytes(header_buf);
            if header.magic != DistributedHeader::MAGIC {
                tracing::error!(magic = ?header.magic, "Received invalid magic header");
                break;
            }

            let input_tensor =
                match deserialize_tensor(&mut stream, "coordinator_tensor".to_string()) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(error = %e, "Failed to deserialize input tensor");
                        break;
                    }
                };

            let is_prefill = header.is_prefill == 1;

            let output_tensor = match session.forward_worker_layers(
                input_tensor,
                is_prefill,
                header.seq_len as usize,
                header.position as usize,
            ) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to run worker forward layers");
                    break;
                }
            };

            if let Err(e) = serialize_tensor(&mut stream, &output_tensor) {
                tracing::error!(error = %e, "Failed to serialize response tensor");
                break;
            }
            let _ = stream.flush();
        }
    }
    Ok(())
}

pub fn run_network_benchmark_worker(addr: &str) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr)?;
    run_network_benchmark_worker_on_listener(listener)
}

/// Benchmark counterpart to [`run_worker_loop_on_listener`]: serve on a
/// listener the caller already bound.
pub fn run_network_benchmark_worker_on_listener(listener: TcpListener) -> anyhow::Result<()> {
    tracing::info!(addr = ?listener.local_addr(), "Network benchmark worker TCP server listening");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "Worker failed to accept benchmark connection");
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        tracing::info!("Worker accepted benchmark connection from coordinator");

        loop {
            let mut header = [0u8; 16]; // [magic (4B), test_type (4B), count (4B), size (4B)]
            if let Err(e) = stream.read_exact(&mut header) {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    tracing::info!("Coordinator closed benchmark connection");
                } else {
                    tracing::error!(error = %e, "Error reading benchmark header");
                }
                break;
            }

            let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
            let test_type = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
            let count = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
            let size =
                u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;

            if magic != DistributedHeader::MAGIC {
                tracing::error!(magic = ?magic, "Received invalid magic benchmark header");
                break;
            }

            match test_type {
                0 => {
                    tracing::info!("Received termination command. Ending benchmark session.");
                    break;
                }
                1 => {
                    // Latency Test
                    tracing::info!(
                        count = count,
                        size = size,
                        "Starting Latency Test loop as receiver"
                    );
                    let mut buf = vec![0u8; size];
                    for _ in 0..count {
                        stream.read_exact(&mut buf)?;
                        stream.write_all(&buf)?;
                        stream.flush()?;
                    }
                    tracing::info!("Latency Test loop completed");
                }
                2 => {
                    // Bandwidth Test
                    let total_bytes = count * 1024 * 1024; // count is in MB
                    tracing::info!(
                        total_mb = count,
                        chunk_size = size,
                        "Starting Bandwidth Test loop as receiver"
                    );
                    let mut buf = vec![0u8; size];
                    let mut bytes_received = 0;
                    while bytes_received < total_bytes {
                        let to_read = std::cmp::min(size, total_bytes - bytes_received);
                        stream.read_exact(&mut buf[..to_read])?;
                        bytes_received += to_read;
                    }
                    // Send 1-byte ACK
                    stream.write_all(&[1u8])?;
                    stream.flush()?;
                    tracing::info!(
                        bytes_received = bytes_received,
                        "Bandwidth Test loop completed"
                    );
                }
                _ => {
                    tracing::error!(test_type = test_type, "Received unknown test type");
                    break;
                }
            }
        }
    }
    Ok(())
}

pub fn run_network_benchmark_coordinator(
    addr: &str,
    ping_count: usize,
    payload_size: usize,
    bandwidth_mb: usize,
) -> anyhow::Result<()> {
    tracing::info!(addr = %addr, "Connecting to benchmark worker...");
    let mut stream = TcpStream::connect(addr)?;
    stream.set_nodelay(true)?;
    tracing::info!("Connected to benchmark worker successfully!");

    // --- Latency Test ---
    println!("\n=== Starting TCP Latency Test ===");
    println!("Payload size: {} bytes", payload_size);
    println!("Ping count: {}", ping_count);

    let header = [
        &DistributedHeader::MAGIC.to_le_bytes()[..],
        &1u32.to_le_bytes()[..], // test_type = 1
        &(ping_count as u32).to_le_bytes()[..],
        &(payload_size as u32).to_le_bytes()[..],
    ]
    .concat();

    stream.write_all(&header)?;
    stream.flush()?;

    let payload = vec![0u8; payload_size];
    let mut response = vec![0u8; payload_size];
    let mut durations_us = Vec::with_capacity(ping_count);

    for _ in 0..ping_count {
        let started = std::time::Instant::now();
        stream.write_all(&payload)?;
        stream.flush()?;
        stream.read_exact(&mut response)?;
        let elapsed = started.elapsed().as_secs_f64() * 1_000_000.0; // in microseconds
        durations_us.push(elapsed);
    }

    let min_us = durations_us.iter().copied().fold(f64::INFINITY, f64::min);
    let max_us = durations_us.iter().copied().fold(0.0, f64::max);
    let avg_us = durations_us.iter().copied().sum::<f64>() / ping_count as f64;

    println!("--- Latency Results ---");
    println!("Round-Trip Time (RTT):");
    println!("  Min RTT: {:.2} μs", min_us);
    println!("  Avg RTT: {:.2} μs", avg_us);
    println!("  Max RTT: {:.2} μs", max_us);
    println!("One-Way Latency (RTT / 2):");
    println!("  Min Latency: {:.2} μs", min_us / 2.0);
    println!("  Avg Latency: {:.2} μs", avg_us / 2.0);
    println!("  Max Latency: {:.2} μs", max_us / 2.0);

    // --- Bandwidth Test ---
    println!("\n=== Starting TCP Bandwidth Test ===");
    let total_mb = bandwidth_mb;
    let chunk_size = 65536; // 64KB chunks
    println!("Total data to send: {} MB", total_mb);
    println!("Chunk size: {} bytes", chunk_size);

    let header = [
        &DistributedHeader::MAGIC.to_le_bytes()[..],
        &2u32.to_le_bytes()[..], // test_type = 2
        &(total_mb as u32).to_le_bytes()[..],
        &(chunk_size as u32).to_le_bytes()[..],
    ]
    .concat();

    stream.write_all(&header)?;
    stream.flush()?;

    let bw_payload = vec![0u8; chunk_size];
    let total_bytes = total_mb * 1024 * 1024;
    let mut bytes_sent = 0;

    let started = std::time::Instant::now();
    while bytes_sent < total_bytes {
        let to_send = std::cmp::min(chunk_size, total_bytes - bytes_sent);
        stream.write_all(&bw_payload[..to_send])?;
        bytes_sent += to_send;
    }
    stream.flush()?;

    // Await ACK
    let mut ack = [0u8; 1];
    stream.read_exact(&mut ack)?;
    let duration = started.elapsed();
    let duration_secs = duration.as_secs_f64();

    let mb_sent = bytes_sent as f64 / (1024.0 * 1024.0);
    let bandwidth_mb_s = mb_sent / duration_secs;
    let bandwidth_gbps = (bytes_sent as f64 * 8.0) / (duration_secs * 1_000_000_000.0);

    println!("--- Bandwidth Results ---");
    println!(
        "  Total Transferred: {:.2} MB in {:.4} seconds",
        mb_sent, duration_secs
    );
    println!(
        "  Throughput: {:.2} MB/s ({:.2} Gbps)",
        bandwidth_mb_s, bandwidth_gbps
    );

    // --- Terminate Session ---
    let header = [
        &DistributedHeader::MAGIC.to_le_bytes()[..],
        &0u32.to_le_bytes()[..], // test_type = 0 (Terminate)
        &0u32.to_le_bytes()[..],
        &0u32.to_le_bytes()[..],
    ]
    .concat();
    let _ = stream.write_all(&header);
    let _ = stream.flush();

    println!("\n=== Benchmark Completed ===");
    Ok(())
}
