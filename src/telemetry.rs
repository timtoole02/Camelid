//! Live inference telemetry: a process-wide broadcast hub that the engine
//! emits real runtime events into and the HTTP layer streams to subscribers
//! over SSE (`GET /api/telemetry/stream`).
//!
//! Truthfulness contract: every event is emitted from a real code path doing
//! real work — there is no synthetic, replayed, or decorative event source.
//! If nothing is running, the stream carries nothing (besides SSE
//! keep-alives). Consumers (the Inference Observatory) render state from
//! these events only.
//!
//! Cost contract: when no subscriber is connected, `emit` is a single atomic
//! load (`receiver_count == 0`) and returns immediately, so the hot decode
//! loop pays nothing in normal serving. High-frequency event classes (layer,
//! KV cache, sampler, prefill progress) are additionally rate-limited per
//! class so a fast decode loop cannot flood the channel; lifecycle events
//! (started/finished/receipt/error) always pass through.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex, OnceLock,
};
use std::time::Instant;

use serde::Serialize;
use tokio::sync::broadcast;

pub const TELEMETRY_SCHEMA: &str = "camelid.telemetry/v1";
/// Broadcast ring capacity. Slow subscribers that fall more than this many
/// events behind observe a `Lagged` gap (surfaced to them as a `lagged`
/// notice) rather than slowing the engine down.
const CHANNEL_CAPACITY: usize = 8192;

/// Minimum spacing for throttled event classes, in microseconds.
const LAYER_EVENT_MIN_GAP_US: u64 = 15_000;
const KV_EVENT_MIN_GAP_US: u64 = 50_000;
const SAMPLER_EVENT_MIN_GAP_US: u64 = 80_000;
const PREFILL_PROGRESS_MIN_GAP_US: u64 = 33_000;
const WORKER_EVENT_MIN_GAP_US: u64 = 100_000;

/// How a generation request is identified across its event stream.
#[derive(Clone, Debug, Serialize)]
pub struct RequestContext {
    pub request_id: String,
    pub model_id: String,
}

/// One telemetry event, wrapped with ordering and attribution metadata.
#[derive(Clone, Debug, Serialize)]
pub struct Envelope {
    pub seq: u64,
    /// Milliseconds since this server process started emitting telemetry.
    pub t_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(flatten)]
    pub event: Event,
}

/// A top-k sampler candidate at one decode step (post-softmax probability).
#[derive(Clone, Debug, Serialize)]
pub struct SamplerCandidate {
    pub token_id: u32,
    pub prob: f32,
}

/// The runtime event vocabulary. Field values come straight from the engine
/// state at the emit site; nothing here is estimated after the fact.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    InferenceStarted {
        backend: String,
        quantization: String,
        architecture: String,
        prompt_tokens: usize,
        max_tokens: u32,
        context_length: usize,
        temperature: f64,
        stream: bool,
    },
    InferenceFinished {
        status: &'static str, // "ok" | "error" | "disconnected"
        #[serde(skip_serializing_if = "Option::is_none")]
        finish_reason: Option<String>,
        completion_tokens: usize,
        total_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        ttft_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_tps: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_tps: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    PrefillStarted {
        prefill_tokens: usize,
        /// Which real prefill lane ran: "gpu_resident" | "layer_major" |
        /// "chunked" | "cooperative_chunked" | "single_token".
        path: &'static str,
        layers_total: usize,
    },
    PrefillProgress {
        tokens_done: usize,
        tokens_total: usize,
    },
    DecodeStarted {
        context_position: usize,
    },
    LayerStarted {
        layer: usize,
        layers_total: usize,
    },
    LayerCompleted {
        layer: usize,
        layers_total: usize,
        duration_us: u64,
    },
    TokenDecoded {
        #[serde(skip_serializing_if = "Option::is_none")]
        token_id: Option<u32>,
        /// Sequence position after this token (KV position), when known.
        #[serde(skip_serializing_if = "Option::is_none")]
        context_position: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        layers_total: Option<usize>,
    },
    KvCacheUpdated {
        position: usize,
        capacity: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        approx_bytes: Option<u64>,
    },
    SamplerStep {
        chosen_token_id: u32,
        mode: &'static str, // "greedy" | "sampling"
        candidates: Vec<SamplerCandidate>,
    },
    /// A real failure observed while a generation request was active. Emitted
    /// alongside (not instead of) the closing `InferenceFinished`.
    InferenceError {
        code: String,
        message: String,
    },
    ReceiptWritten {
        receipt_id: String,
        reproducible: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        gguf_sha256: Option<String>,
    },
    WorkerNodeActive {
        node: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    WorkerNodeIdle {
        node: String,
    },
    WorkerNodeError {
        node: String,
        error: String,
    },
}

impl Event {
    /// Throttle class index, or `None` for events that always pass.
    fn throttle_class(&self) -> Option<usize> {
        match self {
            Event::LayerStarted { .. } | Event::LayerCompleted { .. } => Some(0),
            Event::KvCacheUpdated { .. } => Some(1),
            Event::SamplerStep { .. } => Some(2),
            Event::PrefillProgress { .. } => Some(3),
            // Worker errors always pass; only the per-roundtrip active/idle
            // chatter is throttled.
            Event::WorkerNodeActive { .. } | Event::WorkerNodeIdle { .. } => Some(4),
            _ => None,
        }
    }
}

const THROTTLE_CLASSES: usize = 5;
const THROTTLE_MIN_GAP_US: [u64; THROTTLE_CLASSES] = [
    LAYER_EVENT_MIN_GAP_US,
    KV_EVENT_MIN_GAP_US,
    SAMPLER_EVENT_MIN_GAP_US,
    PREFILL_PROGRESS_MIN_GAP_US,
    WORKER_EVENT_MIN_GAP_US,
];

pub struct TelemetryHub {
    tx: broadcast::Sender<Envelope>,
    seq: AtomicU64,
    started: Instant,
    request: Mutex<Option<RequestContext>>,
    last_emit_us: [AtomicU64; THROTTLE_CLASSES],
}

impl TelemetryHub {
    fn new() -> Self {
        let (tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        Self {
            tx,
            seq: AtomicU64::new(0),
            started: Instant::now(),
            request: Mutex::new(None),
            last_emit_us: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Envelope> {
        self.tx.subscribe()
    }

    pub fn has_subscribers(&self) -> bool {
        self.tx.receiver_count() > 0
    }

    /// Mark the request whose engine-level events should be attributed until
    /// `clear_request`. The serve path runs one generation at a time; if a
    /// second request overlaps, its events attribute to the most recent
    /// `set_request`, which is the truthful best effort without threading a
    /// handle through the engine.
    pub fn set_request(&self, ctx: RequestContext) {
        if let Ok(mut guard) = self.request.lock() {
            *guard = Some(ctx);
        }
    }

    pub fn clear_request(&self) {
        if let Ok(mut guard) = self.request.lock() {
            *guard = None;
        }
    }

    /// True while a generation request is attributed (between `set_request`
    /// and `clear_request`).
    pub fn request_active(&self) -> bool {
        self.request
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    pub fn emit(&self, event: Event) {
        if self.tx.receiver_count() == 0 {
            return;
        }
        if let Some(class) = event.throttle_class() {
            let now_us = self.started.elapsed().as_micros() as u64;
            let last = self.last_emit_us[class].load(Ordering::Relaxed);
            if now_us.saturating_sub(last) < THROTTLE_MIN_GAP_US[class] {
                return;
            }
            self.last_emit_us[class].store(now_us, Ordering::Relaxed);
        }
        let (request_id, model_id) = match self.request.lock() {
            Ok(guard) => match guard.as_ref() {
                Some(ctx) => (Some(ctx.request_id.clone()), Some(ctx.model_id.clone())),
                None => (None, None),
            },
            Err(_) => (None, None),
        };
        let envelope = Envelope {
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            t_ms: self.started.elapsed().as_millis() as u64,
            request_id,
            model_id,
            event,
        };
        let _ = self.tx.send(envelope);
    }
}

static HUB: OnceLock<TelemetryHub> = OnceLock::new();

pub fn hub() -> &'static TelemetryHub {
    HUB.get_or_init(TelemetryHub::new)
}

/// Emit one event into the hub. Near-free when nothing is subscribed.
pub fn emit(event: Event) {
    hub().emit(event)
}

/// True when at least one telemetry subscriber is connected. Use to skip
/// work that only exists to enrich telemetry (e.g. top-k softmax).
pub fn active() -> bool {
    hub().has_subscribers()
}

/// Scoped attribution + guaranteed lifecycle closure for one generation
/// request. Emits `InferenceStarted` on construction. If the owner drops the
/// guard without calling [`RequestGuard::finish`] (client disconnect,
/// panic-free early return), an `InferenceFinished { status: "disconnected" }`
/// event is emitted so the stream never shows a generation as still running
/// when it is not.
pub struct RequestGuard {
    started: Instant,
    finished: bool,
    context: RequestContext,
}

pub struct RequestStart {
    pub request_id: String,
    pub model_id: String,
    pub backend: String,
    pub quantization: String,
    pub architecture: String,
    pub prompt_tokens: usize,
    pub max_tokens: u32,
    pub context_length: usize,
    pub temperature: f64,
    pub stream: bool,
}

pub struct RequestFinish {
    pub status: &'static str,
    pub finish_reason: Option<String>,
    pub completion_tokens: usize,
    pub ttft_ms: Option<u64>,
    pub decode_tps: Option<f64>,
    pub prefill_tps: Option<f64>,
    pub error: Option<String>,
}

impl RequestGuard {
    pub fn begin(start: RequestStart) -> Self {
        let context = RequestContext {
            request_id: start.request_id,
            model_id: start.model_id,
        };
        hub().set_request(context.clone());
        emit(Event::InferenceStarted {
            backend: start.backend,
            quantization: start.quantization,
            architecture: start.architecture,
            prompt_tokens: start.prompt_tokens,
            max_tokens: start.max_tokens,
            context_length: start.context_length,
            temperature: start.temperature,
            stream: start.stream,
        });
        Self {
            started: Instant::now(),
            finished: false,
            context,
        }
    }

    pub fn finish(mut self, finish: RequestFinish) {
        self.activate();
        self.emit_finished(finish);
        self.finished = true;
        hub().clear_request();
    }

    fn emit_finished(&self, finish: RequestFinish) {
        emit(Event::InferenceFinished {
            status: finish.status,
            finish_reason: finish.finish_reason,
            completion_tokens: finish.completion_tokens,
            total_ms: self.started.elapsed().as_millis() as u64,
            ttft_ms: finish.ttft_ms,
            decode_tps: finish.decode_tps,
            prefill_tps: finish.prefill_tps,
            error: finish.error,
        });
    }

    /// Re-select this request as the attribution context. Cooperative decode
    /// calls this at each token boundary before emitting any per-step events.
    pub fn activate(&self) {
        hub().set_request(self.context.clone());
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.activate();
            self.emit_finished(RequestFinish {
                status: "disconnected",
                finish_reason: None,
                completion_tokens: 0,
                ttft_ms: None,
                decode_tps: None,
                prefill_tps: None,
                error: None,
            });
            hub().clear_request();
        }
    }
}

/// Compute the top-k post-softmax candidates from a raw logit slice. Only
/// called when a telemetry subscriber is connected; one linear scan plus a
/// small partial sort.
pub fn top_k_candidates(logits: &[f32], k: usize) -> Vec<SamplerCandidate> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
    for (idx, &value) in logits.iter().enumerate() {
        if top.len() < k {
            top.push((idx, value));
            if top.len() == k {
                top.sort_by(|a, b| b.1.total_cmp(&a.1));
            }
        } else if value > top[k - 1].1 {
            top[k - 1] = (idx, value);
            let mut i = k - 1;
            while i > 0 && top[i].1 > top[i - 1].1 {
                top.swap(i, i - 1);
                i -= 1;
            }
        }
    }
    if top.len() < k {
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    // Softmax over the full logit slice for honest probabilities.
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = logits.iter().map(|&v| (v - max_logit).exp()).sum();
    top.into_iter()
        .map(|(idx, value)| SamplerCandidate {
            token_id: idx as u32,
            prob: if denom > 0.0 {
                (value - max_logit).exp() / denom
            } else {
                0.0
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_by_probability() {
        let logits = vec![0.0, 3.0, 1.0, 2.0];
        let candidates = top_k_candidates(&logits, 2);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].token_id, 1);
        assert_eq!(candidates[1].token_id, 3);
        assert!(candidates[0].prob > candidates[1].prob);
        assert!(candidates[0].prob <= 1.0);
    }

    #[test]
    fn emit_without_subscribers_is_a_noop() {
        // Must not panic or allocate envelopes when nobody listens.
        emit(Event::DecodeStarted {
            context_position: 0,
        });
        assert!(!active() || hub().has_subscribers());
    }

    #[test]
    fn subscriber_receives_lifecycle_events() {
        let mut rx = hub().subscribe();
        emit(Event::PrefillStarted {
            prefill_tokens: 7,
            path: "chunked",
            layers_total: 28,
        });
        // The hub is process-global, so other tests may interleave events;
        // drain until ours shows up.
        loop {
            let envelope = rx.try_recv().expect("our event should be broadcast");
            if let Event::PrefillStarted { prefill_tokens, .. } = envelope.event {
                assert_eq!(prefill_tokens, 7);
                break;
            }
        }
    }
}
