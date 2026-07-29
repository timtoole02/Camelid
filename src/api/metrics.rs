//! Low-overhead Prometheus telemetry for the production HTTP server.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use axum::{
    extract::{Request, State},
    http::{header::CONTENT_TYPE, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use super::{AppState, GenerationTimings};

#[derive(Clone, Default)]
pub(crate) struct ServerMetrics {
    inner: Arc<MetricsInner>,
}

#[derive(Default)]
struct MetricsInner {
    http_requests: AtomicU64,
    http_failures: AtomicU64,
    http_duration_micros: AtomicU64,
    generation_requests: AtomicU64,
    generation_failures: AtomicU64,
    prompt_tokens: AtomicU64,
    decode_tokens: AtomicU64,
    generation_duration_micros: AtomicU64,
    prompt_eval_duration_micros: AtomicU64,
    decode_duration_micros: AtomicU64,
    prompt_cache_hits: AtomicU64,
    prompt_cache_misses: AtomicU64,
    weight_cache_hits: AtomicU64,
    weight_cache_misses: AtomicU64,
}

impl ServerMetrics {
    pub(crate) fn record_generation(
        &self,
        prompt_tokens: usize,
        decode_tokens: usize,
        timings: &GenerationTimings,
    ) {
        self.inner
            .generation_requests
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.inner
            .decode_tokens
            .fetch_add(decode_tokens as u64, Ordering::Relaxed);
        self.inner.generation_duration_micros.fetch_add(
            saturating_u128_to_u64(timings.generate).saturating_mul(1_000),
            Ordering::Relaxed,
        );

        let prompt_eval_ms = timings.prompt_evaluation.prefill.forward_total
            + timings.prompt_evaluation.first_token.forward_total;
        let generation_ms = timings.generation.forward_total;
        self.inner
            .prompt_eval_duration_micros
            .fetch_add(finite_millis_to_micros(prompt_eval_ms), Ordering::Relaxed);
        self.inner.decode_duration_micros.fetch_add(
            finite_millis_to_micros((generation_ms - prompt_eval_ms).max(0.0)),
            Ordering::Relaxed,
        );
        counter_for(
            timings.prompt_cache_hit,
            &self.inner.prompt_cache_hits,
            &self.inner.prompt_cache_misses,
        );
        counter_for(
            timings.weight_cache_hit,
            &self.inner.weight_cache_hits,
            &self.inner.weight_cache_misses,
        );
    }

    pub(crate) fn record_generation_failure(&self) {
        self.inner
            .generation_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    fn render(&self, state: &AppState) -> String {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let slot = state.engine.slot_snapshot();
        let hw = crate::capability::HardwareProfile::cached();
        let mut out = String::with_capacity(4_096);
        metric_counter(
            &mut out,
            "camelid_http_requests_total",
            "HTTP requests completed.",
            load(&self.inner.http_requests),
        );
        metric_counter(
            &mut out,
            "camelid_http_failures_total",
            "HTTP responses with a 4xx or 5xx status.",
            load(&self.inner.http_failures),
        );
        metric_counter(
            &mut out,
            "camelid_http_request_duration_seconds_sum",
            "Cumulative HTTP handler wall time.",
            micros_seconds(load(&self.inner.http_duration_micros)),
        );
        metric_counter(
            &mut out,
            "camelid_generation_requests_total",
            "Generation requests that completed a decode loop.",
            load(&self.inner.generation_requests),
        );
        metric_counter(
            &mut out,
            "camelid_generation_failures_total",
            "Generation jobs that failed, timed out, or were cancelled.",
            load(&self.inner.generation_failures),
        );
        metric_counter(
            &mut out,
            "camelid_prompt_tokens_total",
            "Prompt tokens evaluated by completed generation requests.",
            load(&self.inner.prompt_tokens),
        );
        metric_counter(
            &mut out,
            "camelid_decode_tokens_total",
            "Tokens decoded by completed generation requests.",
            load(&self.inner.decode_tokens),
        );
        metric_counter(
            &mut out,
            "camelid_generation_duration_seconds_sum",
            "Cumulative generation wall time.",
            micros_seconds(load(&self.inner.generation_duration_micros)),
        );
        metric_counter(
            &mut out,
            "camelid_prompt_evaluation_duration_seconds_sum",
            "Cumulative prompt evaluation forward time.",
            micros_seconds(load(&self.inner.prompt_eval_duration_micros)),
        );
        metric_counter(
            &mut out,
            "camelid_decode_duration_seconds_sum",
            "Cumulative post-prompt decode forward time.",
            micros_seconds(load(&self.inner.decode_duration_micros)),
        );
        for (name, help, value) in [
            (
                "camelid_prompt_cache_hits_total",
                "Completed generations that reused the prompt-prefix cache.",
                load(&self.inner.prompt_cache_hits),
            ),
            (
                "camelid_prompt_cache_misses_total",
                "Completed generations that did not reuse the prompt-prefix cache.",
                load(&self.inner.prompt_cache_misses),
            ),
            (
                "camelid_weight_cache_hits_total",
                "Completed generations that reused loaded weights.",
                load(&self.inner.weight_cache_hits),
            ),
            (
                "camelid_weight_cache_misses_total",
                "Completed generations that loaded weights.",
                load(&self.inner.weight_cache_misses),
            ),
        ] {
            metric_counter(&mut out, name, help, value);
        }

        metric_gauge(
            &mut out,
            "camelid_engine_queue_depth",
            "Accepted engine jobs, queued plus active.",
            state.engine.depth() as u64,
        );
        metric_gauge(
            &mut out,
            "camelid_engine_queued_tasks",
            "Engine jobs waiting behind the active job.",
            slot.queued_tasks as u64,
        );
        metric_gauge(
            &mut out,
            "camelid_engine_active_slots",
            "Active single-owner engine slots.",
            u64::from(slot.is_processing()),
        );
        metric_gauge(
            &mut out,
            "camelid_engine_active_generated_tokens",
            "Tokens completed by the active generation job.",
            slot.completed_units,
        );
        metric_gauge(
            &mut out,
            "camelid_engine_active_elapsed_seconds",
            "Wall time since the active engine job started.",
            slot.active_elapsed_seconds,
        );
        metric_gauge(
            &mut out,
            "camelid_engine_active_stalled_seconds",
            "Wall time since the active engine job last reported token progress.",
            slot.stalled_seconds,
        );
        metric_gauge(
            &mut out,
            "camelid_process_resident_memory_bytes",
            "Current process resident memory (0 when the platform cannot report it).",
            current_process_rss_bytes().unwrap_or(0),
        );
        metric_gauge(
            &mut out,
            "camelid_cuda_vram_total_bytes",
            "CUDA VRAM total at the cached hardware probe.",
            hw.cuda_vram_total_bytes,
        );
        metric_gauge(
            &mut out,
            "camelid_cuda_vram_free_bytes",
            "CUDA VRAM free at the cached hardware probe.",
            hw.cuda_vram_free_bytes,
        );
        out
    }
}

pub(crate) async fn observe_http(
    State(metrics): State<ServerMetrics>,
    request: Request,
    next: Next,
) -> Response {
    let started = std::time::Instant::now();
    let response = next.run(request).await;
    metrics.inner.http_requests.fetch_add(1, Ordering::Relaxed);
    metrics.inner.http_duration_micros.fetch_add(
        saturating_u128_to_u64(started.elapsed().as_micros()),
        Ordering::Relaxed,
    );
    if response.status().is_client_error() || response.status().is_server_error() {
        metrics.inner.http_failures.fetch_add(1, Ordering::Relaxed);
    }
    response
}

pub(crate) async fn prometheus(State(state): State<AppState>) -> Response {
    let mut response = (StatusCode::OK, state.metrics.render(&state)).into_response();
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

fn counter_for(value: bool, yes: &AtomicU64, no: &AtomicU64) {
    if value {
        yes.fetch_add(1, Ordering::Relaxed);
    } else {
        no.fetch_add(1, Ordering::Relaxed);
    }
}

fn saturating_u128_to_u64(value: u128) -> u64 {
    value.try_into().unwrap_or(u64::MAX)
}

fn finite_millis_to_micros(value: f64) -> u64 {
    if value.is_finite() && value > 0.0 {
        (value * 1_000.0).min(u64::MAX as f64) as u64
    } else {
        0
    }
}

fn micros_seconds(value: u64) -> String {
    format!("{:.6}", value as f64 / 1_000_000.0)
}

fn metric_counter(out: &mut String, name: &str, help: &str, value: impl ToString) {
    use std::fmt::Write;
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} counter");
    let _ = writeln!(out, "{name} {}", value.to_string());
}

fn metric_gauge(out: &mut String, name: &str, help: &str, value: impl ToString) {
    use std::fmt::Write;
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    let _ = writeln!(out, "{name} {}", value.to_string());
}

#[cfg(target_os = "linux")]
fn current_process_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[cfg(windows)]
fn current_process_rss_bytes() -> Option<u64> {
    use windows_sys::Win32::System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::GetCurrentProcess,
    };
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        PageFaultCount: 0,
        PeakWorkingSetSize: 0,
        WorkingSetSize: 0,
        QuotaPeakPagedPoolUsage: 0,
        QuotaPagedPoolUsage: 0,
        QuotaPeakNonPagedPoolUsage: 0,
        QuotaNonPagedPoolUsage: 0,
        PagefileUsage: 0,
        PeakPagefileUsage: 0,
    };
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (ok != 0).then_some(counters.WorkingSetSize as u64)
}

#[cfg(not(any(target_os = "linux", windows)))]
fn current_process_rss_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposition_is_prometheus_text_and_has_no_model_or_secret_labels() {
        let state = AppState::default();
        state
            .metrics
            .record_generation(12, 5, &GenerationTimings::default());
        let text = state.metrics.render(&state);
        assert!(text.contains("# TYPE camelid_prompt_tokens_total counter"));
        assert!(text.contains("camelid_prompt_tokens_total 12"));
        assert!(text.contains("camelid_decode_tokens_total 5"));
        assert!(!text.contains('{'));
    }
}
