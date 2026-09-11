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
    cuda_true_batch2_forwards: AtomicU64,
    cuda_true_batch2_rows: AtomicU64,
    cuda_true_batch2_shared_projection_launches: AtomicU64,
    cuda_true_batch2_duration_micros: AtomicU64,
    cuda_true_batch2_preflight_fallbacks: AtomicU64,
    cuda_true_batch_forwards: AtomicU64,
    cuda_true_batch_rows: AtomicU64,
    cuda_true_batch_shared_projection_launches: AtomicU64,
    cuda_true_batch_duration_micros: AtomicU64,
    cuda_true_batch_preflight_fallbacks: AtomicU64,
    cuda_true_batch_size_forwards: [AtomicU64; 9],
    cuda_batched_prefill_forwards: AtomicU64,
    cuda_batched_prefill_rows: AtomicU64,
    cuda_batched_prefill_shared_projection_launches: AtomicU64,
    cuda_batched_prefill_duration_micros: AtomicU64,
    cuda_batched_prefill_size_forwards: [AtomicU64; 9],
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

    pub(crate) fn record_cuda_true_batch2(
        &self,
        shared_projection_launches: usize,
        duration_micros: u128,
    ) {
        self.record_cuda_true_batch(2, shared_projection_launches, duration_micros);
    }

    pub(crate) fn record_cuda_true_batch(
        &self,
        batch_size: usize,
        shared_projection_launches: usize,
        duration_micros: u128,
    ) {
        debug_assert!((2..=8).contains(&batch_size));
        self.inner
            .cuda_true_batch_forwards
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .cuda_true_batch_rows
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.inner
            .cuda_true_batch_shared_projection_launches
            .fetch_add(
                shared_projection_launches.try_into().unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        self.inner
            .cuda_true_batch_duration_micros
            .fetch_add(saturating_u128_to_u64(duration_micros), Ordering::Relaxed);
        if let Some(counter) = self.inner.cuda_true_batch_size_forwards.get(batch_size) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        if batch_size != 2 {
            return;
        }
        self.inner
            .cuda_true_batch2_forwards
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .cuda_true_batch2_rows
            .fetch_add(2, Ordering::Relaxed);
        self.inner
            .cuda_true_batch2_shared_projection_launches
            .fetch_add(
                shared_projection_launches.try_into().unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        self.inner
            .cuda_true_batch2_duration_micros
            .fetch_add(saturating_u128_to_u64(duration_micros), Ordering::Relaxed);
    }

    pub(crate) fn record_cuda_true_batch2_preflight_fallback(&self) {
        self.record_cuda_true_batch_preflight_fallback();
        self.inner
            .cuda_true_batch2_preflight_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cuda_true_batch_preflight_fallback(&self) {
        self.inner
            .cuda_true_batch_preflight_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cuda_batched_prefill(
        &self,
        batch_size: usize,
        shared_projection_launches: usize,
        duration_micros: u128,
    ) {
        debug_assert!((2..=8).contains(&batch_size));
        self.inner
            .cuda_batched_prefill_forwards
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .cuda_batched_prefill_rows
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.inner
            .cuda_batched_prefill_shared_projection_launches
            .fetch_add(
                shared_projection_launches.try_into().unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        self.inner
            .cuda_batched_prefill_duration_micros
            .fetch_add(saturating_u128_to_u64(duration_micros), Ordering::Relaxed);
        if let Some(counter) = self
            .inner
            .cuda_batched_prefill_size_forwards
            .get(batch_size)
        {
            counter.fetch_add(1, Ordering::Relaxed);
        }
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
            (
                "camelid_cuda_true_batch2_forwards_total",
                "Successful CUDA forwards that jointly served two independent rows.",
                load(&self.inner.cuda_true_batch2_forwards),
            ),
            (
                "camelid_cuda_true_batch2_rows_total",
                "Independent rows served by successful CUDA true batch-2 forwards.",
                load(&self.inner.cuda_true_batch2_rows),
            ),
            (
                "camelid_cuda_true_batch2_shared_projection_launches_total",
                "Shared projection launches issued by successful CUDA true batch-2 forwards.",
                load(&self.inner.cuda_true_batch2_shared_projection_launches),
            ),
            (
                "camelid_cuda_true_batch2_preflight_fallbacks_total",
                "Candidate CUDA batch-2 pairs refused before shared dispatch and run scalar.",
                load(&self.inner.cuda_true_batch2_preflight_fallbacks),
            ),
        ] {
            metric_counter(&mut out, name, help, value);
        }
        metric_counter(
            &mut out,
            "camelid_cuda_true_batch2_duration_seconds_sum",
            "Cumulative shared-forward wall time for successful CUDA true batch-2 dispatches.",
            micros_seconds(load(&self.inner.cuda_true_batch2_duration_micros)),
        );
        for (name, help, value) in [
            (
                "camelid_cuda_true_batch_forwards_total",
                "Successful CUDA forwards that jointly served two to eight independent rows.",
                load(&self.inner.cuda_true_batch_forwards),
            ),
            (
                "camelid_cuda_true_batch_rows_total",
                "Independent rows served by successful CUDA true-batch forwards.",
                load(&self.inner.cuda_true_batch_rows),
            ),
            (
                "camelid_cuda_true_batch_shared_projection_launches_total",
                "Shared projection launches issued by successful CUDA true-batch forwards.",
                load(&self.inner.cuda_true_batch_shared_projection_launches),
            ),
            (
                "camelid_cuda_true_batch_preflight_fallbacks_total",
                "Candidate CUDA groups refused before shared dispatch and run scalar.",
                load(&self.inner.cuda_true_batch_preflight_fallbacks),
            ),
        ] {
            metric_counter(&mut out, name, help, value);
        }
        metric_counter(
            &mut out,
            "camelid_cuda_true_batch_duration_seconds_sum",
            "Cumulative shared-forward wall time for successful CUDA true-batch dispatches.",
            micros_seconds(load(&self.inner.cuda_true_batch_duration_micros)),
        );
        for batch_size in 2..=8 {
            metric_counter(
                &mut out,
                &format!("camelid_cuda_true_batch_size_{batch_size}_forwards_total"),
                "Successful CUDA true-batch forwards at this exact row count.",
                load(&self.inner.cuda_true_batch_size_forwards[batch_size]),
            );
        }
        for (name, help, value) in [
            (
                "camelid_cuda_batched_prefill_forwards_total",
                "Successful CUDA prefill forwards shared by independent sequences.",
                load(&self.inner.cuda_batched_prefill_forwards),
            ),
            (
                "camelid_cuda_batched_prefill_rows_total",
                "Independent prompt rows served by shared CUDA prefill forwards.",
                load(&self.inner.cuda_batched_prefill_rows),
            ),
            (
                "camelid_cuda_batched_prefill_shared_projection_launches_total",
                "Shared projection launches issued by cross-request CUDA prefill.",
                load(&self.inner.cuda_batched_prefill_shared_projection_launches),
            ),
        ] {
            metric_counter(&mut out, name, help, value);
        }
        metric_counter(
            &mut out,
            "camelid_cuda_batched_prefill_duration_seconds_sum",
            "Cumulative shared-forward wall time for cross-request CUDA prefill.",
            micros_seconds(load(&self.inner.cuda_batched_prefill_duration_micros)),
        );
        for batch_size in 2..=8 {
            metric_counter(
                &mut out,
                &format!("camelid_cuda_batched_prefill_size_{batch_size}_forwards_total"),
                "Successful cross-request CUDA prefill forwards at this exact row count.",
                load(&self.inner.cuda_batched_prefill_size_forwards[batch_size]),
            );
        }
        metric_counter(
            &mut out,
            "camelid_cuda_paged_kv_allocation_failures_total",
            "Paged CUDA KV reservations refused atomically for insufficient capacity.",
            crate::inference::cuda_paged_kv::allocation_failures_total(),
        );
        metric_counter(
            &mut out,
            "camelid_cuda_paged_kv_reclaimed_pages_total",
            "Paged CUDA KV pages reclaimed from completed, cancelled, or aborted sequences.",
            crate::inference::cuda_paged_kv::reclaimed_pages_total(),
        );
        let paged =
            crate::inference::active_resident_cuda_status().and_then(|status| status.paged_kv);
        for (name, help, value) in [
            (
                "camelid_cuda_paged_kv_capacity_pages",
                "Logical page capacity of the active CUDA paged KV allocator.",
                paged.map_or(0, |status| status.capacity_pages as u64),
            ),
            (
                "camelid_cuda_paged_kv_allocated_pages",
                "Pages owned by active sequences in the CUDA paged KV allocator.",
                paged.map_or(0, |status| status.allocated_pages as u64),
            ),
            (
                "camelid_cuda_paged_kv_free_pages",
                "Unallocated logical pages in the active CUDA paged KV allocator.",
                paged.map_or(0, |status| status.free_pages as u64),
            ),
            (
                "camelid_cuda_paged_kv_fragmented_pages",
                "Vacant reusable page indices below the allocator metadata high-water mark.",
                paged.map_or(0, |status| status.fragmented_pages as u64),
            ),
            (
                "camelid_cuda_paged_kv_shared_pages",
                "Immutable CUDA paged KV pages referenced by more than one sequence.",
                paged.map_or(0, |status| status.shared_pages as u64),
            ),
            (
                "camelid_cuda_paged_kv_high_watermark_pages",
                "Maximum simultaneously allocated pages in the active CUDA paged KV allocator.",
                paged.map_or(0, |status| status.high_watermark_pages as u64),
            ),
            (
                "camelid_cuda_paged_kv_allocated_bytes",
                "Logical bytes owned by active CUDA paged KV sequences.",
                paged.map_or(0, |status| status.allocated_bytes as u64),
            ),
            (
                "camelid_cuda_paged_kv_device_allocated_bytes",
                "Physical device bytes held by active CUDA paged KV pages.",
                paged.map_or(0, |status| status.device_allocated_bytes as u64),
            ),
            (
                "camelid_cuda_paged_kv_device_high_watermark_bytes",
                "Maximum physical bytes held by the active CUDA paged KV page store.",
                paged.map_or(0, |status| status.device_high_watermark_bytes as u64),
            ),
            (
                "camelid_cuda_paged_kv_sequences",
                "Sequences currently owning a CUDA paged KV page table.",
                paged.map_or(0, |status| status.sequence_count as u64),
            ),
        ] {
            metric_gauge(&mut out, name, help, value);
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
            state.engine.busy_slots() as u64,
        );
        metric_gauge(
            &mut out,
            "camelid_engine_active_generated_tokens",
            "Tokens completed by the active generation job.",
            slot.completed_units,
        );
        let (prefill_completed, prefill_total) = state.engine.cooperative_prefill_progress();
        metric_gauge(
            &mut out,
            "camelid_engine_active_prefill_tokens",
            "Prompt tokens completed by active cooperative prefill jobs.",
            prefill_completed,
        );
        metric_gauge(
            &mut out,
            "camelid_engine_active_prefill_tokens_total",
            "Total prompt tokens scheduled by active cooperative prefill jobs.",
            prefill_total,
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
pub(super) fn current_process_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[cfg(windows)]
pub(super) fn current_process_rss_bytes() -> Option<u64> {
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

#[cfg(target_os = "macos")]
pub(super) fn current_process_rss_bytes() -> Option<u64> {
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    let result = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V2,
            &mut info as *mut libc::rusage_info_v2 as *mut libc::rusage_info_t,
        )
    };
    (result == 0 && info.ri_phys_footprint > 0).then_some(info.ri_phys_footprint)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub(super) fn current_process_rss_bytes() -> Option<u64> {
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
        state.metrics.record_cuda_true_batch2(15, 1_250);
        state.metrics.record_cuda_true_batch2_preflight_fallback();
        state.metrics.record_cuda_true_batch(4, 29, 2_500);
        state.metrics.record_cuda_true_batch_preflight_fallback();
        state.metrics.record_cuda_batched_prefill(4, 28, 2_000);
        let text = state.metrics.render(&state);
        assert!(text.contains("# TYPE camelid_prompt_tokens_total counter"));
        assert!(text.contains("camelid_prompt_tokens_total 12"));
        assert!(text.contains("camelid_decode_tokens_total 5"));
        assert!(text.contains("camelid_cuda_true_batch2_forwards_total 1"));
        assert!(text.contains("camelid_cuda_true_batch2_rows_total 2"));
        assert!(text.contains("camelid_cuda_true_batch2_shared_projection_launches_total 15"));
        assert!(text.contains("camelid_cuda_true_batch2_preflight_fallbacks_total 1"));
        assert!(text.contains("camelid_cuda_true_batch2_duration_seconds_sum 0.001250"));
        assert!(text.contains("camelid_cuda_true_batch_forwards_total 2"));
        assert!(text.contains("camelid_cuda_true_batch_rows_total 6"));
        assert!(text.contains("camelid_cuda_true_batch_shared_projection_launches_total 44"));
        assert!(text.contains("camelid_cuda_true_batch_preflight_fallbacks_total 2"));
        assert!(text.contains("camelid_cuda_true_batch_duration_seconds_sum 0.003750"));
        assert!(text.contains("camelid_cuda_true_batch_size_2_forwards_total 1"));
        assert!(text.contains("camelid_cuda_true_batch_size_4_forwards_total 1"));
        assert!(text.contains("camelid_cuda_batched_prefill_forwards_total 1"));
        assert!(text.contains("camelid_cuda_batched_prefill_rows_total 4"));
        assert!(text.contains("camelid_cuda_batched_prefill_shared_projection_launches_total 28"));
        assert!(text.contains("camelid_cuda_batched_prefill_duration_seconds_sum 0.002000"));
        assert!(text.contains("camelid_cuda_batched_prefill_size_4_forwards_total 1"));
        assert!(text.contains("# TYPE camelid_cuda_paged_kv_allocation_failures_total counter"));
        assert!(text.contains("# TYPE camelid_cuda_paged_kv_reclaimed_pages_total counter"));
        assert!(text.contains("# TYPE camelid_cuda_paged_kv_fragmented_pages gauge"));
        assert!(text.contains("# TYPE camelid_cuda_paged_kv_device_allocated_bytes gauge"));
        assert!(!text.contains('{'));
    }
}
