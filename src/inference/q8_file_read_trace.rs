use crate::tensor::Q8_0FileReadStats;

use super::LlamaQ8FileReadPhaseTrace;

pub(super) fn q8_file_read_stats_has_activity(stats: Q8_0FileReadStats) -> bool {
    stats.read_calls > 0
        || stats.read_bytes > 0
        || stats.cache_hits > 0
        || stats.cache_hit_bytes > 0
        || stats.cache_misses > 0
        || stats.cache_miss_bytes > 0
        || stats.cache_inserts > 0
        || stats.cache_insert_bytes > 0
        || stats.cache_evictions > 0
        || stats.cache_evicted_bytes > 0
        || stats.cache_merges > 0
        || stats.cache_merged_bytes > 0
        || stats.cache_decoded_scale_hits > 0
        || stats.cache_decoded_scale_hit_blocks > 0
}

pub(super) fn add_q8_file_read_stats_delta(
    target: &mut Q8_0FileReadStats,
    delta: Q8_0FileReadStats,
) {
    target.read_calls = target.read_calls.saturating_add(delta.read_calls);
    target.read_bytes = target.read_bytes.saturating_add(delta.read_bytes);
    target.cache_hits = target.cache_hits.saturating_add(delta.cache_hits);
    target.cache_hit_bytes = target.cache_hit_bytes.saturating_add(delta.cache_hit_bytes);
    target.cache_misses = target.cache_misses.saturating_add(delta.cache_misses);
    target.cache_miss_bytes = target
        .cache_miss_bytes
        .saturating_add(delta.cache_miss_bytes);
    target.cache_inserts = target.cache_inserts.saturating_add(delta.cache_inserts);
    target.cache_insert_bytes = target
        .cache_insert_bytes
        .saturating_add(delta.cache_insert_bytes);
    target.cache_evictions = target.cache_evictions.saturating_add(delta.cache_evictions);
    target.cache_evicted_bytes = target
        .cache_evicted_bytes
        .saturating_add(delta.cache_evicted_bytes);
    target.cache_merges = target.cache_merges.saturating_add(delta.cache_merges);
    target.cache_merged_bytes = target
        .cache_merged_bytes
        .saturating_add(delta.cache_merged_bytes);
    target.cache_decoded_scale_hits = target
        .cache_decoded_scale_hits
        .saturating_add(delta.cache_decoded_scale_hits);
    target.cache_decoded_scale_hit_blocks = target
        .cache_decoded_scale_hit_blocks
        .saturating_add(delta.cache_decoded_scale_hit_blocks);
    // These fields are point-in-time cache state, not additive counters. A merged timing window
    // can span a scoped Q8 cache override followed by a later pass after the override is restored.
    target.cache_entries = target.cache_entries.max(delta.cache_entries);
    target.cache_bytes = target.cache_bytes.max(delta.cache_bytes);
    target.cache_capacity_bytes = target.cache_capacity_bytes.max(delta.cache_capacity_bytes);
}

pub(super) fn add_q8_file_read_phase_trace(
    phases: &mut Vec<LlamaQ8FileReadPhaseTrace>,
    phase: &str,
    delta: Q8_0FileReadStats,
) {
    if let Some(existing) = phases.iter_mut().find(|entry| entry.phase == phase) {
        add_q8_file_read_stats_delta(&mut existing.q8_file_reads, delta);
        return;
    }
    phases.push(LlamaQ8FileReadPhaseTrace {
        phase: phase.to_string(),
        q8_file_reads: delta,
    });
}
