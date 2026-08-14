//! Adaptive context-window policy for agent sessions.
//!
//! This module deliberately separates policy from telemetry collection. Callers
//! take one live host-memory snapshot, derive the model's conservative host KV
//! cost, and pass both into [`select_context_window`]. The pure function is then
//! deterministic and cheap to test. A resident accelerator capacity is retained
//! for diagnostics only: exceeding it can make a request slower by falling back
//! to the host cache, but does not make the model context incorrect.
//! Model-specific paged targets are a separate logical-memory policy. They are
//! valid only when the caller also supplies a smaller host-enforced working set
//! that stays inside the exact row's validated active-prompt envelope.

use serde::Serialize;

use crate::capability::HostMemoryStatus;

pub(crate) const DEFAULT_OPERATIONAL_CEILING_TOKENS: u32 = 65_536;
pub(crate) const UNKNOWN_TELEMETRY_FALLBACK_TOKENS: u32 = 8_192;
const MINIMUM_AUTO_CONTEXT_TOKENS: u32 = 8_192;
const CONTEXT_QUANTUM_TOKENS: u64 = 1_024;
const AVAILABLE_MEMORY_PERCENT: u64 = 70;
const CONFIGURED_MAX_ENV: &str = "CAMELID_AGENT_CONTEXT_MAX_TOKENS";

/// All data needed by the pure adaptive context policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextWindowInputs {
    /// Native trained context from GGUF metadata.
    pub native_context_tokens: u32,
    /// Largest context explicitly validated for this agent/model support lane.
    /// Native GGUF metadata is a correctness ceiling, not a support receipt.
    pub validated_context_tokens: u32,
    /// Server-side total envelope: independently enforced prompt ceiling plus
    /// this session's bounded generation allowance.
    pub server_context_tokens: u32,
    /// A fresh, non-cached physical-memory snapshot.
    pub host_memory: Option<HostMemoryStatus>,
    /// Conservative host-cache cost computed from the model's actual KV shape.
    pub kv_bytes_per_token: Option<u64>,
    /// Maximum simultaneous KV owners: active generation slots plus retained
    /// prompt-prefix cache entries. The raw memory-capacity diagnostic divides
    /// its 70% allowance across these owners; the separate 8K operational floor
    /// can exceed that estimate and is never reported as memory-safe.
    pub kv_owner_slots: u32,
    /// Optional resident GPU/Metal capacity. This is telemetry, never a hard cap.
    pub resident_capacity_tokens: Option<u32>,
    /// Optional operator cap. API callers may supply this directly; the normal
    /// environment-backed value comes from [`configured_agent_context_max`].
    pub configured_max_tokens: Option<u32>,
    /// Logical agent-context target available only to exact rows using bounded
    /// context paging. This is not permission to send a prompt this large.
    pub paged_target_tokens: Option<u32>,
    /// Maximum input + output + safety envelope kept resident on each paged
    /// model request. It must fit the validated active-prompt envelope.
    pub paged_working_set_tokens: Option<u32>,
}

/// The bound that ultimately selected `effective_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ContextWindowLimitingFactor {
    ConfiguredMaximum,
    ValidatedAgentMaximum,
    ModelMaximum,
    ServerContextMaximum,
    AvailableMemory,
    MinimumOperationalEnvelope,
    OperationalCeiling,
    UnknownTelemetryFallback,
    PagedModelTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ContextWindowMode {
    Auto,
}

/// Serializable decision record suitable for the workspace status API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct ContextWindowSelection {
    pub mode: ContextWindowMode,
    pub effective_tokens: u32,
    /// The automatic memory/operational recommendation before immutable model,
    /// server, and explicit operator caps are applied. Under severe memory
    /// pressure this is the 8K minimum operational envelope rather than a claim
    /// that 70% of the current available-memory sample can hold 8K; the runtime
    /// allocation guard remains authoritative.
    pub recommended_max_tokens: u32,
    /// Raw memory-derived capacity after slot sharing and quantization. Unlike
    /// `recommended_max_tokens`, this does not apply the 8K operational floor.
    pub memory_safe_max_tokens: Option<u32>,
    pub model_max_tokens: u32,
    pub validated_max_tokens: u32,
    pub limiting_factor: ContextWindowLimitingFactor,
    pub available_memory_bytes: Option<u64>,
    pub kv_bytes_per_token: Option<u64>,
    pub kv_owner_slots: u32,
    /// Performance hint only; intentionally excluded from the cap calculation.
    pub resident_capacity_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configured_max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paged_target_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paged_working_set_tokens: Option<u32>,
}

/// Read the optional process-wide operator cap. Invalid and zero values are
/// ignored, leaving the automatic policy in control.
pub(crate) fn configured_agent_context_max() -> Option<u32> {
    std::env::var(CONFIGURED_MAX_ENV)
        .ok()
        .as_deref()
        .and_then(parse_configured_max)
}

fn parse_configured_max(value: &str) -> Option<u32> {
    value
        .trim()
        .parse::<u32>()
        .ok()
        .filter(|tokens| *tokens > 0)
}

/// Select a context budget from immutable model/server limits and current host
/// memory. The raw capacity divides 70% of *available* RAM across the active
/// and retained KV owners and rounds down to a 1,024-token quantum. The
/// operational recommendation never grows above 65,536 tokens and retains an
/// 8,192 minimum so the paging reserves remain useful; when that floor exceeds
/// the raw estimate, both values are exposed separately instead of claiming the
/// floor is memory-safe. A smaller validated/model/server/operator cap still wins.
pub(crate) fn select_context_window(inputs: ContextWindowInputs) -> ContextWindowSelection {
    let host_memory = inputs
        .host_memory
        .filter(|memory| memory.total_bytes > 0 && memory.available_bytes <= memory.total_bytes);
    let available_memory_bytes = host_memory.map(|memory| memory.available_bytes);
    let kv_bytes_per_token = inputs.kv_bytes_per_token.filter(|bytes| *bytes > 0);
    let kv_owner_slots = inputs.kv_owner_slots.max(1);
    let configured_max_tokens = inputs.configured_max_tokens.filter(|tokens| *tokens > 0);
    let requested_paged_target = inputs.paged_target_tokens.filter(|tokens| *tokens > 0);
    let requested_paged_working_set = inputs.paged_working_set_tokens.filter(|tokens| *tokens > 0);

    let (recommended_max_tokens, memory_safe_max_tokens, automatic_factor) =
        match (available_memory_bytes, kv_bytes_per_token) {
            (Some(available), Some(bytes_per_token)) => {
                let memory_budget = available
                    .saturating_mul(AVAILABLE_MEMORY_PERCENT)
                    .saturating_div(100)
                    .saturating_div(u64::from(kv_owner_slots));
                let raw_tokens = memory_budget / bytes_per_token;
                let quantized_tokens = raw_tokens
                    .saturating_div(CONTEXT_QUANTUM_TOKENS)
                    .saturating_mul(CONTEXT_QUANTUM_TOKENS)
                    .min(u64::from(u32::MAX)) as u32;
                let memory_safe_tokens = quantized_tokens.max(MINIMUM_AUTO_CONTEXT_TOKENS);
                if quantized_tokens < MINIMUM_AUTO_CONTEXT_TOKENS {
                    (
                        MINIMUM_AUTO_CONTEXT_TOKENS,
                        Some(quantized_tokens),
                        ContextWindowLimitingFactor::MinimumOperationalEnvelope,
                    )
                } else if memory_safe_tokens >= DEFAULT_OPERATIONAL_CEILING_TOKENS {
                    (
                        DEFAULT_OPERATIONAL_CEILING_TOKENS,
                        Some(quantized_tokens),
                        ContextWindowLimitingFactor::OperationalCeiling,
                    )
                } else {
                    (
                        memory_safe_tokens,
                        Some(quantized_tokens),
                        ContextWindowLimitingFactor::AvailableMemory,
                    )
                }
            }
            _ => (
                UNKNOWN_TELEMETRY_FALLBACK_TOKENS,
                None,
                ContextWindowLimitingFactor::UnknownTelemetryFallback,
            ),
        };

    // A paged target widens only the host's logical task envelope. Every real
    // request remains bounded by `working_set`, which must itself fit the exact
    // row's validated prompt envelope and the normal operational recommendation.
    // Invalid or partial policy input fails closed to the ordinary validated cap.
    let paged_policy = requested_paged_target
        .zip(requested_paged_working_set)
        .filter(|(target, working_set)| {
            *target >= *working_set
                && *target <= inputs.native_context_tokens
                && *target <= DEFAULT_OPERATIONAL_CEILING_TOKENS
                && *working_set <= inputs.validated_context_tokens
                && *working_set <= inputs.server_context_tokens
                && *working_set <= recommended_max_tokens
        });
    let paged_target_tokens = paged_policy.map(|(target, _)| target);
    let paged_working_set_tokens = paged_policy.map(|(_, working_set)| working_set);

    // Tie order is intentional: an explicit cap is the most useful explanation,
    // followed by the exact row's validated (or bounded-paging) support policy,
    // immutable model/server limits, then the automatic recommendation.
    let mut effective_tokens = u32::MAX;
    let mut limiting_factor = automatic_factor;
    if let Some(configured) = configured_max_tokens {
        effective_tokens = configured;
        limiting_factor = ContextWindowLimitingFactor::ConfiguredMaximum;
    }
    let (support_tokens, support_factor) = paged_target_tokens.map_or(
        (
            inputs.validated_context_tokens,
            ContextWindowLimitingFactor::ValidatedAgentMaximum,
        ),
        |target| (target, ContextWindowLimitingFactor::PagedModelTarget),
    );
    let (automatic_tokens, selected_automatic_factor) = paged_target_tokens
        .map_or((recommended_max_tokens, automatic_factor), |target| {
            (target, ContextWindowLimitingFactor::PagedModelTarget)
        });
    for (tokens, factor) in [
        (support_tokens, support_factor),
        (
            inputs.native_context_tokens,
            ContextWindowLimitingFactor::ModelMaximum,
        ),
        (
            inputs.server_context_tokens,
            ContextWindowLimitingFactor::ServerContextMaximum,
        ),
        (automatic_tokens, selected_automatic_factor),
    ] {
        if tokens < effective_tokens {
            effective_tokens = tokens;
            limiting_factor = factor;
        }
    }

    ContextWindowSelection {
        mode: ContextWindowMode::Auto,
        effective_tokens,
        recommended_max_tokens,
        memory_safe_max_tokens,
        model_max_tokens: inputs.native_context_tokens,
        validated_max_tokens: inputs.validated_context_tokens,
        limiting_factor,
        available_memory_bytes,
        kv_bytes_per_token,
        kv_owner_slots,
        resident_capacity_tokens: inputs.resident_capacity_tokens,
        configured_max_tokens,
        paged_target_tokens,
        paged_working_set_tokens: paged_working_set_tokens
            .map(|tokens| tokens.min(effective_tokens)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> ContextWindowInputs {
        ContextWindowInputs {
            native_context_tokens: 131_072,
            validated_context_tokens: 131_072,
            server_context_tokens: 131_072,
            host_memory: Some(HostMemoryStatus {
                total_bytes: 32 * 1024 * 1024 * 1024,
                available_bytes: 16 * 1024 * 1024 * 1024,
            }),
            kv_bytes_per_token: Some(64 * 1024),
            kv_owner_slots: 1,
            resident_capacity_tokens: None,
            configured_max_tokens: None,
            paged_target_tokens: None,
            paged_working_set_tokens: None,
        }
    }

    #[test]
    fn ample_memory_stops_at_operational_ceiling() {
        let selection = select_context_window(inputs());
        assert_eq!(selection.recommended_max_tokens, 65_536);
        assert_eq!(selection.effective_tokens, 65_536);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::OperationalCeiling
        );
    }

    #[test]
    fn memory_limit_rounds_down_to_1024_token_quantum() {
        let mut inputs = inputs();
        inputs.host_memory = Some(HostMemoryStatus {
            total_bytes: 20_000_000,
            available_bytes: 15_000_000,
        });
        inputs.kv_bytes_per_token = Some(1_000);
        let selection = select_context_window(inputs);
        assert_eq!(selection.recommended_max_tokens, 10_240);
        assert_eq!(selection.effective_tokens, 10_240);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::AvailableMemory
        );
    }

    #[test]
    fn automatic_selection_has_an_8192_token_floor() {
        let mut inputs = inputs();
        inputs.host_memory = Some(HostMemoryStatus {
            total_bytes: 8_192,
            available_bytes: 4_096,
        });
        inputs.kv_bytes_per_token = Some(4_096);
        let selection = select_context_window(inputs);
        assert_eq!(selection.recommended_max_tokens, 8_192);
        assert_eq!(selection.memory_safe_max_tokens, Some(0));
        assert_eq!(selection.effective_tokens, 8_192);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::MinimumOperationalEnvelope
        );
    }

    #[test]
    fn zero_available_memory_is_a_real_pressure_sample() {
        let mut inputs = inputs();
        inputs.host_memory = Some(HostMemoryStatus {
            total_bytes: 8 * 1024 * 1024 * 1024,
            available_bytes: 0,
        });
        let selection = select_context_window(inputs);
        assert_eq!(selection.available_memory_bytes, Some(0));
        assert_eq!(selection.memory_safe_max_tokens, Some(0));
        assert_eq!(selection.recommended_max_tokens, 8_192);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::MinimumOperationalEnvelope
        );
    }

    #[test]
    fn unknown_memory_or_kv_uses_bounded_fallback() {
        let mut missing_memory = inputs();
        missing_memory.host_memory = None;
        assert_eq!(
            select_context_window(missing_memory).effective_tokens,
            UNKNOWN_TELEMETRY_FALLBACK_TOKENS
        );

        let mut missing_kv = inputs();
        missing_kv.kv_bytes_per_token = None;
        let selection = select_context_window(missing_kv);
        assert_eq!(selection.recommended_max_tokens, 8_192);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::UnknownTelemetryFallback
        );
    }

    #[test]
    fn model_server_and_operator_limits_can_select_below_auto_floor() {
        let mut model_limited = inputs();
        model_limited.native_context_tokens = 3_072;
        let selection = select_context_window(model_limited);
        assert_eq!(selection.effective_tokens, 3_072);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::ModelMaximum
        );

        let mut server_limited = inputs();
        server_limited.server_context_tokens = 2_048;
        let selection = select_context_window(server_limited);
        assert_eq!(selection.effective_tokens, 2_048);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::ServerContextMaximum
        );

        let mut operator_limited = inputs();
        operator_limited.configured_max_tokens = Some(1_024);
        let selection = select_context_window(operator_limited);
        assert_eq!(selection.effective_tokens, 1_024);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::ConfiguredMaximum
        );
    }

    #[test]
    fn configured_max_is_preserved_and_wins_ties() {
        let mut inputs = inputs();
        inputs.configured_max_tokens = Some(16_384);
        inputs.native_context_tokens = 16_384;
        let selection = select_context_window(inputs);
        assert_eq!(selection.effective_tokens, 16_384);
        assert_eq!(selection.configured_max_tokens, Some(16_384));
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::ConfiguredMaximum
        );
    }

    #[test]
    fn resident_capacity_is_diagnostic_only() {
        let mut inputs = inputs();
        inputs.resident_capacity_tokens = Some(2_048);
        let selection = select_context_window(inputs);
        assert_eq!(selection.effective_tokens, 65_536);
        assert_eq!(selection.resident_capacity_tokens, Some(2_048));
    }

    #[test]
    fn qwen_4b_mac_class_stays_inside_the_validated_agent_envelope() {
        const GIB: u64 = 1_073_741_824;
        let mut constrained = inputs();
        constrained.native_context_tokens = 40_960;
        constrained.validated_context_tokens = 8_192;
        constrained.host_memory = Some(HostMemoryStatus {
            total_bytes: 16 * GIB,
            available_bytes: 52 * GIB / 10,
        });
        constrained.kv_bytes_per_token = Some(294_912);
        let selection = select_context_window(constrained);
        assert_eq!(selection.recommended_max_tokens, 12_288);
        assert_eq!(selection.effective_tokens, 8_192);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::ValidatedAgentMaximum
        );

        let mut ample = constrained;
        ample.host_memory = Some(HostMemoryStatus {
            total_bytes: 64 * GIB,
            available_bytes: 48 * GIB,
        });
        let selection = select_context_window(ample);
        assert_eq!(selection.recommended_max_tokens, 65_536);
        assert_eq!(selection.effective_tokens, 8_192);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::ValidatedAgentMaximum
        );
    }

    #[test]
    fn qwen_4b_paging_exposes_16k_logical_context_with_an_8k_active_set() {
        const GIB: u64 = 1_073_741_824;
        let mut inputs = inputs();
        inputs.native_context_tokens = 40_960;
        inputs.validated_context_tokens = 8_192;
        inputs.host_memory = Some(HostMemoryStatus {
            total_bytes: 16 * GIB,
            available_bytes: 52 * GIB / 10,
        });
        inputs.kv_bytes_per_token = Some(294_912);
        inputs.paged_target_tokens = Some(16_384);
        inputs.paged_working_set_tokens = Some(8_000);

        let selection = select_context_window(inputs);
        assert_eq!(selection.effective_tokens, 16_384);
        assert_eq!(selection.validated_max_tokens, 8_192);
        assert_eq!(selection.paged_target_tokens, Some(16_384));
        assert_eq!(selection.paged_working_set_tokens, Some(8_000));
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::PagedModelTarget
        );

        inputs.configured_max_tokens = Some(4_096);
        let configured = select_context_window(inputs);
        assert_eq!(configured.effective_tokens, 4_096);
        assert_eq!(configured.paged_working_set_tokens, Some(4_096));
        assert_eq!(
            configured.limiting_factor,
            ContextWindowLimitingFactor::ConfiguredMaximum
        );

        inputs.configured_max_tokens = None;
        inputs.paged_working_set_tokens = Some(8_193);
        let invalid = select_context_window(inputs);
        assert_eq!(invalid.effective_tokens, 8_192);
        assert_eq!(invalid.paged_target_tokens, None);
        assert_eq!(invalid.paged_working_set_tokens, None);
        assert_eq!(
            invalid.limiting_factor,
            ContextWindowLimitingFactor::ValidatedAgentMaximum
        );
    }

    #[test]
    fn automatic_kv_budget_is_shared_across_active_and_retained_owners() {
        let mut single = inputs();
        single.host_memory = Some(HostMemoryStatus {
            total_bytes: 32_000_000_000,
            available_bytes: 30_000_000_000,
        });
        single.kv_bytes_per_token = Some(1_000_000);
        single.validated_context_tokens = 65_536;
        single.kv_owner_slots = 1;
        let one = select_context_window(single);

        let mut dual = single;
        dual.kv_owner_slots = 2;
        let two = select_context_window(dual);
        assert_eq!(one.recommended_max_tokens, 20_480);
        assert_eq!(two.recommended_max_tokens, 10_240);
        assert!(
            u64::from(two.recommended_max_tokens)
                * u64::from(dual.kv_owner_slots)
                * dual.kv_bytes_per_token.expect("KV cost")
                <= dual.host_memory.expect("memory sample").available_bytes
                    * AVAILABLE_MEMORY_PERCENT
                    / 100
        );
    }

    #[test]
    fn low_memory_multi_owner_floor_is_reported_as_a_shortfall_not_as_safe() {
        const GIB: u64 = 1_073_741_824;
        let mut inputs = inputs();
        inputs.native_context_tokens = 40_960;
        inputs.validated_context_tokens = 8_192;
        inputs.host_memory = Some(HostMemoryStatus {
            total_bytes: 16 * GIB,
            available_bytes: 52 * GIB / 10,
        });
        inputs.kv_bytes_per_token = Some(294_912);
        inputs.kv_owner_slots = 4;

        let selection = select_context_window(inputs);
        assert_eq!(selection.memory_safe_max_tokens, Some(3_072));
        assert_eq!(selection.recommended_max_tokens, 8_192);
        assert_eq!(selection.effective_tokens, 8_192);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::ValidatedAgentMaximum
        );
    }

    #[test]
    fn configured_max_parser_rejects_invalid_and_zero_values() {
        assert_eq!(parse_configured_max(" 32768 "), Some(32_768));
        assert_eq!(parse_configured_max("0"), None);
        assert_eq!(parse_configured_max("-1"), None);
        assert_eq!(parse_configured_max("many"), None);
    }

    #[test]
    fn selection_serializes_with_stable_diagnostic_field_names() {
        let selection = select_context_window(inputs());
        let value = serde_json::to_value(selection).expect("selection serializes");
        assert_eq!(value["mode"], "auto");
        assert_eq!(value["effective_tokens"], 65_536);
        assert_eq!(value["recommended_max_tokens"], 65_536);
        assert_eq!(value["memory_safe_max_tokens"], 183_296);
        assert_eq!(value["model_max_tokens"], 131_072);
        assert_eq!(value["validated_max_tokens"], 131_072);
        assert_eq!(value["kv_owner_slots"], 1);
        assert_eq!(value["limiting_factor"], "operational_ceiling");
        assert!(value.get("available_memory_bytes").is_some());
        assert!(value.get("kv_bytes_per_token").is_some());
        assert!(value.get("resident_capacity_tokens").is_some());
        assert!(value.get("configured_max_tokens").is_none());
        assert!(value.get("paged_target_tokens").is_none());
        assert!(value.get("paged_working_set_tokens").is_none());
    }
}
