//! Adaptive context-window policy for agent sessions.
//!
//! This module deliberately separates policy from telemetry collection. Callers
//! take one live host-memory snapshot, derive the model's conservative host KV
//! cost, and pass both into [`select_context_window`]. The pure function is then
//! deterministic and cheap to test. A resident GPU capacity is retained for
//! diagnostics only: exceeding it can make a request slower by falling back to
//! the host cache, but does not make the model context incorrect.
//! Model-specific paged targets are a separate logical-memory policy: they are
//! valid only when the caller also supplies the much smaller host-enforced
//! paged working set used on every actual model request.

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
    /// Server-side prompt ceiling (independent of the generation-token ceiling).
    pub server_max_prompt_tokens: u32,
    /// A fresh, non-cached physical-memory snapshot.
    pub host_memory: Option<HostMemoryStatus>,
    /// Conservative host-cache cost computed from the model's actual KV shape.
    pub kv_bytes_per_token: Option<u64>,
    /// Optional resident GPU/Metal capacity. This is telemetry, never a hard cap.
    pub resident_capacity_tokens: Option<u32>,
    /// Optional operator cap. API callers may supply this directly; the normal
    /// environment-backed value comes from [`configured_agent_context_max`].
    pub configured_max_tokens: Option<u32>,
    /// Model-specific logical target used only when host-owned context paging
    /// keeps the active prompt bounded independently. This may be higher than
    /// the conservative host-KV recommendation because the paged working set,
    /// not the whole logical task context, is resident on each model step.
    pub paged_target_tokens: Option<u32>,
    /// Maximum prompt + output + safety envelope retained on each paged step.
    /// Diagnostic only; the paging runtime remains the enforcing authority.
    pub paged_working_set_tokens: Option<u32>,
}

/// The bound that ultimately selected `effective_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ContextWindowLimitingFactor {
    ConfiguredMaximum,
    ModelMaximum,
    ServerPromptMaximum,
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
    /// server, and explicit operator caps are applied. For a model-specific
    /// paged target this is the logical target, while
    /// `paged_working_set_tokens` reports the smaller resident request envelope.
    /// Under severe non-paged memory pressure this is the 8K minimum operational
    /// envelope rather than a claim that 70% of the current available-memory
    /// sample can hold 8K; the runtime allocation guard remains authoritative.
    pub safe_max_tokens: u32,
    pub model_max_tokens: u32,
    pub limiting_factor: ContextWindowLimitingFactor,
    pub available_memory_bytes: Option<u64>,
    pub kv_bytes_per_token: Option<u64>,
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
/// memory. The automatic budget uses 70% of *available* RAM, rounds down to a
/// 1,024-token quantum, and never grows above 65,536 tokens by default. An 8,192
/// minimum keeps the context-paging reserves and mandatory capsule useful; a smaller native/server or
/// explicit configured cap still wins exactly.
pub(crate) fn select_context_window(inputs: ContextWindowInputs) -> ContextWindowSelection {
    let host_memory = inputs.host_memory.filter(|memory| {
        memory.total_bytes > 0
            && memory.available_bytes > 0
            && memory.available_bytes <= memory.total_bytes
    });
    let available_memory_bytes = host_memory.map(|memory| memory.available_bytes);
    let kv_bytes_per_token = inputs.kv_bytes_per_token.filter(|bytes| *bytes > 0);
    let configured_max_tokens = inputs.configured_max_tokens.filter(|tokens| *tokens > 0);
    let paged_target_tokens = inputs.paged_target_tokens.filter(|tokens| *tokens > 0);
    let paged_working_set_tokens = inputs.paged_working_set_tokens.filter(|tokens| *tokens > 0);

    let (memory_safe_tokens, memory_factor) = match (available_memory_bytes, kv_bytes_per_token) {
        (Some(available), Some(bytes_per_token)) => {
            let memory_budget = available.saturating_mul(AVAILABLE_MEMORY_PERCENT) / 100;
            let raw_tokens = memory_budget / bytes_per_token;
            let quantized_tokens = raw_tokens
                .saturating_div(CONTEXT_QUANTUM_TOKENS)
                .saturating_mul(CONTEXT_QUANTUM_TOKENS)
                .min(u64::from(u32::MAX)) as u32;
            let memory_safe_tokens = quantized_tokens.max(MINIMUM_AUTO_CONTEXT_TOKENS);
            if quantized_tokens < MINIMUM_AUTO_CONTEXT_TOKENS {
                (
                    MINIMUM_AUTO_CONTEXT_TOKENS,
                    ContextWindowLimitingFactor::MinimumOperationalEnvelope,
                )
            } else if memory_safe_tokens >= DEFAULT_OPERATIONAL_CEILING_TOKENS {
                (
                    DEFAULT_OPERATIONAL_CEILING_TOKENS,
                    ContextWindowLimitingFactor::OperationalCeiling,
                )
            } else {
                (
                    memory_safe_tokens,
                    ContextWindowLimitingFactor::AvailableMemory,
                )
            }
        }
        _ => (
            UNKNOWN_TELEMETRY_FALLBACK_TOKENS,
            ContextWindowLimitingFactor::UnknownTelemetryFallback,
        ),
    };
    let (safe_max_tokens, automatic_factor) = paged_target_tokens
        .map_or((memory_safe_tokens, memory_factor), |target| {
            (target, ContextWindowLimitingFactor::PagedModelTarget)
        });

    // Tie order is intentional: an explicit cap is the most useful explanation,
    // followed by immutable model/server limits, then the automatic recommendation.
    let mut effective_tokens = u32::MAX;
    let mut limiting_factor = automatic_factor;
    if let Some(configured) = configured_max_tokens {
        effective_tokens = configured;
        limiting_factor = ContextWindowLimitingFactor::ConfiguredMaximum;
    }
    for (tokens, factor) in [
        (
            inputs.native_context_tokens,
            ContextWindowLimitingFactor::ModelMaximum,
        ),
        (
            inputs.server_max_prompt_tokens,
            ContextWindowLimitingFactor::ServerPromptMaximum,
        ),
        (safe_max_tokens, automatic_factor),
    ] {
        if tokens < effective_tokens {
            effective_tokens = tokens;
            limiting_factor = factor;
        }
    }

    let paged_working_set_tokens =
        paged_working_set_tokens.map(|tokens| tokens.min(effective_tokens));

    ContextWindowSelection {
        mode: ContextWindowMode::Auto,
        effective_tokens,
        safe_max_tokens,
        model_max_tokens: inputs.native_context_tokens,
        limiting_factor,
        available_memory_bytes,
        kv_bytes_per_token,
        resident_capacity_tokens: inputs.resident_capacity_tokens,
        configured_max_tokens,
        paged_target_tokens,
        paged_working_set_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> ContextWindowInputs {
        ContextWindowInputs {
            native_context_tokens: 131_072,
            server_max_prompt_tokens: 131_072,
            host_memory: Some(HostMemoryStatus {
                total_bytes: 32 * 1024 * 1024 * 1024,
                available_bytes: 16 * 1024 * 1024 * 1024,
            }),
            kv_bytes_per_token: Some(64 * 1024),
            resident_capacity_tokens: None,
            configured_max_tokens: None,
            paged_target_tokens: None,
            paged_working_set_tokens: None,
        }
    }

    #[test]
    fn ample_memory_stops_at_operational_ceiling() {
        let selection = select_context_window(inputs());
        assert_eq!(selection.safe_max_tokens, 65_536);
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
        assert_eq!(selection.safe_max_tokens, 10_240);
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
        assert_eq!(selection.safe_max_tokens, 8_192);
        assert_eq!(selection.effective_tokens, 8_192);
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
        assert_eq!(selection.safe_max_tokens, 8_192);
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
        server_limited.server_max_prompt_tokens = 2_048;
        let selection = select_context_window(server_limited);
        assert_eq!(selection.effective_tokens, 2_048);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::ServerPromptMaximum
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
    fn qwen_4b_windows_class_grows_beyond_8k_and_ample_ram_reaches_native_max() {
        const GIB: u64 = 1_073_741_824;
        let mut constrained = inputs();
        constrained.native_context_tokens = 40_960;
        constrained.host_memory = Some(HostMemoryStatus {
            total_bytes: 16 * GIB,
            available_bytes: 52 * GIB / 10,
        });
        constrained.kv_bytes_per_token = Some(294_912);
        let selection = select_context_window(constrained);
        assert_eq!(selection.safe_max_tokens, 12_288);
        assert_eq!(selection.effective_tokens, 12_288);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::AvailableMemory
        );

        let mut ample = constrained;
        ample.host_memory = Some(HostMemoryStatus {
            total_bytes: 64 * GIB,
            available_bytes: 48 * GIB,
        });
        let selection = select_context_window(ample);
        assert_eq!(selection.safe_max_tokens, 65_536);
        assert_eq!(selection.effective_tokens, 40_960);
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::ModelMaximum
        );
    }

    #[test]
    fn qwen_4b_paged_target_reaches_16k_without_expanding_the_active_working_set() {
        let mut inputs = inputs();
        inputs.host_memory = Some(HostMemoryStatus {
            total_bytes: 16 * 1024 * 1024 * 1024,
            available_bytes: 2 * 1024 * 1024 * 1024,
        });
        inputs.kv_bytes_per_token = Some(294_912);
        inputs.native_context_tokens = 40_960;
        inputs.paged_target_tokens = Some(16_384);
        inputs.paged_working_set_tokens = Some(8_000);

        let selection = select_context_window(inputs);
        assert_eq!(selection.effective_tokens, 16_384);
        assert_eq!(selection.safe_max_tokens, 16_384);
        assert_eq!(selection.paged_target_tokens, Some(16_384));
        assert_eq!(selection.paged_working_set_tokens, Some(8_000));
        assert_eq!(
            selection.limiting_factor,
            ContextWindowLimitingFactor::PagedModelTarget
        );

        inputs.server_max_prompt_tokens = 12_288;
        let server_limited = select_context_window(inputs);
        assert_eq!(server_limited.effective_tokens, 12_288);
        assert_eq!(
            server_limited.limiting_factor,
            ContextWindowLimitingFactor::ServerPromptMaximum
        );

        inputs.server_max_prompt_tokens = 131_072;
        inputs.configured_max_tokens = Some(4_096);
        let configured = select_context_window(inputs);
        assert_eq!(configured.effective_tokens, 4_096);
        assert_eq!(configured.paged_working_set_tokens, Some(4_096));
        assert_eq!(
            configured.limiting_factor,
            ContextWindowLimitingFactor::ConfiguredMaximum
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
        assert_eq!(value["safe_max_tokens"], 65_536);
        assert_eq!(value["model_max_tokens"], 131_072);
        assert_eq!(value["limiting_factor"], "operational_ceiling");
        assert!(value.get("available_memory_bytes").is_some());
        assert!(value.get("kv_bytes_per_token").is_some());
        assert!(value.get("resident_capacity_tokens").is_some());
        assert!(value.get("configured_max_tokens").is_none());
        assert!(value.get("paged_target_tokens").is_none());
        assert!(value.get("paged_working_set_tokens").is_none());
    }
}
