//! Typed multi-sequence semantics and the serial reference oracle.
//!
//! This module is deliberately not wired into serving. It defines the identity,
//! compatibility, scatter, and per-row failure contract that later optimized
//! executors must match while routing every Phase 1 row through the unchanged
//! single-session generation primitive.

use std::collections::{BTreeMap, BTreeSet};

use super::{KvDtype, KvLayout, LlamaGenerationStep, LlamaInferenceSession, LlamaSampler};

/// Stable identity of one independently mutable sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BatchSequenceId(u64);

impl BatchSequenceId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// Exact model-artifact identity supplied by the owning load/request layer.
///
/// The caller must derive these bytes from the authoritative loaded artifact
/// identity (the GGUF SHA-256 in the API today). Phase 1 cannot recompute that
/// digest from `LlamaInferenceSession`, which intentionally owns parsed weights
/// rather than the source-file receipt; binding the two remains the loader's
/// responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BatchModelIdentity([u8; 32]);

impl BatchModelIdentity {
    pub(crate) const fn from_sha256(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// Result shape requested from the unchanged serial generation primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum BatchOutputMode {
    GenerationStep,
    GenerationStepWithDiagnostics,
}

impl BatchOutputMode {
    const fn collect_diagnostics(self) -> bool {
        matches!(self, Self::GenerationStepWithDiagnostics)
    }
}

/// Forward/KV properties that every row in one future mathematical batch must share.
///
/// Sampler configuration and allowed-token masks are deliberately row-local:
/// the unchanged forward produces a logits row before either is applied. The
/// backend is executor-owned rather than a row property. A future optimized
/// executor must add any new kernel-specific constraints at its own admission
/// boundary instead of changing these serial semantics retroactively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchCompatibilityKey {
    model: BatchModelIdentity,
    kv_dtype: KvDtype,
    kv_layout: KvLayout,
    max_sequence_length: usize,
    layer_count: usize,
    kv_head_count: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    output_mode: BatchOutputMode,
}

impl BatchCompatibilityKey {
    pub(crate) fn for_session(
        model: BatchModelIdentity,
        session: &LlamaInferenceSession,
        output_mode: BatchOutputMode,
    ) -> Self {
        let plan = &session.kv_cache.plan;
        Self {
            model,
            kv_dtype: session.kv_cache.dtype,
            kv_layout: session.kv_cache.layout,
            max_sequence_length: plan.max_sequence_length,
            layer_count: plan.layer_count,
            kv_head_count: plan.kv_head_count,
            key_head_dim: plan.k_head_dim,
            value_head_dim: plan.v_head_dim,
            output_mode,
        }
    }

    fn matches_session(self, session: &LlamaInferenceSession) -> bool {
        let plan = &session.kv_cache.plan;
        self.kv_dtype == session.kv_cache.dtype
            && self.kv_layout == session.kv_cache.layout
            && self.max_sequence_length == plan.max_sequence_length
            && self.layer_count == plan.layer_count
            && self.kv_head_count == plan.kv_head_count
            && self.key_head_dim == plan.k_head_dim
            && self.value_head_dim == plan.v_head_dim
    }
}

/// One independently owned serial-oracle row.
pub(crate) struct SerialBatchRow {
    sequence_id: BatchSequenceId,
    compatibility: BatchCompatibilityKey,
    expected_position: usize,
    input_token_id: u32,
    token_history: Vec<u32>,
    sampler: LlamaSampler,
    allowed_tokens: Option<Vec<bool>>,
    session: LlamaInferenceSession,
}

impl SerialBatchRow {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        sequence_id: BatchSequenceId,
        compatibility: BatchCompatibilityKey,
        expected_position: usize,
        input_token_id: u32,
        token_history: Vec<u32>,
        sampler: LlamaSampler,
        allowed_tokens: Option<Vec<bool>>,
        session: LlamaInferenceSession,
    ) -> Self {
        Self {
            sequence_id,
            compatibility,
            expected_position,
            input_token_id,
            token_history,
            sampler,
            allowed_tokens,
            session,
        }
    }

    pub(crate) fn session(&self) -> &LlamaInferenceSession {
        &self.session
    }

    pub(crate) fn session_mut(&mut self) -> &mut LlamaInferenceSession {
        &mut self.session
    }

    pub(crate) fn with_expected_position(mut self, expected_position: usize) -> Self {
        self.expected_position = expected_position;
        self
    }

    pub(crate) fn with_compatibility(mut self, compatibility: BatchCompatibilityKey) -> Self {
        self.compatibility = compatibility;
        self
    }

    pub(crate) fn with_allowed_tokens(mut self, allowed_tokens: Vec<bool>) -> Self {
        self.allowed_tokens = Some(allowed_tokens);
        self
    }
}

/// What computation actually served a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchExecutionMode {
    Exclusive,
    CooperativeSerial,
    #[allow(dead_code)]
    TrueBatch,
}

/// A failure tied to one row; siblings remain independently executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BatchRowFailure {
    PositionMismatch {
        expected: usize,
        actual: usize,
    },
    /// The underlying primitive returned an error. The message is retained
    /// without parsing prose into invented categories. A failed primitive may
    /// have advanced its own session; Phase 1 promises sibling isolation, not
    /// rollback of a failed row.
    GenerationFailed {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BatchRowOutcome {
    Generated(Box<LlamaGenerationStep>),
    Failed(BatchRowFailure),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BatchRowResult {
    pub(crate) sequence_id: BatchSequenceId,
    pub(crate) position_before: usize,
    pub(crate) position_after: usize,
    pub(crate) input_token_id: u32,
    pub(crate) execution_mode: BatchExecutionMode,
    pub(crate) outcome: BatchRowOutcome,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BatchExecutionModeCounts {
    pub(crate) exclusive_rows: usize,
    pub(crate) cooperative_serial_rows: usize,
    pub(crate) true_batch_rows: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BatchExecution {
    pub(crate) results: BTreeMap<BatchSequenceId, BatchRowResult>,
    pub(crate) mode_counts: BatchExecutionModeCounts,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum BatchContractError {
    #[error("a serial batch must contain at least one row")]
    EmptyBatch,
    #[error("batch sequence id {0} appears more than once")]
    DuplicateSequence(u64),
    #[error("batch sequence {sequence_id} is incompatible with the first row")]
    IncompatibleRow { sequence_id: u64 },
    #[error("batch sequence {sequence_id} compatibility does not describe its session")]
    SessionCompatibilityMismatch { sequence_id: u64 },
    #[error("batch produced an unexpected sequence result {0}")]
    UnexpectedResult(u64),
    #[error("batch produced sequence result {0} more than once")]
    DuplicateResult(u64),
    #[error("batch did not produce a result for sequence {0}")]
    MissingResult(u64),
}

/// Behavior-neutral reference executor for future optimized batch implementations.
pub(crate) struct SerialBatchOracle;

impl SerialBatchOracle {
    pub(crate) fn execute(
        rows: &mut [SerialBatchRow],
    ) -> Result<BatchExecution, BatchContractError> {
        preflight(rows)?;
        let execution_mode = if rows.len() == 1 {
            BatchExecutionMode::Exclusive
        } else {
            BatchExecutionMode::CooperativeSerial
        };
        let expected_ids = rows.iter().map(|row| row.sequence_id).collect::<Vec<_>>();
        let mut results = Vec::with_capacity(rows.len());

        for row in rows {
            let position_before = row.session.kv_position();
            let outcome = if position_before != row.expected_position {
                BatchRowOutcome::Failed(BatchRowFailure::PositionMismatch {
                    expected: row.expected_position,
                    actual: position_before,
                })
            } else {
                let collect_diagnostics = row.compatibility.output_mode.collect_diagnostics();
                match row.session.generate_next_token_with_history_diagnostics(
                    &[row.input_token_id],
                    row.sampler.clone(),
                    &row.token_history,
                    collect_diagnostics,
                    row.allowed_tokens.as_deref(),
                ) {
                    Ok(step) => BatchRowOutcome::Generated(Box::new(step)),
                    Err(error) => BatchRowOutcome::Failed(BatchRowFailure::GenerationFailed {
                        message: error.to_string(),
                    }),
                }
            };
            results.push(BatchRowResult {
                sequence_id: row.sequence_id,
                position_before,
                position_after: row.session.kv_position(),
                input_token_id: row.input_token_id,
                execution_mode,
                outcome,
            });
        }

        scatter(expected_ids, results, execution_mode)
    }
}

fn preflight(rows: &[SerialBatchRow]) -> Result<(), BatchContractError> {
    let Some(first) = rows.first() else {
        return Err(BatchContractError::EmptyBatch);
    };
    let expected = first.compatibility;
    let mut seen = BTreeSet::new();
    for row in rows {
        if !seen.insert(row.sequence_id) {
            return Err(BatchContractError::DuplicateSequence(row.sequence_id.get()));
        }
        if row.compatibility != expected {
            return Err(BatchContractError::IncompatibleRow {
                sequence_id: row.sequence_id.get(),
            });
        }
        if !row.compatibility.matches_session(&row.session) {
            return Err(BatchContractError::SessionCompatibilityMismatch {
                sequence_id: row.sequence_id.get(),
            });
        }
    }
    Ok(())
}

fn scatter(
    expected_ids: Vec<BatchSequenceId>,
    results: Vec<BatchRowResult>,
    execution_mode: BatchExecutionMode,
) -> Result<BatchExecution, BatchContractError> {
    let expected = expected_ids.into_iter().collect::<BTreeSet<_>>();
    let mut missing = expected.clone();
    let mut scattered = BTreeMap::new();
    for result in results {
        let sequence_id = result.sequence_id;
        if !expected.contains(&sequence_id) {
            return Err(BatchContractError::UnexpectedResult(sequence_id.get()));
        }
        if !missing.remove(&sequence_id) {
            return Err(BatchContractError::DuplicateResult(sequence_id.get()));
        }
        scattered.insert(sequence_id, result);
    }
    if let Some(sequence_id) = missing.first() {
        return Err(BatchContractError::MissingResult(sequence_id.get()));
    }
    let mode_counts = match execution_mode {
        BatchExecutionMode::Exclusive => BatchExecutionModeCounts {
            exclusive_rows: scattered.len(),
            ..BatchExecutionModeCounts::default()
        },
        BatchExecutionMode::CooperativeSerial => BatchExecutionModeCounts {
            cooperative_serial_rows: scattered.len(),
            ..BatchExecutionModeCounts::default()
        },
        BatchExecutionMode::TrueBatch => BatchExecutionModeCounts {
            true_batch_rows: scattered.len(),
            ..BatchExecutionModeCounts::default()
        },
    };
    Ok(BatchExecution {
        results: scattered,
        mode_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed_result(sequence_id: u64) -> BatchRowResult {
        BatchRowResult {
            sequence_id: BatchSequenceId::new(sequence_id),
            position_before: 0,
            position_after: 0,
            input_token_id: sequence_id as u32,
            execution_mode: BatchExecutionMode::CooperativeSerial,
            outcome: BatchRowOutcome::Failed(BatchRowFailure::PositionMismatch {
                expected: 1,
                actual: 0,
            }),
        }
    }

    #[test]
    fn hostile_result_identity_swap_is_refused() {
        let id_10 = BatchSequenceId::new(10);
        let id_20 = BatchSequenceId::new(20);
        let completion_order = vec![failed_result(20), failed_result(10)];

        let correct = scatter(
            vec![id_10, id_20],
            completion_order.clone(),
            BatchExecutionMode::CooperativeSerial,
        )
        .unwrap();
        assert_eq!(correct.results[&id_10].input_token_id, 10);
        assert_eq!(correct.results[&id_20].input_token_id, 20);

        // Hostile ablation: associate completion order with input order by index.
        let index_scattered = [id_10, id_20]
            .into_iter()
            .zip(completion_order)
            .map(|(input_id, result)| (input_id, result.input_token_id))
            .collect::<BTreeMap<_, _>>();
        assert_ne!(index_scattered[&id_10], 10);
        assert_ne!(index_scattered[&id_20], 20);
    }

    #[test]
    fn empty_batch_is_refused() {
        assert_eq!(
            SerialBatchOracle::execute(&mut []),
            Err(BatchContractError::EmptyBatch)
        );
    }

    #[test]
    fn duplicate_and_unexpected_results_are_refused() {
        let id_10 = BatchSequenceId::new(10);
        let id_20 = BatchSequenceId::new(20);
        assert_eq!(
            scatter(
                vec![id_10, id_20],
                vec![failed_result(10), failed_result(10)],
                BatchExecutionMode::CooperativeSerial,
            ),
            Err(BatchContractError::DuplicateResult(10))
        );
        assert_eq!(
            scatter(
                vec![id_10],
                vec![failed_result(30)],
                BatchExecutionMode::Exclusive,
            ),
            Err(BatchContractError::UnexpectedResult(30))
        );
        assert_eq!(
            scatter(
                vec![id_10, id_20],
                vec![failed_result(10)],
                BatchExecutionMode::CooperativeSerial,
            ),
            Err(BatchContractError::MissingResult(20))
        );
    }
}
