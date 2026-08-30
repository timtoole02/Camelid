//! Serving orchestration for lossless Llama-3.2-3B EAGLE-3 speculation.
//!
//! The target model remains authoritative for every emitted token. A cheap
//! suffix chain gets first refusal (and can use all 16 verifier rows); misses
//! fall through to the measured N8/K4/X4 learned tree. Target activation rows
//! from successful suffix rounds are buffered and applied to the learned head
//! only when that fallback is actually needed.

use std::sync::Arc;

use crate::{
    eagle3::{Eagle3DraftModel, TARGET_LAYER_INPUT_IDS},
    eagle3_runtime::{Eagle3AuthoritativeCatchup, Eagle3Drafter, Eagle3DynamicFrontierConfig},
    error::{BackendError, Result},
    inference::{
        spec_tree::TREE_MAX_NODES,
        speculative::accepted_draft_prefix,
        suffix_decoding::{SuffixAdmissionEvidence, SuffixDecodingDrafter},
        LlamaForwardTimings, LlamaInferenceSession, LlamaLoadedWeights,
    },
};

/// The packed Metal verifier has a hard 16-row ceiling, root included.
pub const MAX_DRAFT_TOKENS: usize = TREE_MAX_NODES - 1;
/// Serving refuses a logical prompt + output budget beyond the training and
/// verified runtime envelope instead of silently dropping to a different lane.
pub const MAX_LOGICAL_TOKENS: usize = 2_048;

const DEFAULT_DYNAMIC_VERIFY_NODES: usize = 8;
const DEFAULT_DYNAMIC_TOP_K: usize = 4;
const DEFAULT_DYNAMIC_EXPANSIONS: usize = 4;

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::InvalidModelMetadata(message.into())
}

pub fn validate_logical_budget(prompt_tokens: usize, max_tokens: usize) -> Result<usize> {
    let logical_tokens = prompt_tokens
        .checked_add(max_tokens)
        .ok_or_else(|| invalid("EAGLE-3 logical token budget overflow"))?;
    if logical_tokens > MAX_LOGICAL_TOKENS {
        return Err(invalid(format!(
            "EAGLE-3 serving is fail-closed above {MAX_LOGICAL_TOKENS} logical tokens; prompt {prompt_tokens} + max_tokens {max_tokens} = {logical_tokens}"
        )));
    }
    Ok(logical_tokens)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eagle3ServingConfig {
    pub draft_tokens: usize,
    pub dynamic_verify_nodes: usize,
    pub dynamic_top_k: usize,
    pub dynamic_expansions: usize,
}

impl Eagle3ServingConfig {
    pub fn new(draft_tokens: usize) -> Result<Self> {
        if !(1..=MAX_DRAFT_TOKENS).contains(&draft_tokens) {
            return Err(invalid(format!(
                "EAGLE-3 serving draft width must be in 1..={MAX_DRAFT_TOKENS}, got {draft_tokens}"
            )));
        }
        Ok(Self {
            draft_tokens,
            // N8/K4/X4 is the receipted general-chat learned-tree point. The
            // suffix lane can still consume the full physical width 16.
            dynamic_verify_nodes: DEFAULT_DYNAMIC_VERIFY_NODES,
            dynamic_top_k: DEFAULT_DYNAMIC_TOP_K,
            dynamic_expansions: DEFAULT_DYNAMIC_EXPANSIONS,
        })
    }
}

pub struct Eagle3ServingBootstrap {
    pub first_token: u32,
    pub timings: LlamaForwardTimings,
}

pub struct Eagle3ServingRound {
    pub emitted: Vec<u32>,
    pub offered: usize,
    pub verify_nodes: usize,
    pub suffix: bool,
    pub suffix_evidence: SuffixAdmissionEvidence,
    pub timings: LlamaForwardTimings,
}

/// Per-request mutable draft state. The checkpoint itself is shared across
/// requests; only the small one-layer KV state is private to a generation.
pub struct Eagle3ServingState {
    checkpoint: Option<Arc<Eagle3DraftModel>>,
    drafter: Option<Eagle3Drafter>,
    suffix: SuffixDecodingDrafter,
    pending_suffix_head: Eagle3AuthoritativeCatchup,
    config: Eagle3ServingConfig,
}

impl Eagle3ServingState {
    pub fn new(checkpoint: Arc<Eagle3DraftModel>, config: Eagle3ServingConfig) -> Self {
        Self {
            checkpoint: Some(checkpoint),
            drafter: None,
            suffix: SuffixDecodingDrafter::default(),
            pending_suffix_head: Eagle3AuthoritativeCatchup::default(),
            config,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.drafter.is_some()
    }

    /// Capture the real target prompt activations, obtain the first target
    /// greedy token, upload the head, and seed its authoritative cache.
    pub fn bootstrap(
        &mut self,
        session: &mut LlamaInferenceSession,
        target_weights: &Arc<LlamaLoadedWeights>,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> Result<Eagle3ServingBootstrap> {
        if self.is_initialized() {
            return Err(invalid("EAGLE-3 serving bootstrap may only run once"));
        }
        if prompt_tokens.len() < 3 {
            return Err(invalid(format!(
                "EAGLE-3 resident activation capture needs at least 3 prompt tokens, got {}",
                prompt_tokens.len()
            )));
        }
        if !session.prewarm_resident_weights() {
            return Err(invalid(
                "EAGLE-3 serving requires the resident Metal target lane",
            ));
        }
        // Target and head share Metal's serial queue. A precommitted target
        // decode graph must never sit ahead of a head update.
        session.set_resident_encode_ahead_enabled(false);
        let prompt = session
            .forward_greedy_resident_prefill_with_layer_inputs(
                prompt_tokens,
                &TARGET_LAYER_INPUT_IDS,
            )?
            .ok_or_else(|| {
                invalid("resident Metal prompt prefill with EAGLE-3 capture is unavailable")
            })?;
        let first_token = *prompt
            .predictions
            .last()
            .ok_or_else(|| invalid("EAGLE-3 target prompt produced no greedy prediction"))?;
        let head_capacity = prompt_tokens
            .len()
            .checked_add(max_tokens)
            .and_then(|positions| positions.checked_add(self.config.draft_tokens + 1))
            .ok_or_else(|| invalid("EAGLE-3 head cache capacity overflow"))?;
        let checkpoint = self
            .checkpoint
            .take()
            .ok_or_else(|| invalid("EAGLE-3 checkpoint was consumed before bootstrap"))?;
        let mut drafter = Eagle3Drafter::new(checkpoint.as_ref(), head_capacity)?;
        drafter.seed_prompt(
            target_weights,
            prompt_tokens,
            first_token,
            &prompt.layer_inputs,
        )?;
        self.drafter = Some(drafter);
        Ok(Eagle3ServingBootstrap {
            first_token,
            timings: prompt.timings,
        })
    }

    /// Run one target-authoritative serving round. A `None` result means there
    /// is no context room left; callers should finish rather than changing
    /// execution lanes after EAGLE has made target KV GPU-authoritative.
    pub fn run_round(
        &mut self,
        session: &mut LlamaInferenceSession,
        target_weights: &Arc<LlamaLoadedWeights>,
        history: &[u32],
        remaining_output: usize,
    ) -> Result<Option<Eagle3ServingRound>> {
        let anchor = *history
            .last()
            .ok_or_else(|| invalid("EAGLE-3 serving history is empty"))?;
        let drafter = self
            .drafter
            .as_mut()
            .ok_or_else(|| invalid("EAGLE-3 serving round ran before bootstrap"))?;
        if remaining_output == 0 {
            return Ok(None);
        }
        let context_room = session.remaining_context();
        if context_room == 0 {
            return Ok(None);
        }
        let budget = self
            .config
            .draft_tokens
            .min(remaining_output.saturating_sub(1))
            .min(context_room.saturating_sub(1));

        // The last requested token has no useful successor to draft. Keep it
        // on the resident target; no head update is observable after it.
        if budget == 0 {
            let (token, _sample_us) = session
                .generate_next_token_greedy_resident(anchor)?
                .ok_or_else(|| invalid("resident Metal target became unavailable"))?;
            return Ok(Some(Eagle3ServingRound {
                emitted: vec![token],
                offered: 0,
                verify_nodes: 1,
                suffix: false,
                suffix_evidence: SuffixAdmissionEvidence::default(),
                timings: LlamaForwardTimings::default(),
            }));
        }

        let target_before = session.kv_position();
        let suffix_node_budget = (budget + 1).min(context_room).min(TREE_MAX_NODES);
        let suffix_proposal =
            self.suffix
                .draft_confident_chain(history, anchor, suffix_node_budget, budget);
        let suffix_evidence = suffix_proposal.evidence;
        let suffix_drafts = suffix_proposal.tokens;
        tracing::debug!(
            raw_depth = suffix_evidence.raw_depth,
            confident_depth = suffix_evidence.confident_depth,
            root_match_len = suffix_evidence.root_match_len,
            root_support = suffix_evidence.root_support,
            root_branch_count = suffix_evidence.root_branch_count,
            expected_accepted_q16 = suffix_evidence.expected_accepted_q16,
            terminal_survival_q16 = suffix_evidence.terminal_survival_q16,
            admitted = suffix_evidence.admitted,
            "EAGLE-3 suffix confidence admission"
        );

        let round = if !suffix_drafts.is_empty() {
            let verified = session
                .verify_drafts_metal_with_layer_inputs(
                    anchor,
                    &suffix_drafts,
                    &TARGET_LAYER_INPUT_IDS,
                )?
                .ok_or_else(|| {
                    invalid(format!(
                        "resident Metal suffix verify became unavailable at target position {target_before}"
                    ))
                })?;
            if verified.predictions.len() != suffix_drafts.len() + 1 {
                return Err(invalid(format!(
                    "EAGLE-3 suffix target returned {} predictions for {} drafts",
                    verified.predictions.len(),
                    suffix_drafts.len()
                )));
            }
            let accepted = accepted_draft_prefix(&suffix_drafts, &verified.predictions);
            let emitted = verified.predictions[..=accepted].to_vec();
            self.pending_suffix_head
                .push(&verified.layer_inputs, &emitted)?;
            Eagle3ServingRound {
                emitted,
                offered: suffix_drafts.len(),
                verify_nodes: suffix_drafts.len() + 1,
                suffix: true,
                suffix_evidence,
                timings: verified.timings,
            }
        } else {
            if !self.pending_suffix_head.is_empty() {
                drafter
                    .accept_authoritative_catchup(target_weights, &mut self.pending_suffix_head)?;
            }
            if drafter.filled() != session.kv_position() {
                return Err(invalid(format!(
                    "EAGLE-3 catch-up watermark diverged: head={} target={}",
                    drafter.filled(),
                    session.kv_position()
                )));
            }

            let node_budget = self
                .config
                .dynamic_verify_nodes
                .min(context_room)
                .min(budget + 1);
            if node_budget < 2 {
                return Err(invalid(format!(
                    "EAGLE-3 dynamic verifier has only {node_budget} rows"
                )));
            }
            let lattice_nodes = self
                .config
                .dynamic_top_k
                .checked_mul(self.config.dynamic_expansions)
                .and_then(|nodes| nodes.checked_add(1))
                .map(|nodes| nodes.max(node_budget))
                .ok_or_else(|| invalid("EAGLE-3 dynamic lattice budget overflow"))?;
            let frontier = drafter.draft_dynamic_frontier(
                target_weights,
                anchor,
                Eagle3DynamicFrontierConfig {
                    max_verify_nodes: node_budget,
                    max_lattice_nodes: lattice_nodes,
                    max_depth: budget,
                    candidates_per_parent: self.config.dynamic_top_k,
                    max_head_expansions: self.config.dynamic_expansions,
                    adaptive_branching: false,
                },
            )?;
            let forest = frontier.finish()?;
            let actual_nodes = forest.scored.tree.nodes();
            if !(2..=node_budget).contains(&actual_nodes) {
                return Err(invalid(format!(
                    "dynamic EAGLE-3 forest produced {actual_nodes} rows for budget {node_budget}"
                )));
            }
            let verified = session
                .verify_tree_metal_with_layer_inputs(
                    &forest.scored.tree,
                    &TARGET_LAYER_INPUT_IDS,
                )?
                .ok_or_else(|| {
                    invalid(format!(
                        "resident Metal EAGLE-3 tree verify became unavailable at target position {target_before}"
                    ))
                })?;
            if verified.predictions.len() != actual_nodes {
                return Err(invalid(format!(
                    "EAGLE-3 tree target returned {} predictions for {actual_nodes} rows",
                    verified.predictions.len()
                )));
            }
            let acceptance = forest.accept_target_predictions(&verified.predictions)?;
            if acceptance.capture_rows.len() != acceptance.emitted_tokens.len() {
                return Err(invalid(format!(
                    "EAGLE-3 accepted tree capture/token lengths diverged: {}/{}",
                    acceptance.capture_rows.len(),
                    acceptance.emitted_tokens.len()
                )));
            }
            drafter.accept_authoritative_forest(
                target_weights,
                &verified.layer_inputs,
                &acceptance,
            )?;
            Eagle3ServingRound {
                emitted: acceptance.emitted_tokens,
                offered: actual_nodes - 1,
                verify_nodes: actual_nodes,
                suffix: false,
                suffix_evidence,
                timings: verified.timings,
            }
        };

        if session.kv_position() != target_before + round.emitted.len() {
            return Err(invalid(format!(
                "EAGLE-3 target watermark advanced {} rows for {} emitted tokens",
                session.kv_position().saturating_sub(target_before),
                round.emitted.len()
            )));
        }
        let effective_head = self
            .pending_suffix_head
            .effective_filled(drafter.filled())?;
        if effective_head != session.kv_position() {
            return Err(invalid(format!(
                "EAGLE-3 cache watermarks diverged: materialized_head={} pending_head={} target={}",
                drafter.filled(),
                self.pending_suffix_head.pending_rows(),
                session.kv_position()
            )));
        }
        Ok(Some(round))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_width_is_strict_and_keeps_dynamic_tree_at_receipted_point() {
        assert!(Eagle3ServingConfig::new(0).is_err());
        assert!(Eagle3ServingConfig::new(MAX_DRAFT_TOKENS + 1).is_err());
        let config = Eagle3ServingConfig::new(MAX_DRAFT_TOKENS).unwrap();
        assert_eq!(config.draft_tokens + 1, TREE_MAX_NODES);
        assert_eq!(config.dynamic_verify_nodes, 8);
        assert_eq!(config.dynamic_top_k, 4);
        assert_eq!(config.dynamic_expansions, 4);
    }

    #[test]
    fn logical_budget_accepts_boundary_and_fails_closed_above_2048() {
        assert_eq!(validate_logical_budget(1_024, 1_024).unwrap(), 2_048);
        assert!(validate_logical_budget(1_024, 1_025).is_err());
        assert!(validate_logical_budget(usize::MAX, 1).is_err());
    }
}
