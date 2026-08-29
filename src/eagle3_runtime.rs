//! Correctness-first recurrent EAGLE-3 drafting orchestration.
//!
//! The target remains authoritative. This module owns only the learned draft head and its
//! private one-layer KV cache; every proposed token is still checked by the target model's
//! existing greedy speculative verifier before it can be emitted.

use crate::eagle3::{Eagle3DraftModel, HIDDEN_SIZE, TARGET_LAYER_INPUT_IDS};
use crate::error::{BackendError, Result};
use crate::inference::spec_tree::{
    normalize_draft_top_logits, DynamicDraftLattice, PackedForestPlan, ScoredTokenTree,
    TREE_MAX_NODES,
};
use crate::inference::LlamaLoadedWeights;
use crate::metal::{
    Eagle3MetalOutput, Eagle3MetalState, Eagle3MetalWeights, EAGLE3_AUX_WIDTH, EAGLE3_DRAFT_VOCAB,
};
use crate::tensor::CpuTensor;

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::RuntimeShapeMismatch(message.into())
}

fn metal<T>(result: std::result::Result<T, String>) -> Result<T> {
    result.map_err(|message| invalid(format!("EAGLE-3 Metal runtime: {message}")))
}

/// Interleave three target layer-input captures from `[tap][row][hidden]` into the
/// checkpoint encoder's required `[row][low || middle || high]` layout.
pub fn interleave_target_layer_inputs(captures: &[CpuTensor]) -> Result<Vec<f32>> {
    if captures.len() != TARGET_LAYER_INPUT_IDS.len() {
        return Err(invalid(format!(
            "EAGLE-3 requires {} target layer captures, got {}",
            TARGET_LAYER_INPUT_IDS.len(),
            captures.len()
        )));
    }
    let rows = captures[0].dim(0)?;
    for (slot, capture) in captures.iter().enumerate() {
        let capture_rows = capture.dim(0)?;
        let width = capture.dim(1)?;
        if capture_rows != rows || width != HIDDEN_SIZE {
            return Err(invalid(format!(
                "EAGLE-3 target capture {} (layer input {}) has shape {:?}, expected [{rows}, {HIDDEN_SIZE}]",
                slot, TARGET_LAYER_INPUT_IDS[slot], capture.shape.dims
            )));
        }
    }
    let mut features = vec![0.0f32; rows * EAGLE3_AUX_WIDTH];
    for row in 0..rows {
        for (tap, capture) in captures.iter().enumerate() {
            let source = row * HIDDEN_SIZE;
            let destination = row * EAGLE3_AUX_WIDTH + tap * HIDDEN_SIZE;
            features[destination..destination + HIDDEN_SIZE]
                .copy_from_slice(&capture.data[source..source + HIDDEN_SIZE]);
        }
    }
    Ok(features)
}

/// A log-sum-exp reduction over every row in the compact EAGLE draft vocabulary.
///
/// This deliberately has a distinct type instead of accepting a bare `f32` at the dynamic-tree
/// seam.  A reduction over only `Eagle3MetalOutput::top_candidates` is not interchangeable: it
/// would redistribute all omitted probability mass over the retained branches and bias the
/// global frontier toward wide parents.  Construction from [`Eagle3MetalOutput`] also verifies
/// that an experimental reduced-row head did not silently omit part of the vocabulary. A full
/// 32k Q8 head remains valid because approximation changes draft quality, not probability mass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eagle3FullVocabularyLogsumexp(f32);

impl Eagle3FullVocabularyLogsumexp {
    pub fn from_output(output: &Eagle3MetalOutput) -> Result<Self> {
        if output.evaluated_vocab_rows != EAGLE3_DRAFT_VOCAB {
            return Err(invalid(format!(
                "EAGLE-3 dynamic frontier requires all {EAGLE3_DRAFT_VOCAB} draft rows, but the head evaluated {}",
                output.evaluated_vocab_rows
            )));
        }
        let value = output.evaluated_vocab_logsumexp;
        if !value.is_finite() {
            return Err(invalid(format!(
                "EAGLE-3 full-vocabulary logsumexp must be finite, got {value}"
            )));
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<&Eagle3MetalOutput> for Eagle3FullVocabularyLogsumexp {
    type Error = BackendError;

    fn try_from(output: &Eagle3MetalOutput) -> Result<Self> {
        Self::from_output(output)
    }
}

/// Bounds for EAGLE-2-style dynamic candidate expansion.
///
/// `max_lattice_nodes` is allowed to exceed `max_verify_nodes`: the head may explore a wider
/// temporary lattice before the globally strongest connected subset is packed for the target.
/// `max_head_expansions` counts the root distribution, so it is also a direct bound on learned
/// head work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eagle3DynamicFrontierConfig {
    pub max_verify_nodes: usize,
    pub max_lattice_nodes: usize,
    pub max_depth: usize,
    pub candidates_per_parent: usize,
    pub max_head_expansions: usize,
}

impl Default for Eagle3DynamicFrontierConfig {
    fn default() -> Self {
        Self {
            max_verify_nodes: TREE_MAX_NODES,
            max_lattice_nodes: 60,
            max_depth: 6,
            candidates_per_parent: 8,
            max_head_expansions: 8,
        }
    }
}

impl Eagle3DynamicFrontierConfig {
    fn validate(self) -> Result<Self> {
        if self.max_verify_nodes == 0 || self.max_verify_nodes > TREE_MAX_NODES {
            return Err(invalid(format!(
                "EAGLE-3 dynamic verifier nodes must be in 1..={TREE_MAX_NODES}, got {}",
                self.max_verify_nodes
            )));
        }
        if self.max_lattice_nodes < self.max_verify_nodes {
            return Err(invalid(format!(
                "EAGLE-3 lattice node budget {} is smaller than verifier budget {}",
                self.max_lattice_nodes, self.max_verify_nodes
            )));
        }
        if self.max_depth == 0 {
            return Err(invalid("EAGLE-3 dynamic frontier requires non-zero depth"));
        }
        if self.candidates_per_parent == 0 {
            return Err(invalid(
                "EAGLE-3 dynamic frontier requires at least one candidate per parent",
            ));
        }
        if self.max_head_expansions == 0 {
            return Err(invalid(
                "EAGLE-3 dynamic frontier requires at least the root head expansion",
            ));
        }
        Ok(self)
    }
}

/// Minimal cache movement needed to materialize `next_path` from an already-resident
/// ephemeral `current_path`. Paths are stable lattice-node ids in root-first order.
///
/// The EAGLE cache watermark counts only non-root rows: the root distribution is the stable
/// authoritative seed, while every path element after it consumes one private head-cache row.
/// Keeping this arithmetic in a pure helper makes the cursor optimization independently
/// testable and keeps cache mutation out of the probability scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Eagle3PathTransition {
    /// Number of root-first path nodes shared by both paths, including the root.
    shared_nodes: usize,
    /// Number of already-materialized non-root rows that survive rollback.
    retained_rows: usize,
    /// First index in the next root-first path that must be replayed.
    replay_from: usize,
}

fn eagle3_path_transition(
    current_path: &[usize],
    next_path: &[usize],
) -> Result<Eagle3PathTransition> {
    if current_path.is_empty() || next_path.is_empty() {
        return Err(invalid(
            "EAGLE-3 dynamic cursor paths must contain their root",
        ));
    }
    if current_path[0] != next_path[0] {
        return Err(invalid(format!(
            "EAGLE-3 dynamic cursor changed roots from {} to {}",
            current_path[0], next_path[0]
        )));
    }
    let shared_nodes = current_path
        .iter()
        .zip(next_path)
        .take_while(|(left, right)| left == right)
        .count();
    // Equal roots guarantee at least one shared node. Keep the checked form so a future
    // representation change cannot turn a subtraction into an underflow.
    let retained_rows = shared_nodes
        .checked_sub(1)
        .ok_or_else(|| invalid("EAGLE-3 dynamic cursor paths share no root"))?;
    Ok(Eagle3PathTransition {
        shared_nodes,
        retained_rows,
        replay_from: shared_nodes,
    })
}

/// Deterministic host scheduler for a budgeted EAGLE candidate lattice.
///
/// The scheduler owns no model state.  [`Self::next_parent`] names the globally strongest
/// unexpanded node; the caller materializes that node's draft-head path and returns its output
/// through [`Self::record_expansion`].  Every child retains the parent's raw recurrent state,
/// which is enough for [`Eagle3Drafter`] to replay a branch from the stable head watermark
/// without copying the whole one-layer KV cache.
#[derive(Debug, Clone, PartialEq)]
pub struct Eagle3DynamicFrontier {
    config: Eagle3DynamicFrontierConfig,
    lattice: DynamicDraftLattice,
    expanded: Vec<bool>,
    /// Input `g` used when the corresponding node is replayed through the EAGLE cell.  The
    /// root is already represented by `Eagle3Drafter::stable_seed` and therefore has no entry.
    recurrent_g: Vec<Option<Vec<f32>>>,
    head_expansions: usize,
    /// Successful non-root `forward_token` calls used to materialize this lattice. The root
    /// distribution comes from `stable_seed` and therefore costs no call here.
    materialized_head_forwards: usize,
}

/// One measured target-verifier cost point.  Units are arbitrary but must be consistent across
/// the table (microseconds is convenient).  Keeping the table caller-supplied lets mini2 choose
/// width 8 when its K-quant k4/k8 kernels are flat without baking one machine's timing into the
/// checkpoint or the generic scheduler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eagle3VerifierBudgetCost {
    pub max_nodes: usize,
    pub round_cost: f64,
}

/// Hardware-aware, confidence-aware target admission decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Eagle3ForestSelection {
    pub forest: Eagle3DraftForest,
    pub admitted_node_budget: usize,
    pub estimated_emitted_tokens: f64,
    /// `estimated_emitted_tokens / round_cost`; larger is better.
    pub estimated_tokens_per_cost: f64,
}

impl Eagle3DynamicFrontier {
    pub fn new(anchor: u32, config: Eagle3DynamicFrontierConfig) -> Result<Self> {
        Ok(Self {
            config: config.validate()?,
            lattice: DynamicDraftLattice::new(anchor),
            expanded: vec![false],
            recurrent_g: vec![None],
            head_expansions: 0,
            materialized_head_forwards: 0,
        })
    }

    pub fn config(&self) -> Eagle3DynamicFrontierConfig {
        self.config
    }

    pub fn lattice(&self) -> &DynamicDraftLattice {
        &self.lattice
    }

    pub fn head_expansions(&self) -> usize {
        self.head_expansions
    }

    pub fn materialized_head_forwards(&self) -> usize {
        self.materialized_head_forwards
    }

    /// Globally strongest parent still worth expanding.
    ///
    /// Ranking is cumulative path probability descending, then shallower depth and stable
    /// lattice index.  This is the same deterministic ordering used for the final connected
    /// rerank.  Returning `None` means one of the explicit expansion/node/depth budgets is
    /// exhausted.
    pub fn next_parent(&self) -> Option<usize> {
        if self.head_expansions >= self.config.max_head_expansions
            || self.lattice.nodes().len() >= self.config.max_lattice_nodes
        {
            return None;
        }
        let mut candidates: Vec<usize> = self
            .lattice
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                (!self.expanded[index] && usize::from(node.depth) < self.config.max_depth)
                    .then_some(index)
            })
            .collect();
        candidates.sort_by(|&left, &right| {
            self.lattice.nodes()[right]
                .cumulative_log_probability
                .total_cmp(&self.lattice.nodes()[left].cumulative_log_probability)
                .then_with(|| {
                    self.lattice.nodes()[left]
                        .depth
                        .cmp(&self.lattice.nodes()[right].depth)
                })
                .then_with(|| left.cmp(&right))
        });
        candidates.first().copied()
    }

    /// Stable source-node path from the root through `node`, root first.
    pub fn source_path_to(&self, node: usize) -> Result<Vec<usize>> {
        if node >= self.lattice.nodes().len() {
            return Err(invalid(format!(
                "EAGLE-3 frontier node {node} is out of range"
            )));
        }
        let mut path = Vec::new();
        let mut cursor = Some(node);
        while let Some(index) = cursor {
            path.push(index);
            cursor = self.lattice.nodes()[index].parent;
        }
        path.reverse();
        Ok(path)
    }

    fn recurrent_g(&self, node: usize) -> Result<&[f32]> {
        self.recurrent_g
            .get(node)
            .and_then(|state| state.as_deref())
            .ok_or_else(|| {
                invalid(format!(
                    "EAGLE-3 frontier node {node} has no recurrent input state"
                ))
            })
    }

    /// Attach one full-vocabulary-normalized Metal head observation to the scheduled parent.
    ///
    /// The top candidates are never renormalized.  If the lattice budget cuts an expansion
    /// short, the skipped candidates simply remain omitted probability mass.
    pub fn record_expansion(
        &mut self,
        parent: usize,
        output: &Eagle3MetalOutput,
    ) -> Result<Vec<usize>> {
        let expected = self.next_parent().ok_or_else(|| {
            invalid("EAGLE-3 dynamic frontier has no remaining scheduled expansion")
        })?;
        if parent != expected {
            return Err(invalid(format!(
                "EAGLE-3 dynamic frontier expected parent {expected}, got {parent}"
            )));
        }
        if output.raw_hidden.len() != HIDDEN_SIZE {
            return Err(invalid(format!(
                "EAGLE-3 frontier output hidden width is {}, expected {HIDDEN_SIZE}",
                output.raw_hidden.len()
            )));
        }

        for (slot, candidate) in output.top_candidates.iter().enumerate() {
            if !candidate.logit.is_finite() {
                return Err(invalid(format!(
                    "EAGLE-3 top-k candidate {slot} has non-finite logit {}",
                    candidate.logit
                )));
            }
            if output.top_candidates[..slot].iter().any(|earlier| {
                earlier.draft_token == candidate.draft_token
                    || earlier.target_token == candidate.target_token
            }) {
                return Err(invalid(format!(
                    "EAGLE-3 top-k candidate {slot} duplicates a draft or target token"
                )));
            }
            if slot > 0 {
                let previous = &output.top_candidates[slot - 1];
                if previous.logit < candidate.logit
                    || (previous.logit == candidate.logit
                        && previous.draft_token > candidate.draft_token)
                {
                    return Err(invalid(format!(
                        "EAGLE-3 top-k candidates are not in deterministic rank order at slot {slot}"
                    )));
                }
            }
        }

        let remaining = self
            .config
            .max_lattice_nodes
            .saturating_sub(self.lattice.nodes().len());
        let retained = output
            .top_candidates
            .len()
            .min(self.config.candidates_per_parent)
            .min(remaining);
        let top_logits: Vec<(u32, f32)> = output.top_candidates[..retained]
            .iter()
            .map(|candidate| (candidate.target_token, candidate.logit))
            .collect();
        let normalizer = Eagle3FullVocabularyLogsumexp::from_output(output)?;
        let scores = normalize_draft_top_logits(&top_logits, normalizer.get())
            .map_err(|message| invalid(format!("EAGLE-3 dynamic frontier: {message}")))?;
        let children = self
            .lattice
            .expand(parent, &scores)
            .map_err(|message| invalid(format!("EAGLE-3 dynamic frontier: {message}")))?;
        self.expanded[parent] = true;
        self.expanded
            .extend(std::iter::repeat_n(false, children.len()));
        self.recurrent_g
            .extend(children.iter().map(|_| Some(output.raw_hidden.clone())));
        self.head_expansions += 1;
        Ok(children)
    }

    pub fn finish(self) -> Result<Eagle3DraftForest> {
        let scored = self
            .lattice
            .rerank_connected(self.config.max_verify_nodes, self.config.max_depth)
            .map_err(|message| invalid(format!("EAGLE-3 dynamic frontier: {message}")))?;
        let packed_plan = scored.tree.packed_forest_plan();
        Ok(Eagle3DraftForest {
            scored,
            packed_plan,
        })
    }

    /// Select a verifier width by expected emitted tokens per measured target-round cost.
    ///
    /// This is a small Sequoia-style admission policy over an already-normalized lattice.  It
    /// does not alter candidate probabilities and cannot affect losslessness: the selected
    /// forest still passes through target-authoritative acceptance.  Exact utility ties prefer
    /// the smaller node budget, keeping the decision deterministic and conservative.
    pub fn select_for_verifier_costs(
        &self,
        costs: &[Eagle3VerifierBudgetCost],
    ) -> Result<Eagle3ForestSelection> {
        if costs.is_empty() {
            return Err(invalid(
                "EAGLE-3 verifier admission requires at least one cost point",
            ));
        }
        let mut best: Option<Eagle3ForestSelection> = None;
        for (slot, cost) in costs.iter().enumerate() {
            if cost.max_nodes == 0 || cost.max_nodes > self.config.max_verify_nodes {
                return Err(invalid(format!(
                    "EAGLE-3 verifier cost point {slot} has node budget {}, expected 1..={} ",
                    cost.max_nodes, self.config.max_verify_nodes
                )));
            }
            if !cost.round_cost.is_finite() || cost.round_cost <= 0.0 {
                return Err(invalid(format!(
                    "EAGLE-3 verifier cost point {slot} has invalid round cost {}",
                    cost.round_cost
                )));
            }
            if costs[..slot]
                .iter()
                .any(|earlier| earlier.max_nodes == cost.max_nodes)
            {
                return Err(invalid(format!(
                    "EAGLE-3 verifier cost table repeats node budget {}",
                    cost.max_nodes
                )));
            }

            let scored = self
                .lattice
                .rerank_connected(cost.max_nodes, self.config.max_depth)
                .map_err(|message| invalid(format!("EAGLE-3 verifier admission: {message}")))?;
            let estimated_emitted_tokens = scored.estimated_emitted_tokens();
            let estimated_tokens_per_cost = estimated_emitted_tokens / cost.round_cost;
            let packed_plan = scored.tree.packed_forest_plan();
            let candidate = Eagle3ForestSelection {
                forest: Eagle3DraftForest {
                    scored,
                    packed_plan,
                },
                admitted_node_budget: cost.max_nodes,
                estimated_emitted_tokens,
                estimated_tokens_per_cost,
            };
            let replace = best.as_ref().is_none_or(|incumbent| {
                candidate
                    .estimated_tokens_per_cost
                    .total_cmp(&incumbent.estimated_tokens_per_cost)
                    .is_gt()
                    || (candidate.estimated_tokens_per_cost == incumbent.estimated_tokens_per_cost
                        && candidate.admitted_node_budget < incumbent.admitted_node_budget)
            });
            if replace {
                best = Some(candidate);
            }
        }
        best.ok_or_else(|| invalid("EAGLE-3 verifier admission produced no candidate"))
    }
}

/// Verifier-ready dynamic EAGLE tree plus its packed-forest ancestry plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Eagle3DraftForest {
    pub scored: ScoredTokenTree,
    pub packed_plan: PackedForestPlan,
}

impl Eagle3DraftForest {
    /// Apply target-authoritative greedy acceptance.  Draft scores only choose which rows the
    /// target evaluates; they never choose an emitted token.
    pub fn accept_target_predictions(&self, predictions: &[u32]) -> Result<Eagle3ForestAcceptance> {
        if predictions.len() != self.scored.tree.nodes() {
            return Err(invalid(format!(
                "EAGLE-3 forest has {} rows but target returned {} predictions",
                self.scored.tree.nodes(),
                predictions.len()
            )));
        }
        let (emitted_tokens, leaf_row) = self.scored.tree.accept_longest_path(predictions);
        let capture_rows = self.scored.tree.path_to(leaf_row);
        if emitted_tokens.len() != capture_rows.len() {
            return Err(invalid(format!(
                "EAGLE-3 forest acceptance produced {} tokens from {} target rows",
                emitted_tokens.len(),
                capture_rows.len()
            )));
        }
        let source_nodes = capture_rows
            .iter()
            .map(|&row| self.scored.source_node[row])
            .collect();
        Ok(Eagle3ForestAcceptance {
            emitted_tokens,
            leaf_row,
            capture_rows,
            source_nodes,
        })
    }
}

/// Accepted target path through a draft forest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3ForestAcceptance {
    pub emitted_tokens: Vec<u32>,
    pub leaf_row: usize,
    /// BFS verifier rows whose target layer inputs must update the stable EAGLE head.
    pub capture_rows: Vec<usize>,
    /// Expansion-lattice node ids corresponding to `capture_rows`.
    pub source_nodes: Vec<usize>,
}

impl Eagle3ForestAcceptance {
    /// Gather all-row target captures into accepted-path order for the existing authoritative
    /// EAGLE cache update.  This is the host half of the future tree-verify-with-captures seam.
    pub fn gather_layer_inputs(&self, captures: &[CpuTensor]) -> Result<Vec<CpuTensor>> {
        captures
            .iter()
            .map(|capture| {
                let rows = capture.dim(0)?;
                let width = capture.dim(1)?;
                let mut gathered = Vec::with_capacity(self.capture_rows.len() * width);
                for &row in &self.capture_rows {
                    if row >= rows {
                        return Err(invalid(format!(
                            "EAGLE-3 accepted capture row {row} is outside {} rows of {}",
                            rows, capture.name
                        )));
                    }
                    let start = row * width;
                    gathered.extend_from_slice(&capture.data[start..start + width]);
                }
                CpuTensor::from_f32(
                    format!("{}_accepted_forest", capture.name),
                    vec![self.capture_rows.len(), width],
                    gathered,
                )
            })
            .collect()
    }
}

/// Linear top-1 EAGLE-3 drafter. `stable_seed` is the output of the newest
/// authoritative head-cache row and therefore predicts the first token of the next round.
pub struct Eagle3Drafter {
    head: Eagle3MetalState,
    stable_seed: Option<Eagle3MetalOutput>,
}

impl Eagle3Drafter {
    fn forward_authoritative_last_output(
        &mut self,
        embeddings: &[f32],
        fused: &[f32],
        start: usize,
    ) -> Result<Eagle3MetalOutput> {
        if std::env::var("CAMELID_EAGLE3_FULL_AUTHORITATIVE")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes" | "enabled"
                )
            })
        {
            return metal(self.head.forward_batch(embeddings, fused, start))?
                .pop()
                .ok_or_else(|| invalid("EAGLE-3 authoritative batch produced no output"));
        }
        metal(
            self.head
                .forward_batch_last_output(embeddings, fused, start),
        )
    }

    /// Upload the validated head to Metal. The large host checkpoint buffers are released
    /// before this returns; `Eagle3MetalState` owns its uploaded copies.
    pub fn new(model: Eagle3DraftModel, max_positions: usize) -> Result<Self> {
        let matrices = &model.matrices;
        let norms = &model.norms;
        let weights = Eagle3MetalWeights {
            fc_bf16: &matrices.feature_fusion.bytes,
            q_proj_bf16: &matrices.attention_q.bytes,
            k_proj_bf16: &matrices.attention_k.bytes,
            v_proj_bf16: &matrices.attention_v.bytes,
            o_proj_bf16: &matrices.attention_o.bytes,
            gate_proj_bf16: &matrices.mlp_gate.bytes,
            up_proj_bf16: &matrices.mlp_up.bytes,
            down_proj_bf16: &matrices.mlp_down.bytes,
            lm_head_bf16: &matrices.lm_head.bytes,
            input_layernorm: &norms.input,
            hidden_norm: &norms.hidden,
            post_attention_layernorm: &norms.post_attention,
            output_norm: &norms.output,
            d2t_offsets: &model.d2t_offsets,
        };
        let head = metal(Eagle3MetalState::new(weights, max_positions))?;
        Ok(Self {
            head,
            stable_seed: None,
        })
    }

    pub fn filled(&self) -> usize {
        self.head.filled()
    }

    /// Explore a budgeted dynamic frontier while retaining the longest common prefix between
    /// consecutively selected branches in the ephemeral EAGLE cache.
    ///
    /// No whole-cache snapshots are needed: `Eagle3DynamicFrontier` retains the raw recurrent
    /// `g` input for every candidate, while this method rolls the one-layer KV watermark back
    /// only to the branches' longest common prefix and deterministically replays the divergent
    /// suffix.  The learned head therefore conditions every expansion on its real branch, not
    /// on a top-1 surrogate spine, without paying triangular replay cost for a confident spine.
    ///
    /// Every observation carries the all-evaluated-row log-sum-exp produced beside its Metal
    /// logits. Reduced-row heads fail closed inside `record_expansion`; no callback can
    /// accidentally substitute a top-k-only normalizer.
    pub fn draft_dynamic_frontier(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        anchor: u32,
        config: Eagle3DynamicFrontierConfig,
    ) -> Result<Eagle3DynamicFrontier> {
        let stable_seed = self
            .stable_seed
            .clone()
            .ok_or_else(|| invalid("EAGLE-3 must be seeded before dynamic drafting"))?;
        let stable = self.head.filled();
        let mut frontier = Eagle3DynamicFrontier::new(anchor, config)?;
        frontier.record_expansion(0, &stable_seed)?;

        // Lattice root zero is represented by `stable_seed`, not a private cache row. Each
        // successful non-root materialization below extends this cursor by exactly one row per
        // path node. Regardless of success, the outer cleanup restores the authoritative
        // watermark before returning.
        let mut cursor_path = vec![0usize];
        let materialization = (|| -> Result<Eagle3DynamicFrontier> {
            while let Some(parent) = frontier.next_parent() {
                // Root was consumed from `stable_seed` above. Every subsequent parent has a
                // concrete token path that starts one row beyond the authoritative watermark.
                if parent == 0 {
                    return Err(invalid(
                        "EAGLE-3 dynamic frontier scheduled its root more than once",
                    ));
                }
                let path = frontier.source_path_to(parent)?;
                let transition = eagle3_path_transition(&cursor_path, &path)?;
                debug_assert_eq!(transition.shared_nodes, transition.replay_from);
                if transition.replay_from >= path.len() {
                    return Err(invalid(format!(
                        "EAGLE-3 dynamic frontier scheduled already-materialized parent {parent}"
                    )));
                }

                let expected_cursor_filled = stable
                    .checked_add(cursor_path.len().saturating_sub(1))
                    .ok_or_else(|| invalid("EAGLE-3 dynamic cursor watermark overflow"))?;
                if self.head.filled() != expected_cursor_filled {
                    return Err(invalid(format!(
                        "EAGLE-3 dynamic cursor expected head watermark {expected_cursor_filled}, got {}",
                        self.head.filled()
                    )));
                }
                let retained_filled = stable
                    .checked_add(transition.retained_rows)
                    .ok_or_else(|| invalid("EAGLE-3 dynamic retained watermark overflow"))?;
                metal(self.head.rollback_to_position(retained_filled))?;

                let mut selected_output = None;
                for &source in &path[transition.replay_from..] {
                    let token = frontier.lattice.nodes()[source].token;
                    let recurrent = frontier.recurrent_g(source)?.to_vec();
                    let embedding = target_weights
                        .token_embedding
                        .embedding_lookup(&[token], "eagle3_dynamic_frontier_token_embedding")?;
                    let output = metal(self.head.forward_token(
                        &embedding.data,
                        &recurrent,
                        self.head.filled(),
                    ))?;
                    frontier.materialized_head_forwards += 1;
                    if source == parent {
                        selected_output = Some(output);
                    }
                }
                let output = selected_output.ok_or_else(|| {
                    invalid(format!(
                        "EAGLE-3 dynamic frontier path did not materialize parent {parent}"
                    ))
                })?;
                let expected_path_filled = stable
                    .checked_add(path.len().saturating_sub(1))
                    .ok_or_else(|| invalid("EAGLE-3 dynamic path watermark overflow"))?;
                if self.head.filled() != expected_path_filled {
                    return Err(invalid(format!(
                        "EAGLE-3 dynamic path expected head watermark {expected_path_filled}, got {}",
                        self.head.filled()
                    )));
                }
                cursor_path = path;
                frontier.record_expansion(parent, &output)?;
            }
            Ok(frontier)
        })();
        let rollback = metal(self.head.rollback_to_position(stable));
        match (materialization, rollback) {
            // Preserve the primary materialization error, matching the previous error behavior;
            // the rollback was nevertheless attempted before this match.
            (Err(error), _) => Err(error),
            (Ok(frontier), rollback) => {
                rollback?;
                Ok(frontier)
            }
        }
    }

    /// Seed the stable draft cache from every authoritative prompt row. At head position
    /// `P`, EAGLE consumes `(token[P+1], target_features[P])`; the final prompt row is paired
    /// with the target's freshly sampled, still-unconsumed anchor token.
    pub fn seed_prompt(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        prompt_tokens: &[u32],
        anchor: u32,
        captures: &[CpuTensor],
    ) -> Result<()> {
        if prompt_tokens.is_empty() {
            return Err(invalid("EAGLE-3 prompt must contain at least one token"));
        }
        if self.head.filled() != 0 || self.stable_seed.is_some() {
            return Err(invalid(
                "EAGLE-3 prompt seed may only run on a fresh drafter",
            ));
        }
        let features = interleave_target_layer_inputs(captures)?;
        let rows = features.len() / EAGLE3_AUX_WIDTH;
        if rows != prompt_tokens.len() {
            return Err(invalid(format!(
                "EAGLE-3 prompt capture rows {rows} do not match prompt tokens {}",
                prompt_tokens.len()
            )));
        }
        let fused = metal(self.head.fuse_features(&features))?;
        let mut paired_tokens = Vec::with_capacity(prompt_tokens.len());
        paired_tokens.extend_from_slice(&prompt_tokens[1..]);
        paired_tokens.push(anchor);
        let embeddings = target_weights
            .token_embedding
            .embedding_lookup(&paired_tokens, "eagle3_prompt_next_token_embeddings")?;
        let output = self.forward_authoritative_last_output(&embeddings.data, &fused, 0)?;
        self.stable_seed = Some(output);
        Ok(())
    }

    /// Propose a top-1 linear chain. Recursive rows are ephemeral: the head watermark is
    /// restored to the authoritative stable prefix before this method returns.
    pub fn draft(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let seed = self
            .stable_seed
            .as_ref()
            .ok_or_else(|| invalid("EAGLE-3 must be seeded before drafting"))?;
        let stable = self.head.filled();
        let mut drafts = Vec::with_capacity(max_tokens);
        drafts.push(seed.target_token);
        let mut recurrent = seed.raw_hidden.clone();
        let result = (|| -> Result<()> {
            while drafts.len() < max_tokens {
                let token = *drafts.last().expect("first draft was pushed above");
                let embedding = target_weights
                    .token_embedding
                    .embedding_lookup(&[token], "eagle3_recursive_token_embedding")?;
                let output = metal(self.head.forward_token(
                    &embedding.data,
                    &recurrent,
                    self.head.filled(),
                ))?;
                recurrent = output.raw_hidden;
                drafts.push(output.target_token);
            }
            Ok(())
        })();
        let rollback = metal(self.head.rollback_to_position(stable));
        result?;
        rollback?;
        Ok(drafts)
    }

    /// Extend the stable draft cache using target-verified rows only. `captures` is the
    /// whole verify batch; its first `emitted.len()` rows correspond one-for-one to the
    /// emitted target tokens and are the only rows allowed to survive rejection rollback.
    pub fn accept_authoritative(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        captures: &[CpuTensor],
        emitted: &[u32],
    ) -> Result<()> {
        if emitted.is_empty() {
            return Err(invalid(
                "an EAGLE-3 verify round must emit at least one token",
            ));
        }
        let features = interleave_target_layer_inputs(captures)?;
        let rows = features.len() / EAGLE3_AUX_WIDTH;
        if emitted.len() > rows {
            return Err(invalid(format!(
                "EAGLE-3 verify emitted {} tokens but captured only {rows} target rows",
                emitted.len()
            )));
        }
        let fused = metal(
            self.head
                .fuse_features(&features[..emitted.len() * EAGLE3_AUX_WIDTH]),
        )?;
        let embeddings = target_weights
            .token_embedding
            .embedding_lookup(emitted, "eagle3_authoritative_next_token_embeddings")?;
        let start = self.head.filled();
        let output = self.forward_authoritative_last_output(&embeddings.data, &fused, start)?;
        self.stable_seed = Some(output);
        Ok(())
    }

    /// Commit a target-accepted forest path to the stable EAGLE cache.
    ///
    /// A target tree verifier produces captures in BFS forest-row order.  Only rows on the
    /// accepted root-to-leaf path are authoritative sequence history, so gather those rows
    /// before delegating to the existing linear stable-cache update.
    pub fn accept_authoritative_forest(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        all_row_captures: &[CpuTensor],
        acceptance: &Eagle3ForestAcceptance,
    ) -> Result<()> {
        let accepted_captures = acceptance.gather_layer_inputs(all_row_captures)?;
        self.accept_authoritative(
            target_weights,
            &accepted_captures,
            &acceptance.emitted_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Eagle3DraftCandidate;

    fn capture(name: &str, rows: usize, base: f32) -> CpuTensor {
        let data = (0..rows * HIDDEN_SIZE)
            .map(|index| base + index as f32)
            .collect();
        CpuTensor::from_f32(name, vec![rows, HIDDEN_SIZE], data).unwrap()
    }

    fn output(candidates: &[(u32, f32)], hidden_marker: f32) -> Eagle3MetalOutput {
        let top_candidates: Vec<Eagle3DraftCandidate> = candidates
            .iter()
            .enumerate()
            .map(
                |(draft_token, &(target_token, probability))| Eagle3DraftCandidate {
                    draft_token: draft_token as u32,
                    target_token,
                    logit: probability.ln(),
                },
            )
            .collect();
        Eagle3MetalOutput {
            draft_token: 0,
            target_token: top_candidates
                .first()
                .map(|candidate| candidate.target_token)
                .unwrap_or(0),
            top_candidates,
            evaluated_vocab_rows: EAGLE3_DRAFT_VOCAB,
            evaluated_vocab_logsumexp: 0.0,
            raw_hidden: vec![hidden_marker; HIDDEN_SIZE],
        }
    }

    fn frontier_config(
        max_verify_nodes: usize,
        max_lattice_nodes: usize,
        max_head_expansions: usize,
    ) -> Eagle3DynamicFrontierConfig {
        Eagle3DynamicFrontierConfig {
            max_verify_nodes,
            max_lattice_nodes,
            max_depth: 6,
            candidates_per_parent: 8,
            max_head_expansions,
        }
    }

    #[test]
    fn target_taps_are_interleaved_row_major_in_trained_order() {
        let captures = vec![
            capture("low", 2, 1.0),
            capture("middle", 2, 10_000.0),
            capture("high", 2, 20_000.0),
        ];
        let fused = interleave_target_layer_inputs(&captures).unwrap();
        assert_eq!(&fused[..HIDDEN_SIZE], &captures[0].data[..HIDDEN_SIZE]);
        assert_eq!(
            &fused[HIDDEN_SIZE..2 * HIDDEN_SIZE],
            &captures[1].data[..HIDDEN_SIZE]
        );
        assert_eq!(
            &fused[2 * HIDDEN_SIZE..3 * HIDDEN_SIZE],
            &captures[2].data[..HIDDEN_SIZE]
        );
        assert_eq!(
            &fused[EAGLE3_AUX_WIDTH..EAGLE3_AUX_WIDTH + HIDDEN_SIZE],
            &captures[0].data[HIDDEN_SIZE..]
        );
    }

    #[test]
    fn target_tap_shape_mismatch_fails_closed() {
        let captures = vec![
            capture("low", 2, 0.0),
            capture("middle", 1, 0.0),
            capture("high", 2, 0.0),
        ];
        assert!(interleave_target_layer_inputs(&captures).is_err());
    }

    #[test]
    fn dynamic_cursor_reuses_a_confident_spine_without_triangular_replay() {
        let paths: [&[usize]; 4] = [&[0], &[0, 1], &[0, 1, 3], &[0, 1, 3, 7]];
        let mut cursor_forwards = 0;
        for (depth, pair) in paths.windows(2).enumerate() {
            let transition = eagle3_path_transition(pair[0], pair[1]).unwrap();
            assert_eq!(transition.shared_nodes, depth + 1);
            assert_eq!(transition.retained_rows, depth);
            assert_eq!(transition.replay_from, depth + 1);
            cursor_forwards += pair[1].len() - transition.replay_from;
        }
        let stable_replay_forwards: usize = paths[1..].iter().map(|path| path.len() - 1).sum();
        assert_eq!(cursor_forwards, 3);
        assert_eq!(stable_replay_forwards, 6);
    }

    #[test]
    fn dynamic_cursor_rolls_back_to_lcp_and_replays_only_divergent_suffix() {
        let branch = [0, 1, 3, 7];
        let cousin = [0, 1, 4, 9];
        let transition = eagle3_path_transition(&branch, &cousin).unwrap();
        assert_eq!(transition.shared_nodes, 2);
        assert_eq!(transition.retained_rows, 1);
        assert_eq!(transition.replay_from, 2);
        assert_eq!(&cousin[transition.replay_from..], &[4, 9]);

        let sibling = [0, 2];
        let transition = eagle3_path_transition(&cousin, &sibling).unwrap();
        assert_eq!(transition.retained_rows, 0);
        assert_eq!(&sibling[transition.replay_from..], &[2]);

        assert!(eagle3_path_transition(&[], &[0]).is_err());
        assert!(eagle3_path_transition(&[0], &[]).is_err());
        assert!(eagle3_path_transition(&[0, 1], &[9, 1]).is_err());
    }

    #[test]
    fn dynamic_frontier_schedules_global_probability_and_keeps_omitted_mass() {
        let mut frontier = Eagle3DynamicFrontier::new(10, frontier_config(5, 7, 3)).unwrap();
        let root = output(&[(11, 0.50), (12, 0.30)], 1.0);
        let root_children = frontier.record_expansion(0, &root).unwrap();
        assert_eq!(root_children, vec![1, 2]);
        assert_eq!(frontier.next_parent(), Some(1));

        let under_11 = output(&[(13, 0.50), (14, 0.40)], 2.0);
        frontier.record_expansion(1, &under_11).unwrap();
        // Node 12 has path mass .30, ahead of the new .25 and .20 descendants of node 11.
        assert_eq!(frontier.next_parent(), Some(2));
        assert_eq!(frontier.source_path_to(4).unwrap(), vec![0, 1, 4]);
        assert!(frontier.lattice().nodes().iter().any(|node| (node
            .cumulative_log_probability
            .exp()
            - 0.20)
            .abs()
            < 1.0e-6));

        let under_12 = output(&[(15, 0.50), (16, 0.25)], 3.0);
        frontier.record_expansion(2, &under_12).unwrap();
        let forest = frontier.finish().unwrap();
        assert_eq!(forest.scored.tree.tokens, vec![10, 11, 12, 13, 14]);
        assert_eq!(forest.scored.tree.parent, vec![-1, 0, 0, 1, 1]);
        // The root retained only .80 of its full probability mass. No top-k renormalization
        // turns that into one: the two depth-one scores remain exactly .50 and .30.
        assert!((forest.scored.cumulative_log_probability[1].exp() - 0.50).abs() < 1.0e-6);
        assert!((forest.scored.cumulative_log_probability[2].exp() - 0.30).abs() < 1.0e-6);
    }

    #[test]
    fn verifier_admission_chooses_eight_when_four_and_eight_cost_the_same() {
        let mut frontier = Eagle3DynamicFrontier::new(10, frontier_config(16, 17, 2)).unwrap();
        frontier
            .record_expansion(
                0,
                &output(
                    &[
                        (11, 0.30),
                        (12, 0.20),
                        (13, 0.15),
                        (14, 0.10),
                        (15, 0.08),
                        (16, 0.06),
                        (17, 0.05),
                        (18, 0.03),
                    ],
                    1.0,
                ),
            )
            .unwrap();
        frontier
            .record_expansion(
                1,
                &output(
                    &[
                        (21, 0.30),
                        (22, 0.20),
                        (23, 0.15),
                        (24, 0.10),
                        (25, 0.08),
                        (26, 0.06),
                        (27, 0.05),
                        (28, 0.03),
                    ],
                    2.0,
                ),
            )
            .unwrap();
        let selected = frontier
            .select_for_verifier_costs(&[
                Eagle3VerifierBudgetCost {
                    max_nodes: 1,
                    round_cost: 1.0,
                },
                Eagle3VerifierBudgetCost {
                    max_nodes: 4,
                    round_cost: 1.1,
                },
                Eagle3VerifierBudgetCost {
                    max_nodes: 8,
                    round_cost: 1.1,
                },
                Eagle3VerifierBudgetCost {
                    max_nodes: 16,
                    round_cost: 2.0,
                },
            ])
            .unwrap();
        assert_eq!(selected.admitted_node_budget, 8);
        assert_eq!(selected.forest.scored.tree.nodes(), 8);
    }

    #[test]
    fn verifier_admission_falls_back_to_one_row_for_diffuse_confidence() {
        let mut frontier = Eagle3DynamicFrontier::new(10, frontier_config(16, 16, 1)).unwrap();
        frontier
            .record_expansion(
                0,
                &output(
                    &[
                        (11, 0.02),
                        (12, 0.02),
                        (13, 0.02),
                        (14, 0.02),
                        (15, 0.02),
                        (16, 0.02),
                        (17, 0.02),
                        (18, 0.02),
                    ],
                    1.0,
                ),
            )
            .unwrap();
        let selected = frontier
            .select_for_verifier_costs(&[
                Eagle3VerifierBudgetCost {
                    max_nodes: 1,
                    round_cost: 1.0,
                },
                Eagle3VerifierBudgetCost {
                    max_nodes: 4,
                    round_cost: 1.2,
                },
                Eagle3VerifierBudgetCost {
                    max_nodes: 8,
                    round_cost: 1.3,
                },
                Eagle3VerifierBudgetCost {
                    max_nodes: 16,
                    round_cost: 1.5,
                },
            ])
            .unwrap();
        assert_eq!(selected.admitted_node_budget, 1);
        assert_eq!(selected.forest.scored.tree.tokens, vec![10]);
    }

    #[test]
    fn forest_acceptance_is_target_authoritative_and_gathers_only_its_path() {
        let mut frontier = Eagle3DynamicFrontier::new(10, frontier_config(4, 4, 2)).unwrap();
        frontier
            .record_expansion(0, &output(&[(11, 0.60), (12, 0.40)], 1.0))
            .unwrap();
        frontier
            .record_expansion(1, &output(&[(13, 0.90)], 2.0))
            .unwrap();
        let forest = frontier.finish().unwrap();
        assert_eq!(forest.scored.tree.tokens, vec![10, 11, 12, 13]);

        // The target chooses sibling 12 at the root, then a target-only bonus 99. Draft rank
        // favored 11, but it has no authority over the emitted stream.
        let acceptance = forest.accept_target_predictions(&[12, 0, 99, 0]).unwrap();
        assert_eq!(acceptance.emitted_tokens, vec![12, 99]);
        assert_eq!(acceptance.capture_rows, vec![0, 2]);
        assert_eq!(acceptance.source_nodes, vec![0, 2]);

        let capture = CpuTensor::from_f32(
            "tap",
            vec![4, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        let gathered = acceptance.gather_layer_inputs(&[capture]).unwrap();
        assert_eq!(gathered[0].shape.dims, vec![2, 2]);
        assert_eq!(gathered[0].data, vec![0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn dynamic_frontier_rejects_reduced_or_nonfinite_vocabulary_normalizers() {
        let mut reduced = output(&[(11, 0.60), (12, 0.40)], 1.0);
        reduced.evaluated_vocab_rows = EAGLE3_DRAFT_VOCAB / 2;
        assert!(Eagle3FullVocabularyLogsumexp::from_output(&reduced).is_err());
        let mut frontier = Eagle3DynamicFrontier::new(10, frontier_config(4, 4, 1)).unwrap();
        assert!(frontier.record_expansion(0, &reduced).is_err());
        assert_eq!(frontier.lattice().nodes().len(), 1);
        assert_eq!(frontier.head_expansions(), 0);

        let mut nonfinite = output(&[(11, 0.60), (12, 0.40)], 1.0);
        nonfinite.evaluated_vocab_logsumexp = f32::NAN;
        assert!(Eagle3FullVocabularyLogsumexp::from_output(&nonfinite).is_err());
        nonfinite.evaluated_vocab_logsumexp = f32::INFINITY;
        assert!(Eagle3FullVocabularyLogsumexp::from_output(&nonfinite).is_err());
    }
}
