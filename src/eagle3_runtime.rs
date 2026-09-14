//! Correctness-first recurrent EAGLE-3 drafting orchestration.
//!
//! The target remains authoritative. This module owns only the learned draft head and its
//! private one-layer KV cache; every proposed token is still checked by the target model's
//! existing greedy speculative verifier before it can be emitted.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::eagle3::{Eagle3DraftModel, HIDDEN_SIZE, TARGET_LAYER_INPUT_IDS, TARGET_VOCAB_SIZE};
use crate::error::{BackendError, Result};
use crate::inference::spec_tree::{
    normalize_draft_top_logits, DynamicDraftLattice, PackedForestPlan, ScoredTokenTree, TokenTree,
    TREE_MAX_NODES,
};
use crate::inference::LlamaLoadedWeights;
use crate::metal::{
    Eagle3AuthoritativeE1ShadowComparison, Eagle3MetalOutput, Eagle3MetalScoredRow,
    Eagle3MetalState, Eagle3MetalWeights, Eagle3SelectiveEdgePromotionAttempt,
    Eagle3SelectiveEdgePromotionReceipt, ResidentIndexedHeadEarlyRow,
    ResidentIndexedHeadEarlySnapshot, EAGLE3_AUX_WIDTH, EAGLE3_DRAFT_VOCAB,
    RESIDENT_INDEXED_HEAD_SHADOW_MAX_CANDIDATES,
};
use crate::tensor::CpuTensor;

pub const EAGLE3_AUTHORITATIVE_CB_FUSION_ENV: &str = "CAMELID_BENCH_EAGLE3_AUTHORITATIVE_CB_FUSION";

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::RuntimeShapeMismatch(message.into())
}

fn metal<T>(result: std::result::Result<T, String>) -> Result<T> {
    result.map_err(|message| invalid(format!("EAGLE-3 Metal runtime: {message}")))
}

fn eagle3_env_enabled_from(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes" | "enabled"
        )
    })
}

fn eagle3_batch_authoritative_kv_enabled_from(value: Option<&str>) -> bool {
    eagle3_env_enabled_from(value)
}

fn eagle3_batch_authoritative_kv_enabled() -> bool {
    let value = std::env::var("CAMELID_EAGLE3_BATCH_AUTHORITATIVE_KV").ok();
    eagle3_batch_authoritative_kv_enabled_from(value.as_deref())
}

fn eagle3_full_authoritative_enabled() -> bool {
    let value = std::env::var("CAMELID_EAGLE3_FULL_AUTHORITATIVE").ok();
    eagle3_env_enabled_from(value.as_deref())
}

fn eagle3_authoritative_cb_fusion_enabled_from(value: Option<&str>) -> bool {
    eagle3_env_enabled_from(value)
}

fn eagle3_authoritative_cb_fusion_enabled() -> bool {
    let value = std::env::var(EAGLE3_AUTHORITATIVE_CB_FUSION_ENV).ok();
    eagle3_authoritative_cb_fusion_enabled_from(value.as_deref())
}

fn validate_authoritative_cb_fusion_dependencies(
    enabled: bool,
    batched_kv: bool,
    full_authoritative: bool,
) -> Result<()> {
    if enabled && !batched_kv {
        return Err(invalid(
            "CAMELID_BENCH_EAGLE3_AUTHORITATIVE_CB_FUSION=1 requires CAMELID_EAGLE3_BATCH_AUTHORITATIVE_KV=1",
        ));
    }
    if enabled && full_authoritative {
        return Err(invalid(
            "CAMELID_BENCH_EAGLE3_AUTHORITATIVE_CB_FUSION=1 cannot be combined with CAMELID_EAGLE3_FULL_AUTHORITATIVE=1",
        ));
    }
    Ok(())
}

// Cache capacity and attention span are independent. E9 keeps absolute K/V rows for
// rollback/catch-up but each query reads at most its checkpoint-declared trailing window.
fn validate_drafter_capacity(sliding_window: Option<usize>, max_positions: usize) -> Result<()> {
    if max_positions == 0 {
        return Err(invalid("EAGLE-3 head cache capacity must be non-zero"));
    }
    if sliding_window == Some(0) {
        return Err(BackendError::UnsupportedModelArchitecture(
            "EAGLE-3 checkpoint requests a zero-position sliding attention window".to_string(),
        ));
    }
    Ok(())
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

/// Target-authoritative rows accepted while drafting does not depend on the EAGLE head.
///
/// A suffix drafter can verify several rounds without consulting `stable_seed`.  Updating the
/// learned head after every such round is wasted work: all intermediate decoder outputs are
/// unobservable, while the target captures and emitted tokens are sufficient to reconstruct the
/// exact one-layer K/V history later.  This buffer keeps those pairs in sequence order until a
/// learned-head draft is actually needed.
///
/// The memory bound is small for the intended benchmark lane: one pending row is three target
/// layer inputs (`3 * HIDDEN_SIZE * sizeof(f32)`, 36 KiB for Llama-3.2 3B) plus one token id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Eagle3AuthoritativeCatchup {
    emitted: Vec<u32>,
    interleaved_features: Vec<f32>,
}

impl Eagle3AuthoritativeCatchup {
    pub fn is_empty(&self) -> bool {
        self.emitted.is_empty()
    }

    pub fn pending_rows(&self) -> usize {
        self.emitted.len()
    }

    /// The target watermark represented by the materialized head prefix plus these pending rows.
    pub fn effective_filled(&self, materialized_filled: usize) -> Result<usize> {
        materialized_filled
            .checked_add(self.pending_rows())
            .ok_or_else(|| invalid("EAGLE-3 deferred authoritative watermark overflow"))
    }

    /// Retain only the authoritative capture prefix paired with `emitted`.
    ///
    /// Linear target verification may return additional capture rows for rejected draft tokens;
    /// those rows must never enter the stable learned-head cache.  Validation is transactional:
    /// an invalid round leaves every previously queued row intact.
    pub fn push(&mut self, captures: &[CpuTensor], emitted: &[u32]) -> Result<()> {
        if emitted.is_empty() {
            return Err(invalid(
                "an EAGLE-3 deferred authoritative round must emit at least one token",
            ));
        }
        let features = interleave_target_layer_inputs(captures)?;
        let rows = features.len() / EAGLE3_AUX_WIDTH;
        if emitted.len() > rows {
            return Err(invalid(format!(
                "EAGLE-3 deferred verify emitted {} tokens but captured only {rows} target rows",
                emitted.len()
            )));
        }
        let admitted_values = emitted
            .len()
            .checked_mul(EAGLE3_AUX_WIDTH)
            .ok_or_else(|| invalid("EAGLE-3 deferred feature length overflow"))?;
        self.emitted.extend_from_slice(emitted);
        self.interleaved_features
            .extend_from_slice(&features[..admitted_values]);
        Ok(())
    }

    fn clear(&mut self) {
        self.emitted.clear();
        self.interleaved_features.clear();
    }
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
    pub adaptive_branching: bool,
    /// Retain every full-target-vocabulary candidate id observed at each learned-head
    /// expansion, including candidates later removed by lattice/adaptive admission. This is
    /// diagnostic evidence only: it never participates in scheduling, reranking, verification,
    /// acceptance, or token emission.
    pub certified_argmax_shadow: bool,
}

impl Default for Eagle3DynamicFrontierConfig {
    fn default() -> Self {
        Self {
            max_verify_nodes: TREE_MAX_NODES,
            max_lattice_nodes: 60,
            max_depth: 6,
            candidates_per_parent: 8,
            max_head_expansions: 8,
            adaptive_branching: false,
            certified_argmax_shadow: false,
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

/// Candidate ids observed at one learned-head expansion before any lattice admission.
///
/// `parent_source_node` is the stable expansion-lattice id, not a verifier BFS row. The
/// verifier-ready forest carries the corresponding `ScoredTokenTree::source_node` mapping, so
/// diagnostics can recover the candidates associated with selected verifier rows without
/// changing tree selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3ArgmaxCandidateExpansion {
    pub parent_source_node: usize,
    /// Full head output in deterministic draft-rank order, before lattice/adaptive truncation.
    pub candidate_target_tokens: Vec<u32>,
    /// The subset actually admitted as lattice children, in the same deterministic order.
    pub lattice_admitted_target_tokens: Vec<u32>,
}

/// Default-off, read-only candidate evidence for a possible future certified target head.
///
/// This type is intentionally named `Shadow`: candidate coverage alone is not an argmax proof.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Eagle3ArgmaxCandidateShadow {
    pub expansions: Vec<Eagle3ArgmaxCandidateExpansion>,
}

impl Eagle3ArgmaxCandidateShadow {
    /// Stable ascending union suitable for one diagnostic indexed-head scoring dispatch.
    pub fn target_token_union(&self) -> Vec<u32> {
        self.expansions
            .iter()
            .flat_map(|expansion| expansion.candidate_target_tokens.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn candidate_observations(&self) -> usize {
        self.expansions
            .iter()
            .map(|expansion| expansion.candidate_target_tokens.len())
            .sum()
    }

    pub fn lattice_admitted_observations(&self) -> usize {
        self.expansions
            .iter()
            .map(|expansion| expansion.lattice_admitted_target_tokens.len())
            .sum()
    }
}

pub const EAGLE3_REPLAY_SCATTER_ENV: &str = "CAMELID_BENCH_EAGLE3_REPLAY_SCATTER";

/// Benchmark-only replay-by-scatter gate. Only the exact spelling `1` arms it; every other
/// value, including `true`, fails closed to the established replay-by-forward path.
fn eagle3_replay_scatter_enabled_from(value: Option<&str>) -> bool {
    value.is_some_and(|value| value.trim() == "1")
}

pub fn eagle3_replay_scatter_enabled() -> bool {
    eagle3_replay_scatter_enabled_from(std::env::var(EAGLE3_REPLAY_SCATTER_ENV).ok().as_deref())
}

pub const EAGLE3_DRAFT_EARLY_EXIT_ENV: &str = "CAMELID_EAGLE3_DRAFT_EARLY_EXIT";

/// Parse the confidence-gated draft early-exit threshold.
///
/// `Ok(None)` is the byte-preserving default: unset, empty, or an explicit zero leave the
/// dynamic frontier scheduler exactly as it is. `Ok(Some(theta))` for a finite `theta` in
/// `(0, 1]` arms the exit. Anything else is malformed and reported so the caller can fail
/// closed to the default rather than guess.
fn eagle3_draft_early_exit_theta_from(
    value: Option<&str>,
) -> std::result::Result<Option<f64>, String> {
    let Some(raw) = value.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(None);
    };
    let theta = raw.parse::<f64>().map_err(|error| {
        format!("{EAGLE3_DRAFT_EARLY_EXIT_ENV} must be a number in (0, 1], got {raw:?}: {error}")
    })?;
    if !theta.is_finite() || !(0.0..=1.0).contains(&theta) {
        return Err(format!(
            "{EAGLE3_DRAFT_EARLY_EXIT_ENV} must be a finite number in (0, 1], got {raw:?}"
        ));
    }
    Ok((theta > 0.0).then_some(theta))
}

/// Process-wide confidence-gated draft early-exit threshold, read once.
///
/// A malformed value prints a single stderr line and behaves as unset, so a typo can never
/// silently change the shipped X5 lattice sizing or the expansion order.
pub fn eagle3_draft_early_exit_theta() -> Option<f64> {
    static THETA: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *THETA.get_or_init(|| {
        match eagle3_draft_early_exit_theta_from(
            std::env::var(EAGLE3_DRAFT_EARLY_EXIT_ENV).ok().as_deref(),
        ) {
            Ok(theta) => theta,
            Err(message) => {
                eprintln!("[eagle3-draft-early-exit] {message}; the gate stays off");
                None
            }
        }
    })
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
    /// Post-RoPE key/value of every scored non-root node, retained only under the
    /// replay-by-scatter gate. A branch switch that has to re-materialize an already-scored
    /// node restores these through the F16 scatter instead of streaming the head weights.
    scored_kv: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    /// Path-replay rows restored by scatter instead of by a head forward.
    replay_scatter_commits: usize,
    /// Completely target-blind observation sidecar. `None` is the byte-preserving default.
    certified_argmax_shadow: Option<Eagle3ArgmaxCandidateShadow>,
    /// The scheduled expansion that the confidence-gated early exit declined to materialize.
    /// `None` when the exit is off or never fired for this lattice.
    early_exit: Option<Eagle3NextExpansionEvidence>,
}

/// Target-blind evidence available immediately before one dynamic-frontier head expansion.
///
/// The scheduled parent and its cumulative probability are derived entirely from draft-head
/// observations already attached to the lattice. In particular, no target-verifier result from
/// the current round exists when this value is produced. This makes the seam safe for causal,
/// benchmark-only expansion admission without weakening target-authoritative acceptance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eagle3NextExpansionEvidence {
    /// Head observations already recorded, including the root distribution.
    pub completed_head_expansions: usize,
    /// Stable expansion-lattice id of the next globally ranked parent.
    pub next_parent: usize,
    pub next_parent_depth: usize,
    pub next_parent_cumulative_log_probability: f32,
    pub next_parent_cumulative_probability: f32,
}

/// One measured verification-round cost point. Units are arbitrary but must be consistent across
/// the table (microseconds is convenient). A whole-run throughput policy should include shared
/// work already paid before this choice, such as frontier materialization, in every point. Keeping
/// the table caller-supplied lets mini2 choose width 8 when its K-quant k4/k8 kernels are flat
/// without baking one machine's timing into the checkpoint or the generic scheduler.
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
        let config = config.validate()?;
        if config.certified_argmax_shadow && anchor as usize >= TARGET_VOCAB_SIZE {
            return Err(invalid(format!(
                "EAGLE-3 certified-argmax shadow anchor id {anchor} is outside target 0..{TARGET_VOCAB_SIZE}"
            )));
        }
        Ok(Self {
            config,
            lattice: DynamicDraftLattice::new(anchor),
            expanded: vec![false],
            recurrent_g: vec![None],
            head_expansions: 0,
            materialized_head_forwards: 0,
            scored_kv: vec![None],
            replay_scatter_commits: 0,
            certified_argmax_shadow: config
                .certified_argmax_shadow
                .then(Eagle3ArgmaxCandidateShadow::default),
            early_exit: None,
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

    /// Path-replay rows restored by a scatter-only commit rather than a head forward. Zero
    /// with the gate off, so a receipt can prove the lane fired.
    pub fn replay_scatter_commits(&self) -> usize {
        self.replay_scatter_commits
    }

    /// [`Self::record_expansion`] that also retains the scored row's key/value for later
    /// scatter-only replays.
    pub fn record_scored_expansion(
        &mut self,
        parent: usize,
        row: &Eagle3MetalScoredRow,
    ) -> Result<Vec<usize>> {
        let kv_dim = crate::metal::EAGLE3_KV_HEADS * crate::metal::EAGLE3_HEAD_DIM;
        if row.key.len() != kv_dim || row.value.len() != kv_dim {
            return Err(invalid(format!(
                "EAGLE-3 scored expansion for node {parent} carries key/value widths {}/{}, expected {kv_dim}",
                row.key.len(),
                row.value.len()
            )));
        }
        let children = self.record_expansion(parent, &row.output)?;
        let node_count = self.lattice.nodes().len();
        if self.scored_kv.len() < node_count {
            self.scored_kv.resize(node_count, None);
        }
        self.scored_kv[parent] = Some((row.key.clone(), row.value.clone()));
        Ok(children)
    }

    /// Key/value retained for an already-scored node.
    fn scored_kv(&self, node: usize) -> Result<(&[f32], &[f32])> {
        self.scored_kv
            .get(node)
            .and_then(|entry| entry.as_ref())
            .map(|(key, value)| (key.as_slice(), value.as_slice()))
            .ok_or_else(|| {
                invalid(format!(
                    "EAGLE-3 replay-by-scatter has no retained key/value for node {node}"
                ))
            })
    }

    pub fn certified_argmax_shadow(&self) -> Option<&Eagle3ArgmaxCandidateShadow> {
        self.certified_argmax_shadow.as_ref()
    }

    /// Read the target-blind evidence for the next scheduled expansion without mutating the
    /// lattice or model cache.
    pub fn next_expansion_evidence(&self) -> Option<Eagle3NextExpansionEvidence> {
        let next_parent = self.next_parent()?;
        self.expansion_evidence_for_parent(next_parent)
    }

    fn expansion_evidence_for_parent(
        &self,
        next_parent: usize,
    ) -> Option<Eagle3NextExpansionEvidence> {
        let node = self.lattice.nodes().get(next_parent)?;
        Some(Eagle3NextExpansionEvidence {
            completed_head_expansions: self.head_expansions,
            next_parent,
            next_parent_depth: usize::from(node.depth),
            next_parent_cumulative_log_probability: node.cumulative_log_probability,
            next_parent_cumulative_probability: node.cumulative_log_probability.exp(),
        })
    }

    /// The expansion the confidence-gated early exit declined, if it fired for this lattice.
    pub fn draft_early_exit(&self) -> Option<Eagle3NextExpansionEvidence> {
        self.early_exit
    }

    /// Confidence-gated early exit, evaluated immediately before the next non-root expansion.
    ///
    /// The rule reads the scheduler's own ranking key: the cumulative log probability of the
    /// globally strongest unexpanded node, i.e. the parent [`Self::next_parent`] would
    /// materialize next. When that joint probability is strictly below `theta` the caller
    /// should stop drafting and verify the lattice built so far. Equality keeps expanding, a
    /// `theta` of zero or less never stops, and the root distribution is never subject to the
    /// rule because it costs no head forward. No budget, depth, or admission parameter is
    /// consulted or altered here; only the number of materialized expansions can change.
    pub fn early_exit_before_next_expansion(
        &self,
        theta: f64,
    ) -> Option<Eagle3NextExpansionEvidence> {
        let next_parent = self.next_parent()?;
        self.early_exit_for_parent(next_parent, theta)
    }

    fn early_exit_for_parent(
        &self,
        next_parent: usize,
        theta: f64,
    ) -> Option<Eagle3NextExpansionEvidence> {
        if self.head_expansions == 0 {
            return None;
        }
        let evidence = self.expansion_evidence_for_parent(next_parent)?;
        (f64::from(evidence.next_parent_cumulative_probability) < theta).then_some(evidence)
    }

    /// Record that drafting stopped at `evidence` instead of materializing it. The evidence
    /// must name the expansion this frontier would schedule next; anything else fails closed
    /// so a receipt can never attribute an exit to the wrong expansion.
    pub fn record_early_exit(&mut self, evidence: Eagle3NextExpansionEvidence) -> Result<()> {
        let expected = self.next_expansion_evidence().ok_or_else(|| {
            invalid("EAGLE-3 dynamic frontier early exit has no remaining scheduled expansion")
        })?;
        if evidence != expected || self.early_exit.is_some() {
            return Err(invalid(format!(
                "EAGLE-3 dynamic frontier early exit {evidence:?} does not match the scheduled expansion {expected:?}"
            )));
        }
        self.early_exit = Some(evidence);
        Ok(())
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

        // Keep the stronger target-vocabulary validation scoped to the shadow. With the gate
        // off, the established frontier validation and mutation order remain unchanged.
        if self.certified_argmax_shadow.is_some() {
            if output.draft_token as usize >= EAGLE3_DRAFT_VOCAB
                || output.target_token as usize >= TARGET_VOCAB_SIZE
            {
                return Err(invalid(format!(
                    "EAGLE-3 certified-argmax shadow top-1 ids draft={} target={} are outside draft 0..{EAGLE3_DRAFT_VOCAB} / target 0..{TARGET_VOCAB_SIZE}",
                    output.draft_token, output.target_token
                )));
            }
            for (slot, candidate) in output.top_candidates.iter().enumerate() {
                if candidate.draft_token as usize >= EAGLE3_DRAFT_VOCAB
                    || candidate.target_token as usize >= TARGET_VOCAB_SIZE
                {
                    return Err(invalid(format!(
                        "EAGLE-3 certified-argmax shadow candidate {slot} ids draft={} target={} are outside draft 0..{EAGLE3_DRAFT_VOCAB} / target 0..{TARGET_VOCAB_SIZE}",
                        candidate.draft_token, candidate.target_token
                    )));
                }
            }
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
        let filtered_scores = if scores.len() > 1 && self.config.adaptive_branching {
            let p0 = scores[0].log_probability.exp();
            let mut retained = vec![scores[0]];
            let mut cum_p = p0;
            if p0 < 0.70 {
                for candidate in &scores[1..] {
                    let p = candidate.log_probability.exp();
                    if p >= 0.08 && cum_p < 0.85 {
                        cum_p += p;
                        retained.push(*candidate);
                    }
                }
            }
            retained
        } else {
            scores
        };
        let children = self
            .lattice
            .expand(parent, &filtered_scores)
            .map_err(|message| invalid(format!("EAGLE-3 dynamic frontier: {message}")))?;
        if let Some(shadow) = self.certified_argmax_shadow.as_mut() {
            let lattice_admitted_target_tokens = filtered_scores
                .iter()
                .map(|candidate| candidate.token)
                .collect::<Vec<_>>();
            debug_assert_eq!(lattice_admitted_target_tokens.len(), children.len());
            shadow.expansions.push(Eagle3ArgmaxCandidateExpansion {
                parent_source_node: parent,
                candidate_target_tokens: output
                    .top_candidates
                    .iter()
                    .map(|candidate| candidate.target_token)
                    .collect(),
                lattice_admitted_target_tokens,
            });
        }
        self.expanded[parent] = true;
        self.expanded
            .extend(std::iter::repeat_n(false, children.len()));
        self.recurrent_g
            .extend(children.iter().map(|_| Some(output.raw_hidden.clone())));
        self.head_expansions += 1;
        Ok(children)
    }

    pub fn finish(self) -> Result<Eagle3DraftForest> {
        self.finish_borrowed()
    }

    /// Produce ordinary verifier N without consuming the already-materialized frontier.
    /// Benchmark-only candidate selectors may inspect the same causal lattice immediately after
    /// this call, then drop the frontier before target verification.
    pub fn finish_borrowed(&self) -> Result<Eagle3DraftForest> {
        let scored = self
            .lattice
            .rerank_connected(self.config.max_verify_nodes, self.config.max_depth)
            .map_err(|message| invalid(format!("EAGLE-3 dynamic frontier: {message}")))?;
        let packed_plan = scored.tree.packed_forest_plan();
        Ok(Eagle3DraftForest {
            scored,
            packed_plan,
            certified_argmax_shadow: self.certified_argmax_shadow.clone(),
        })
    }

    /// Select a verifier width by expected emitted tokens per measured verification-round cost.
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
                    certified_argmax_shadow: self.certified_argmax_shadow.clone(),
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
    /// Present only when the default-off shadow gate was enabled while building this frontier.
    pub certified_argmax_shadow: Option<Eagle3ArgmaxCandidateShadow>,
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

    /// Freeze the exact integer inputs for device-side target acceptance.
    ///
    /// The future Metal selector scans verifier rows in the same increasing-row order as
    /// `TokenTree::accept_longest_path`.  Dynamic lattice expansion already rejects duplicate
    /// tokens under one parent; this second check makes the default-off device lane fail closed
    /// if a different forest producer ever violates that invariant.  Increasing-row selection
    /// is nevertheless retained by the device ABI, so its result is also defined exactly for a
    /// malformed duplicate-sibling fixture used by the Metal falsifier.
    pub fn plan_device_acceptance(&self) -> Result<Eagle3DeviceAcceptancePlan> {
        let tree = &self.scored.tree;
        let nodes = tree.nodes();
        if nodes == 0 || nodes > TREE_MAX_NODES {
            return Err(invalid(format!(
                "EAGLE-3 device acceptance requires 1..={TREE_MAX_NODES} rows, got {nodes}"
            )));
        }
        if tree.parent.len() != nodes || tree.depth.len() != nodes {
            return Err(invalid(format!(
                "EAGLE-3 device acceptance tree has token/parent/depth lengths {}/{}/{}",
                nodes,
                tree.parent.len(),
                tree.depth.len()
            )));
        }
        if tree.parent[0] != -1 || tree.depth[0] != 0 {
            return Err(invalid(
                "EAGLE-3 device acceptance root must have parent -1 and depth 0",
            ));
        }
        for row in 0..nodes {
            if tree.tokens[row] as usize >= TARGET_VOCAB_SIZE {
                return Err(invalid(format!(
                    "EAGLE-3 device acceptance token {} at row {row} is outside target vocabulary 0..{TARGET_VOCAB_SIZE}",
                    tree.tokens[row]
                )));
            }
            if row == 0 {
                continue;
            }
            let parent = usize::try_from(tree.parent[row]).map_err(|_| {
                invalid(format!(
                    "EAGLE-3 device acceptance row {row} has no verifier parent"
                ))
            })?;
            let depth = usize::from(tree.depth[row]);
            if parent >= row
                || depth == 0
                || usize::from(tree.depth[parent]).checked_add(1) != Some(depth)
            {
                return Err(invalid(format!(
                    "EAGLE-3 device acceptance row {row} has invalid parent/depth {parent}/{depth}"
                )));
            }
            if (1..row).any(|earlier| {
                tree.parent[earlier] == tree.parent[row] && tree.tokens[earlier] == tree.tokens[row]
            }) {
                return Err(invalid(format!(
                    "EAGLE-3 device acceptance parent {parent} has duplicate child token {}",
                    tree.tokens[row]
                )));
            }
        }
        Ok(Eagle3DeviceAcceptancePlan {
            tree_tokens: tree.tokens.clone(),
            tree_parent: tree.parent.clone(),
            tree_depth: tree.depth.clone(),
        })
    }

    /// Build the immutable row mapping for a target-overlapped authoritative precompute.
    ///
    /// A speculative EAGLE row cannot be committed as authoritative: its recurrent `g` is the
    /// parent draft cell's `raw_hidden`, while an authoritative row's `g` is `fc` applied to the
    /// target captures.  The exact reusable unit is instead an *edge cell*.  For verifier edge
    /// `parent -> child`, the authoritative cell consumes the child's token embedding and the
    /// target captures from `parent`.  Its logical EAGLE position is
    /// `stable_prefix + depth(child) - 1`.
    ///
    /// `terminal_candidate_tokens[row]` is an optional target-blind guess for the final bonus
    /// token predicted at `row`.  It must not duplicate an already-verified child: such a token
    /// would continue target acceptance rather than terminate there.  A future Metal lane can
    /// project every verified edge's K/V plus these virtual terminal cells while the target tail
    /// is still running, then use [`Self::resolve_authoritative_precompute`] after the ordinary
    /// target argmax.  This helper owns no model state and cannot affect acceptance.
    pub fn plan_authoritative_precompute(
        &self,
        terminal_candidate_tokens: &[Option<u32>],
    ) -> Result<Eagle3AuthoritativePrecomputePlan> {
        let tree = &self.scored.tree;
        if terminal_candidate_tokens.len() != tree.nodes() {
            return Err(invalid(format!(
                "EAGLE-3 authoritative precompute has {} terminal candidates for {} verifier rows",
                terminal_candidate_tokens.len(),
                tree.nodes()
            )));
        }
        let device_acceptance = self.plan_device_acceptance()?;

        let mut verified_edges = Vec::with_capacity(tree.nodes().saturating_sub(1));
        for verifier_row in 1..tree.nodes() {
            let parent = usize::try_from(tree.parent[verifier_row]).map_err(|_| {
                invalid(format!(
                    "EAGLE-3 authoritative edge row {verifier_row} has no verifier parent"
                ))
            })?;
            let depth = usize::from(tree.depth[verifier_row]);
            if parent >= verifier_row
                || depth == 0
                || usize::from(tree.depth[parent]).checked_add(1) != Some(depth)
            {
                return Err(invalid(format!(
                    "EAGLE-3 authoritative edge row {verifier_row} has invalid parent/depth {parent}/{depth}"
                )));
            }
            verified_edges.push(Eagle3AuthoritativePrecomputeCell {
                verifier_row,
                token: tree.tokens[verifier_row],
                capture_row: parent,
                logical_position_offset: depth - 1,
                predecessor_edge_rows: tree.path_to(parent)[1..].to_vec(),
            });
        }

        let mut terminal_candidates = Vec::with_capacity(tree.nodes());
        for (verifier_row, candidate) in terminal_candidate_tokens.iter().copied().enumerate() {
            let Some(token) = candidate else {
                terminal_candidates.push(None);
                continue;
            };
            if token as usize >= TARGET_VOCAB_SIZE {
                return Err(invalid(format!(
                    "EAGLE-3 authoritative terminal candidate {token} at row {verifier_row} is outside target vocabulary 0..{TARGET_VOCAB_SIZE}"
                )));
            }
            if (verifier_row + 1..tree.nodes()).any(|child| {
                tree.parent[child] == verifier_row as i32 && tree.tokens[child] == token
            }) {
                return Err(invalid(format!(
                    "EAGLE-3 authoritative terminal candidate {token} at row {verifier_row} is already a verified child"
                )));
            }
            terminal_candidates.push(Some(Eagle3AuthoritativePrecomputeCell {
                verifier_row,
                token,
                capture_row: verifier_row,
                logical_position_offset: usize::from(tree.depth[verifier_row]),
                predecessor_edge_rows: tree.path_to(verifier_row)[1..].to_vec(),
            }));
        }

        Ok(Eagle3AuthoritativePrecomputePlan {
            device_acceptance,
            verified_edges,
            terminal_candidates,
        })
    }

    /// Resolve an already-built authoritative precompute against the unchanged target result.
    ///
    /// A `Complete` result identifies a wholly precomputed exact commit path.  `PrefixOnly`
    /// still reuses every accepted verified edge's authoritative K/V, but the final target bonus
    /// missed the target-blind candidate and must run one ordinary terminal cell.  In both cases
    /// the target's predictions remain the sole authority for emitted tokens.
    pub fn resolve_authoritative_precompute(
        &self,
        plan: &Eagle3AuthoritativePrecomputePlan,
        acceptance: &Eagle3ForestAcceptance,
    ) -> Result<Eagle3AuthoritativeCommitResolution> {
        let tree = &self.scored.tree;
        if plan.device_acceptance.tree_tokens != tree.tokens
            || plan.device_acceptance.tree_parent != tree.parent
            || plan.device_acceptance.tree_depth != tree.depth
            || plan.verified_edges.len() != tree.nodes().saturating_sub(1)
            || plan.terminal_candidates.len() != tree.nodes()
        {
            return Err(invalid(
                "EAGLE-3 authoritative precompute plan belongs to a different verifier tree",
            ));
        }
        if acceptance.emitted_tokens.is_empty()
            || acceptance.emitted_tokens.len() != acceptance.capture_rows.len()
            || acceptance.capture_rows.first() != Some(&0)
            || acceptance.capture_rows.last() != Some(&acceptance.leaf_row)
            || acceptance.capture_rows != tree.path_to(acceptance.leaf_row)
        {
            return Err(invalid(
                "EAGLE-3 authoritative precompute received an invalid accepted path",
            ));
        }

        let mut verified_edge_rows = Vec::with_capacity(acceptance.capture_rows.len() - 1);
        for (emitted_index, &verifier_row) in acceptance.capture_rows.iter().enumerate().skip(1) {
            let edge = plan
                .verified_edges
                .get(verifier_row - 1)
                .ok_or_else(|| invalid("EAGLE-3 authoritative accepted edge is missing"))?;
            let expected_parent = acceptance.capture_rows[emitted_index - 1];
            let expected_token = acceptance.emitted_tokens[emitted_index - 1];
            if edge.verifier_row != verifier_row
                || edge.capture_row != expected_parent
                || edge.token != expected_token
                || edge.logical_position_offset != emitted_index - 1
                || edge.predecessor_edge_rows != acceptance.capture_rows[1..emitted_index]
            {
                return Err(invalid(format!(
                    "EAGLE-3 authoritative precompute edge {verifier_row} does not match accepted token {expected_token}"
                )));
            }
            verified_edge_rows.push(verifier_row);
        }

        let terminal_capture_row = acceptance.leaf_row;
        let terminal_token = *acceptance
            .emitted_tokens
            .last()
            .expect("non-empty acceptance checked above");
        let terminal = plan.terminal_candidates[terminal_capture_row].as_ref();
        if terminal.is_some_and(|cell| {
            cell.verifier_row == terminal_capture_row
                && cell.capture_row == terminal_capture_row
                && cell.token == terminal_token
                && cell.logical_position_offset == acceptance.capture_rows.len() - 1
                && cell.predecessor_edge_rows == acceptance.capture_rows[1..]
        }) {
            return Ok(Eagle3AuthoritativeCommitResolution::Complete {
                verified_edge_rows,
                terminal_candidate_row: terminal_capture_row,
            });
        }
        Ok(Eagle3AuthoritativeCommitResolution::PrefixOnly {
            verified_edge_rows,
            terminal_capture_row,
            terminal_token,
        })
    }
}

/// Frozen device ABI for exact target-authoritative tree acceptance.
///
/// Metal consumes these three arrays plus the production target `pred_buf`.  No scores or draft
/// ranks enter selection.  Parent rows precede children, which bounds the single-thread selector
/// to at most [`TREE_MAX_NODES`] iterations without a dynamic dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3DeviceAcceptancePlan {
    pub tree_tokens: Vec<u32>,
    pub tree_parent: Vec<i32>,
    pub tree_depth: Vec<u16>,
}

impl Eagle3DeviceAcceptancePlan {
    /// Host mirror of the proposed one-thread Metal selector.
    ///
    /// This is a falsification oracle and a safe fallback, not a second acceptance policy.  It
    /// deliberately uses the same increasing child-row scan as `accept_longest_path`, including
    /// its first-row behavior if a synthetic tree contains duplicate sibling tokens.
    pub fn select_reference(&self, predictions: &[u32]) -> Result<Eagle3DeviceAcceptanceOutput> {
        let nodes = self.tree_tokens.len();
        if nodes == 0
            || nodes > TREE_MAX_NODES
            || self.tree_parent.len() != nodes
            || self.tree_depth.len() != nodes
            || predictions.len() != nodes
        {
            return Err(invalid(format!(
                "EAGLE-3 device acceptance received tree/prediction lengths {}/{}/{}/{}",
                nodes,
                self.tree_parent.len(),
                self.tree_depth.len(),
                predictions.len()
            )));
        }

        let mut path_rows = [u32::MAX; TREE_MAX_NODES];
        let mut emitted_tokens = [u32::MAX; TREE_MAX_NODES];
        let mut current = 0usize;
        for emitted_index in 0..nodes {
            path_rows[emitted_index] = current as u32;
            let next = predictions[current];
            emitted_tokens[emitted_index] = next;
            let matched = (current + 1..nodes).find(|&child| {
                self.tree_parent[child] == current as i32 && self.tree_tokens[child] == next
            });
            if let Some(child) = matched {
                current = child;
                continue;
            }
            let terminal_token_valid = (next as usize) < TARGET_VOCAB_SIZE;
            return Ok(Eagle3DeviceAcceptanceOutput {
                leaf_row: current as u32,
                emitted_count: (emitted_index + 1) as u32,
                terminal_token: next,
                safe_terminal_token: if terminal_token_valid { next } else { 0 },
                terminal_depth: u32::from(self.tree_depth[current]),
                terminal_token_valid,
                path_rows,
                emitted_tokens,
            });
        }
        Err(invalid(
            "EAGLE-3 device acceptance did not terminate within the frozen tree",
        ))
    }
}

/// Fixed-width result written by device acceptance before the pre-encoded N1 terminal graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3DeviceAcceptanceOutput {
    pub leaf_row: u32,
    /// Matched edges plus the final target-only bonus.  This is also the live prefix length of
    /// `path_rows` and `emitted_tokens`.
    pub emitted_count: u32,
    /// Raw target prediction at `leaf_row`, retained for parity diagnostics and token emission.
    pub terminal_token: u32,
    /// `terminal_token` when in vocabulary, otherwise zero.  The pre-encoded embedding gather
    /// binds this device scalar so an impossible UINT_MAX argmax cannot read out of bounds; the
    /// transaction is discarded unless `terminal_token_valid` is true.
    pub safe_terminal_token: u32,
    pub terminal_depth: u32,
    pub terminal_token_valid: bool,
    pub path_rows: [u32; TREE_MAX_NODES],
    pub emitted_tokens: [u32; TREE_MAX_NODES],
}

impl Eagle3DeviceAcceptanceOutput {
    pub fn selected_path(&self) -> &[u32] {
        &self.path_rows[..self.emitted_count as usize]
    }

    pub fn emitted(&self) -> &[u32] {
        &self.emitted_tokens[..self.emitted_count as usize]
    }
}

/// One exact target-authoritative EAGLE cell that can be prepared before target acceptance.
///
/// `predecessor_edge_rows` names the non-root verifier rows whose authoritative edge K/V must
/// precede this cell.  The cell itself is not in that list.  Consequently a Metal implementation
/// can scatter every edge to a private physical slot, use the row list as its tree-attention
/// tail, and later compact only the resolved path into the stable prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3AuthoritativePrecomputeCell {
    pub verifier_row: usize,
    pub token: u32,
    pub capture_row: usize,
    pub logical_position_offset: usize,
    pub predecessor_edge_rows: Vec<usize>,
}

/// Target-blind work available to a future overlapped authoritative EAGLE lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3AuthoritativePrecomputePlan {
    /// Exact verifier-tree identity and the immutable integer buffers consumed by the device
    /// selector. Equal row counts are insufficient because token/parent/depth changes alter both
    /// acceptance and every edge cell's input, ancestry, or RoPE position.
    pub device_acceptance: Eagle3DeviceAcceptancePlan,
    /// One K/V cell per non-root verifier row, ordered by verifier row minus one.
    pub verified_edges: Vec<Eagle3AuthoritativePrecomputeCell>,
    /// At most one full virtual terminal cell per possible accepted endpoint.
    pub terminal_candidates: Vec<Option<Eagle3AuthoritativePrecomputeCell>>,
}

/// Exact post-target disposition of a target-blind authoritative precompute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Eagle3AuthoritativeCommitResolution {
    /// Every committed EAGLE K/V row and the final `stable_seed` were precomputed.
    Complete {
        verified_edge_rows: Vec<usize>,
        terminal_candidate_row: usize,
    },
    /// Accepted-edge K/V is ready, but one final authoritative cell remains on the critical path.
    PrefixOnly {
        verified_edge_rows: Vec<usize>,
        terminal_capture_row: usize,
        terminal_token: u32,
    },
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

/// Portfolio widths measured by the target-blind layer-25 transaction diagnostic.
pub const EAGLE3_TRANSACTION_PORTFOLIO_BUDGETS: [usize; 4] = [1, 2, 4, 8];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3EarlyTransactionCandidate {
    pub path_rows: Vec<usize>,
    pub terminal_token: u32,
    /// Candidate-restricted joint log probability encoded exactly as host f64 bits.
    pub log_probability_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3EarlyPathCandidate {
    pub path_rows: Vec<usize>,
    /// Accepted-edge probability times the marginalized non-child terminal mass.
    pub log_probability_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3EarlyEndpointCandidate {
    pub verifier_row: usize,
    pub terminal_token: u32,
    /// Layer-25 candidate-restricted token log probability at `verifier_row`.
    pub log_probability_bits: u64,
}

/// Authority-free portfolio frozen from one immutable verifier tree and one layer-25 snapshot.
/// The type intentionally has no target-prediction or acceptance field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3EarlyTransactionPortfolio {
    pub layer_id: usize,
    pub tree_tokens: Vec<u32>,
    pub tree_parent: Vec<i32>,
    pub tree_depth: Vec<u16>,
    pub candidate_union: Vec<u32>,
    pub early_rows: Vec<ResidentIndexedHeadEarlyRow>,
    pub transaction_candidates_considered: usize,
    pub path_candidates_considered: usize,
    pub endpoint_candidates_considered: usize,
    pub top_transactions: Vec<Eagle3EarlyTransactionCandidate>,
    pub top_paths: Vec<Eagle3EarlyPathCandidate>,
    pub top_endpoints: Vec<Eagle3EarlyEndpointCandidate>,
    pub indexed_head_encode_us: u128,
    pub indexed_head_commit_wait_us: u128,
    pub indexed_head_gpu_busy_us: u128,
    pub indexed_head_kernel_window_us: u128,
    /// Complete deterministic rank keys retained privately so post-authority evaluation can
    /// report exact global ranks, including misses beyond the largest measured B=8 portfolio.
    ranked_transactions: Vec<(Vec<usize>, u32)>,
    ranked_paths: Vec<Vec<usize>>,
    ranked_endpoints: Vec<(usize, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eagle3EarlyTransactionCoverageAtBudget {
    pub budget: usize,
    pub transaction_covered: bool,
    pub path_covered: bool,
    pub endpoint_covered: bool,
    /// Diagnostic decomposition only: terminal rank after the authoritative leaf is known.
    pub terminal_at_authoritative_leaf_covered: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3EarlyTransactionPortfolioEvaluation {
    pub authoritative_path_rows: Vec<usize>,
    pub authoritative_leaf_row: usize,
    pub authoritative_terminal_token: u32,
    pub transaction_rank: Option<usize>,
    pub path_rank: Option<usize>,
    pub endpoint_rank: Option<usize>,
    /// One-based layer-25 terminal-token rank among non-child candidates at the true leaf.
    /// This is never an input to the primary target-blind portfolio.
    pub terminal_rank_at_authoritative_leaf: Option<usize>,
    pub coverage: [Eagle3EarlyTransactionCoverageAtBudget; 4],
}

/// Authority-free edge set selected from the top path-only portfolio entries.
///
/// Rows are verifier-row ids, not dense scratch slots. They are strictly increasing, exclude
/// the root, and retain the complete verifier-tree identity that produced them. This prevents a
/// same-width plan from being replayed against a different forest. The terminal recurrent cell
/// is deliberately absent: this plan can prepare only independent FC + K/V edge work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3SelectiveEdgePrepPlan {
    pub budget: usize,
    pub tree_tokens: Vec<u32>,
    pub tree_parent: Vec<i32>,
    pub tree_depth: Vec<u16>,
    pub predicted_paths: Vec<Vec<usize>>,
    pub prepared_edge_rows: Vec<usize>,
}

/// Post-authority coverage accounting for a selective edge-preparation plan.
///
/// `theoretical_serial_edge_rows_displaced` counts exact edge rows whose FC + K/V work could be
/// removed after a future byte-identical scratch-to-live commit. The current shadow consumes no
/// private cache bytes, so `actually_reused_edge_rows` must remain zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3SelectiveEdgePrepEvaluation {
    pub budget: usize,
    pub predicted_unique_edge_rows: usize,
    pub authoritative_edge_rows: usize,
    pub prepared_hits: usize,
    pub prepared_misses: usize,
    pub authoritative_path_fully_covered: bool,
    pub theoretical_serial_edge_rows_displaced: usize,
    pub actually_reused_edge_rows: usize,
}

fn eagle3_selective_edge_tree_valid(tree: &TokenTree) -> bool {
    let nodes = tree.nodes();
    if nodes == 0
        || nodes > 8
        || tree.parent.len() != nodes
        || tree.depth.len() != nodes
        || tree.parent[0] != -1
        || tree.depth[0] != 0
        || tree
            .tokens
            .iter()
            .any(|token| *token as usize >= TARGET_VOCAB_SIZE)
    {
        return false;
    }
    (1..nodes).all(|row| {
        usize::try_from(tree.parent[row]).is_ok_and(|parent| {
            parent < row
                && tree.depth[row] > 0
                && tree.depth[parent].checked_add(1) == Some(tree.depth[row])
        })
    })
}

fn eagle3_selective_edge_rows_valid(tree: &TokenTree, rows: &[usize]) -> bool {
    rows.windows(2).all(|pair| pair[0] < pair[1])
        && rows.iter().all(|&row| {
            if row == 0 || row >= tree.nodes() {
                return false;
            }
            let parent = tree.parent[row] as usize;
            parent == 0 || rows.binary_search(&parent).is_ok()
        })
}

fn eagle3_selective_edge_path_union(
    tree: &TokenTree,
    predicted_paths: &[Vec<usize>],
) -> Option<Vec<usize>> {
    if !eagle3_selective_edge_tree_valid(tree) {
        return None;
    }
    let mut prepared_edge_rows = BTreeSet::new();
    for path in predicted_paths {
        let leaf = path.last().copied()?;
        if path.first().copied() != Some(0)
            || path.iter().any(|&row| row >= tree.nodes())
            || path.as_slice() != tree.path_to(leaf)
        {
            return None;
        }
        prepared_edge_rows.extend(path.iter().copied().skip(1));
    }
    Some(prepared_edge_rows.into_iter().collect())
}

fn eagle3_early_ranked_tokens(candidate_union: &[u32], logit_bits: &[u32]) -> Vec<u32> {
    let mut scores = candidate_union
        .iter()
        .copied()
        .zip(logit_bits.iter().copied().map(f32::from_bits))
        .filter(|(_, score)| *score > f32::NEG_INFINITY)
        .collect::<Vec<_>>();
    scores.sort_by(|(left_token, left_score), (right_token, right_score)| {
        if left_score == right_score {
            left_token.cmp(right_token)
        } else {
            right_score.total_cmp(left_score)
        }
    });
    scores.into_iter().map(|(token, _)| token).collect()
}

fn eagle3_candidate_log_probabilities(logit_bits: &[u32]) -> Vec<Option<f64>> {
    let scores = logit_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
    let positive_infinities = scores
        .iter()
        .filter(|score| **score == f32::INFINITY)
        .count();
    if positive_infinities > 0 {
        let winner_log_probability = -(positive_infinities as f64).ln();
        return scores
            .into_iter()
            .map(|score| (score == f32::INFINITY).then_some(winner_log_probability))
            .collect();
    }
    let maximum = scores
        .iter()
        .copied()
        .filter(|score| *score > f32::NEG_INFINITY)
        .max_by(f32::total_cmp);
    let Some(maximum) = maximum else {
        return vec![None; scores.len()];
    };
    let denominator = scores
        .iter()
        .copied()
        .filter(|score| *score > f32::NEG_INFINITY)
        .map(|score| (f64::from(score) - f64::from(maximum)).exp())
        .sum::<f64>();
    let log_denominator = denominator.ln();
    scores
        .into_iter()
        .map(|score| {
            (score > f32::NEG_INFINITY)
                .then_some(f64::from(score) - f64::from(maximum) - log_denominator)
        })
        .collect()
}

fn eagle3_logsumexp(values: &[f64]) -> Option<f64> {
    let maximum = values.iter().copied().max_by(f64::total_cmp)?;
    let sum = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f64>();
    Some(maximum + sum.ln())
}

impl Eagle3EarlyTransactionPortfolio {
    /// Freeze all portfolio rankings using only the immutable N8 tree and layer-25 candidate
    /// scores. Current-round target predictions are absent from this signature by construction.
    pub fn freeze(tree: &TokenTree, early: &ResidentIndexedHeadEarlySnapshot) -> Result<Self> {
        let nodes = tree.nodes();
        if nodes == 0
            || nodes > 8
            || tree.parent.len() != nodes
            || tree.depth.len() != nodes
            || tree.parent[0] != -1
            || tree.depth[0] != 0
            || tree
                .tokens
                .iter()
                .any(|token| *token as usize >= TARGET_VOCAB_SIZE)
        {
            return Err(invalid(
                "EAGLE-3 transaction portfolio received an invalid tree",
            ));
        }
        for row in 1..nodes {
            let parent = usize::try_from(tree.parent[row]).map_err(|_| {
                invalid(format!(
                    "EAGLE-3 transaction portfolio row {row} has no parent"
                ))
            })?;
            if parent >= row
                || tree.depth[row] == 0
                || tree.depth[parent].checked_add(1) != Some(tree.depth[row])
                || (1..row).any(|earlier| {
                    tree.parent[earlier] == tree.parent[row]
                        && tree.tokens[earlier] == tree.tokens[row]
                })
            {
                return Err(invalid(format!(
                    "EAGLE-3 transaction portfolio row {row} has invalid or duplicate ancestry"
                )));
            }
        }
        if let Some(reason) = early.fallback_reason {
            return Err(invalid(format!(
                "EAGLE-3 transaction portfolio layer-25 projection fell back: {}",
                reason.label()
            )));
        }
        if early.layer_id
            != *TARGET_LAYER_INPUT_IDS
                .last()
                .expect("capture contract is non-empty")
            || early.compile_fast_math_enabled
            || early.rows.len() != nodes
            || early.candidate_union.is_empty()
            || early.candidate_union.len() > RESIDENT_INDEXED_HEAD_SHADOW_MAX_CANDIDATES
            || !early
                .candidate_union
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            || early
                .candidate_union
                .iter()
                .any(|token| *token as usize >= TARGET_VOCAB_SIZE)
        {
            return Err(invalid(
                "EAGLE-3 transaction portfolio layer-25 snapshot is unavailable or malformed",
            ));
        }
        for (row, evidence) in early.rows.iter().enumerate() {
            if evidence.verifier_row != row
                || evidence.candidate_logit_bits.len() != early.candidate_union.len()
                || evidence.ranked_candidate_tokens
                    != eagle3_early_ranked_tokens(
                        &early.candidate_union,
                        &evidence.candidate_logit_bits,
                    )
            {
                return Err(invalid(format!(
                    "EAGLE-3 transaction portfolio layer-25 row {row} lost exact scores or rank order"
                )));
            }
        }

        let log_probabilities = early
            .rows
            .iter()
            .map(|row| eagle3_candidate_log_probabilities(&row.candidate_logit_bits))
            .collect::<Vec<_>>();
        let candidate_index = |token: u32| early.candidate_union.binary_search(&token).ok();
        let mut transactions = Vec::<(Eagle3EarlyTransactionCandidate, f64)>::new();
        let mut paths = Vec::<(Eagle3EarlyPathCandidate, f64)>::new();
        let mut endpoints = Vec::<(Eagle3EarlyEndpointCandidate, f64)>::new();
        for verifier_row in 0..nodes {
            let path_rows = tree.path_to(verifier_row);
            let mut edge_log_probability = 0.0f64;
            let mut path_possible = true;
            for edge in path_rows.windows(2) {
                let parent = edge[0];
                let child_token = tree.tokens[edge[1]];
                let Some(log_probability) =
                    candidate_index(child_token).and_then(|index| log_probabilities[parent][index])
                else {
                    path_possible = false;
                    break;
                };
                edge_log_probability += log_probability;
            }
            let child_tokens = (verifier_row + 1..nodes)
                .filter(|child| tree.parent[*child] == verifier_row as i32)
                .map(|child| tree.tokens[child])
                .collect::<BTreeSet<_>>();
            let terminal_candidates = early
                .candidate_union
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(candidate, token)| {
                    if child_tokens.contains(&token) {
                        None
                    } else {
                        log_probabilities[verifier_row][candidate].map(|score| (token, score))
                    }
                })
                .collect::<Vec<_>>();
            let terminal_scores = terminal_candidates
                .iter()
                .map(|(_, score)| *score)
                .collect::<Vec<_>>();
            for (terminal_token, terminal_log_probability) in &terminal_candidates {
                endpoints.push((
                    Eagle3EarlyEndpointCandidate {
                        verifier_row,
                        terminal_token: *terminal_token,
                        log_probability_bits: terminal_log_probability.to_bits(),
                    },
                    *terminal_log_probability,
                ));
            }
            if !path_possible {
                continue;
            }
            if let Some(terminal_mass) = eagle3_logsumexp(&terminal_scores) {
                let score = edge_log_probability + terminal_mass;
                paths.push((
                    Eagle3EarlyPathCandidate {
                        path_rows: path_rows.clone(),
                        log_probability_bits: score.to_bits(),
                    },
                    score,
                ));
            }
            for (terminal_token, terminal_log_probability) in terminal_candidates {
                let transaction_score = edge_log_probability + terminal_log_probability;
                transactions.push((
                    Eagle3EarlyTransactionCandidate {
                        path_rows: path_rows.clone(),
                        terminal_token,
                        log_probability_bits: transaction_score.to_bits(),
                    },
                    transaction_score,
                ));
            }
        }
        transactions.sort_by(|(left, left_score), (right, right_score)| {
            if left_score == right_score {
                left.path_rows
                    .cmp(&right.path_rows)
                    .then_with(|| left.terminal_token.cmp(&right.terminal_token))
            } else {
                right_score.total_cmp(left_score)
            }
        });
        paths.sort_by(|(left, left_score), (right, right_score)| {
            if left_score == right_score {
                left.path_rows.cmp(&right.path_rows)
            } else {
                right_score.total_cmp(left_score)
            }
        });
        endpoints.sort_by(|(left, left_score), (right, right_score)| {
            if left_score == right_score {
                left.verifier_row
                    .cmp(&right.verifier_row)
                    .then_with(|| left.terminal_token.cmp(&right.terminal_token))
            } else {
                right_score.total_cmp(left_score)
            }
        });
        let transaction_candidates_considered = transactions.len();
        let path_candidates_considered = paths.len();
        let endpoint_candidates_considered = endpoints.len();
        let ranked_transactions = transactions
            .iter()
            .map(|(candidate, _)| (candidate.path_rows.clone(), candidate.terminal_token))
            .collect();
        let ranked_paths = paths
            .iter()
            .map(|(candidate, _)| candidate.path_rows.clone())
            .collect();
        let ranked_endpoints = endpoints
            .iter()
            .map(|(candidate, _)| (candidate.verifier_row, candidate.terminal_token))
            .collect();
        let keep = *EAGLE3_TRANSACTION_PORTFOLIO_BUDGETS
            .last()
            .expect("portfolio budgets are non-empty");
        Ok(Self {
            layer_id: early.layer_id,
            tree_tokens: tree.tokens.clone(),
            tree_parent: tree.parent.clone(),
            tree_depth: tree.depth.clone(),
            candidate_union: early.candidate_union.clone(),
            early_rows: early.rows.clone(),
            transaction_candidates_considered,
            path_candidates_considered,
            endpoint_candidates_considered,
            top_transactions: transactions
                .into_iter()
                .take(keep)
                .map(|(candidate, _)| candidate)
                .collect(),
            top_paths: paths
                .into_iter()
                .take(keep)
                .map(|(candidate, _)| candidate)
                .collect(),
            top_endpoints: endpoints
                .into_iter()
                .take(keep)
                .map(|(candidate, _)| candidate)
                .collect(),
            indexed_head_encode_us: early.encode_us,
            indexed_head_commit_wait_us: early.commit_wait_us,
            indexed_head_gpu_busy_us: early.gpu_busy_us,
            indexed_head_kernel_window_us: early.kernel_window_us,
            ranked_transactions,
            ranked_paths,
            ranked_endpoints,
        })
    }

    /// Union the non-root verifier rows from the top-B path-only candidates.
    ///
    /// The ranking was frozen before target authority was observed. A `BTreeSet` both removes
    /// shared-prefix duplication and produces the verifier-row order used by the dense private
    /// scratch mapping. Only the measured B=1/2/4/8 portfolio widths are admitted.
    pub fn plan_selective_edge_prep(&self, budget: usize) -> Result<Eagle3SelectiveEdgePrepPlan> {
        if !EAGLE3_TRANSACTION_PORTFOLIO_BUDGETS.contains(&budget) {
            return Err(invalid(format!(
                "EAGLE-3 selective edge preparation budget must be one of {:?}, got {budget}",
                EAGLE3_TRANSACTION_PORTFOLIO_BUDGETS
            )));
        }
        let tree = TokenTree {
            tokens: self.tree_tokens.clone(),
            parent: self.tree_parent.clone(),
            depth: self.tree_depth.clone(),
        };
        if !eagle3_selective_edge_tree_valid(&tree)
            || self.top_paths.len() != self.path_candidates_considered.min(8)
        {
            return Err(invalid(
                "EAGLE-3 selective edge preparation received a malformed frozen portfolio",
            ));
        }

        let predicted_paths = self
            .top_paths
            .iter()
            .take(budget)
            .map(|candidate| candidate.path_rows.clone())
            .collect::<Vec<_>>();
        let prepared_edge_rows = eagle3_selective_edge_path_union(&tree, &predicted_paths)
            .ok_or_else(|| {
                invalid("EAGLE-3 selective edge preparation found a non-canonical predicted path")
            })?;
        Ok(Eagle3SelectiveEdgePrepPlan {
            budget,
            tree_tokens: self.tree_tokens.clone(),
            tree_parent: self.tree_parent.clone(),
            tree_depth: self.tree_depth.clone(),
            predicted_paths,
            prepared_edge_rows,
        })
    }

    /// Compare a previously frozen portfolio with the target-authoritative transaction. This is
    /// the first API in the pipeline allowed to observe acceptance truth.
    pub fn evaluate(
        &self,
        acceptance: &Eagle3ForestAcceptance,
    ) -> Result<Eagle3EarlyTransactionPortfolioEvaluation> {
        let tree = TokenTree {
            tokens: self.tree_tokens.clone(),
            parent: self.tree_parent.clone(),
            depth: self.tree_depth.clone(),
        };
        if !eagle3_selective_edge_tree_valid(&tree)
            || self.early_rows.len() != tree.nodes()
            || acceptance.emitted_tokens.is_empty()
            || acceptance.leaf_row >= tree.nodes()
            || acceptance.capture_rows != tree.path_to(acceptance.leaf_row)
            || acceptance.emitted_tokens.len() != acceptance.capture_rows.len()
            || acceptance
                .emitted_tokens
                .iter()
                .take(acceptance.emitted_tokens.len() - 1)
                .zip(acceptance.capture_rows.iter().skip(1))
                .any(|(token, row)| *token != tree.tokens[*row])
        {
            return Err(invalid(
                "EAGLE-3 transaction portfolio received malformed authoritative acceptance",
            ));
        }
        let terminal_token = *acceptance
            .emitted_tokens
            .last()
            .expect("non-empty acceptance checked above");
        let child_tokens = (acceptance.leaf_row + 1..tree.nodes())
            .filter(|child| tree.parent[*child] == acceptance.leaf_row as i32)
            .map(|child| tree.tokens[child])
            .collect::<BTreeSet<_>>();
        if child_tokens.contains(&terminal_token) {
            return Err(invalid(
                "EAGLE-3 transaction portfolio acceptance stopped on a matching child token",
            ));
        }
        let transaction_rank = self
            .ranked_transactions
            .iter()
            .position(|(path_rows, candidate_terminal)| {
                path_rows.as_slice() == acceptance.capture_rows.as_slice()
                    && *candidate_terminal == terminal_token
            })
            .map(|rank| rank + 1);
        let path_rank = self
            .ranked_paths
            .iter()
            .position(|path_rows| path_rows.as_slice() == acceptance.capture_rows.as_slice())
            .map(|rank| rank + 1);
        let endpoint_rank = self
            .ranked_endpoints
            .iter()
            .position(|(verifier_row, candidate_terminal)| {
                *verifier_row == acceptance.leaf_row && *candidate_terminal == terminal_token
            })
            .map(|rank| rank + 1);
        let terminal_rank_at_authoritative_leaf = self.early_rows[acceptance.leaf_row]
            .ranked_candidate_tokens
            .iter()
            .filter(|token| !child_tokens.contains(token))
            .position(|token| *token == terminal_token)
            .map(|rank| rank + 1);
        let coverage = EAGLE3_TRANSACTION_PORTFOLIO_BUDGETS.map(|budget| {
            Eagle3EarlyTransactionCoverageAtBudget {
                budget,
                transaction_covered: transaction_rank.is_some_and(|rank| rank <= budget),
                path_covered: path_rank.is_some_and(|rank| rank <= budget),
                endpoint_covered: endpoint_rank.is_some_and(|rank| rank <= budget),
                terminal_at_authoritative_leaf_covered: terminal_rank_at_authoritative_leaf
                    .is_some_and(|rank| rank <= budget),
            }
        });
        Ok(Eagle3EarlyTransactionPortfolioEvaluation {
            authoritative_path_rows: acceptance.capture_rows.clone(),
            authoritative_leaf_row: acceptance.leaf_row,
            authoritative_terminal_token: terminal_token,
            transaction_rank,
            path_rank,
            endpoint_rank,
            terminal_rank_at_authoritative_leaf,
            coverage,
        })
    }
}

impl Eagle3SelectiveEdgePrepPlan {
    /// Measure coverage only after the target-authoritative path is known. No prepared row is
    /// consumed here; the existing serial update remains the sole cache mutation.
    pub fn evaluate(
        &self,
        acceptance: &Eagle3ForestAcceptance,
    ) -> Result<Eagle3SelectiveEdgePrepEvaluation> {
        let tree = TokenTree {
            tokens: self.tree_tokens.clone(),
            parent: self.tree_parent.clone(),
            depth: self.tree_depth.clone(),
        };
        let derived_edge_rows = eagle3_selective_edge_path_union(&tree, &self.predicted_paths);
        if !EAGLE3_TRANSACTION_PORTFOLIO_BUDGETS.contains(&self.budget)
            || self.predicted_paths.len() > self.budget
            || derived_edge_rows.as_deref() != Some(self.prepared_edge_rows.as_slice())
            || acceptance.emitted_tokens.is_empty()
            || acceptance.leaf_row >= tree.nodes()
            || acceptance.capture_rows != tree.path_to(acceptance.leaf_row)
            || acceptance.emitted_tokens.len() != acceptance.capture_rows.len()
            || acceptance
                .emitted_tokens
                .iter()
                .take(acceptance.emitted_tokens.len().saturating_sub(1))
                .zip(acceptance.capture_rows.iter().skip(1))
                .any(|(token, row)| *token != tree.tokens[*row])
            || !eagle3_selective_edge_rows_valid(&tree, &self.prepared_edge_rows)
        {
            return Err(invalid(
                "EAGLE-3 selective edge preparation received a mismatched plan or acceptance",
            ));
        }
        let authoritative_edge_rows = acceptance.capture_rows.len().saturating_sub(1);
        let prepared_hits = acceptance
            .capture_rows
            .iter()
            .skip(1)
            .filter(|row| self.prepared_edge_rows.binary_search(row).is_ok())
            .count();
        let prepared_misses = authoritative_edge_rows.saturating_sub(prepared_hits);
        Ok(Eagle3SelectiveEdgePrepEvaluation {
            budget: self.budget,
            predicted_unique_edge_rows: self.prepared_edge_rows.len(),
            authoritative_edge_rows,
            prepared_hits,
            prepared_misses,
            authoritative_path_fully_covered: prepared_misses == 0,
            theoretical_serial_edge_rows_displaced: prepared_hits,
            actually_reused_edge_rows: 0,
        })
    }
}

/// Receipt-facing counters for the benchmark-only authoritative command-buffer fusion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Eagle3AuthoritativeFusionTelemetry {
    /// Successful decode-time authoritative updates routed through the experimental lane.
    pub fused_updates: u64,
    /// Authoritative rows consumed by those updates.
    pub fused_rows: u64,
    /// Metal command buffers used by those updates. The fused lane contributes exactly one
    /// for every successful update; exposing the count lets a benchmark receipt prove that
    /// the synchronization experiment actually ran.
    pub command_buffers: u64,
}

impl Eagle3AuthoritativeFusionTelemetry {
    fn note_fused_update(&mut self, rows: usize) {
        self.fused_updates = self.fused_updates.saturating_add(1);
        self.fused_rows = self.fused_rows.saturating_add(rows as u64);
        self.command_buffers = self.command_buffers.saturating_add(1);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Eagle3SelectiveEdgePromotionOutcome {
    Promoted(Eagle3SelectiveEdgePromotionReceipt),
    /// The candidate touched no visible watermark. The established serial update then
    /// overwrote the complete accepted range and remains authoritative.
    Fallback {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Eagle3AuthoritativeE1ShadowCounters {
    requested: u64,
    encoded: u64,
    matched: u64,
    mismatched: u64,
    fallback: u64,
    prepared_edges: u64,
    authoritative_edges: u64,
    prepared_hits: u64,
    prepared_misses: u64,
    fully_covered_rounds: u64,
}

static EAGLE3_E1_REQUESTED: AtomicU64 = AtomicU64::new(0);
static EAGLE3_E1_ENCODED: AtomicU64 = AtomicU64::new(0);
static EAGLE3_E1_MATCHED: AtomicU64 = AtomicU64::new(0);
static EAGLE3_E1_MISMATCHED: AtomicU64 = AtomicU64::new(0);
static EAGLE3_E1_FALLBACK: AtomicU64 = AtomicU64::new(0);
static EAGLE3_E1_PREPARED_EDGES: AtomicU64 = AtomicU64::new(0);
static EAGLE3_E1_AUTHORITATIVE_EDGES: AtomicU64 = AtomicU64::new(0);
static EAGLE3_E1_PREPARED_HITS: AtomicU64 = AtomicU64::new(0);
static EAGLE3_E1_PREPARED_MISSES: AtomicU64 = AtomicU64::new(0);
static EAGLE3_E1_FULLY_COVERED_ROUNDS: AtomicU64 = AtomicU64::new(0);

fn record_authoritative_e1_shadow(
    comparison: &Eagle3AuthoritativeE1ShadowComparison,
) -> Eagle3AuthoritativeE1ShadowCounters {
    EAGLE3_E1_REQUESTED.fetch_add(1, Ordering::Relaxed);
    match comparison {
        Eagle3AuthoritativeE1ShadowComparison::Matched {
            prepared_edges,
            authoritative_edges,
            prepared_hits,
            prepared_misses,
            authoritative_path_fully_covered,
            ..
        } => {
            EAGLE3_E1_ENCODED.fetch_add(1, Ordering::Relaxed);
            EAGLE3_E1_MATCHED.fetch_add(1, Ordering::Relaxed);
            EAGLE3_E1_PREPARED_EDGES.fetch_add(*prepared_edges as u64, Ordering::Relaxed);
            EAGLE3_E1_AUTHORITATIVE_EDGES.fetch_add(*authoritative_edges as u64, Ordering::Relaxed);
            EAGLE3_E1_PREPARED_HITS.fetch_add(*prepared_hits as u64, Ordering::Relaxed);
            EAGLE3_E1_PREPARED_MISSES.fetch_add(*prepared_misses as u64, Ordering::Relaxed);
            EAGLE3_E1_FULLY_COVERED_ROUNDS.fetch_add(
                u64::from(*authoritative_path_fully_covered),
                Ordering::Relaxed,
            );
        }
        Eagle3AuthoritativeE1ShadowComparison::Mismatched {
            prepared_edges,
            authoritative_edges,
            prepared_hits,
            prepared_misses,
            authoritative_path_fully_covered,
            ..
        } => {
            EAGLE3_E1_ENCODED.fetch_add(1, Ordering::Relaxed);
            EAGLE3_E1_MISMATCHED.fetch_add(1, Ordering::Relaxed);
            EAGLE3_E1_PREPARED_EDGES.fetch_add(*prepared_edges as u64, Ordering::Relaxed);
            EAGLE3_E1_AUTHORITATIVE_EDGES.fetch_add(*authoritative_edges as u64, Ordering::Relaxed);
            EAGLE3_E1_PREPARED_HITS.fetch_add(*prepared_hits as u64, Ordering::Relaxed);
            EAGLE3_E1_PREPARED_MISSES.fetch_add(*prepared_misses as u64, Ordering::Relaxed);
            EAGLE3_E1_FULLY_COVERED_ROUNDS.fetch_add(
                u64::from(*authoritative_path_fully_covered),
                Ordering::Relaxed,
            );
        }
        Eagle3AuthoritativeE1ShadowComparison::Fallback(_) => {
            EAGLE3_E1_FALLBACK.fetch_add(1, Ordering::Relaxed);
        }
    }
    Eagle3AuthoritativeE1ShadowCounters {
        requested: EAGLE3_E1_REQUESTED.load(Ordering::Relaxed),
        encoded: EAGLE3_E1_ENCODED.load(Ordering::Relaxed),
        matched: EAGLE3_E1_MATCHED.load(Ordering::Relaxed),
        mismatched: EAGLE3_E1_MISMATCHED.load(Ordering::Relaxed),
        fallback: EAGLE3_E1_FALLBACK.load(Ordering::Relaxed),
        prepared_edges: EAGLE3_E1_PREPARED_EDGES.load(Ordering::Relaxed),
        authoritative_edges: EAGLE3_E1_AUTHORITATIVE_EDGES.load(Ordering::Relaxed),
        prepared_hits: EAGLE3_E1_PREPARED_HITS.load(Ordering::Relaxed),
        prepared_misses: EAGLE3_E1_PREPARED_MISSES.load(Ordering::Relaxed),
        fully_covered_rounds: EAGLE3_E1_FULLY_COVERED_ROUNDS.load(Ordering::Relaxed),
    }
}

/// Linear top-1 EAGLE-3 drafter. `stable_seed` is the output of the newest
/// authoritative head-cache row and therefore predicts the first token of the next round.
pub struct Eagle3Drafter {
    head: Eagle3MetalState,
    stable_seed: Option<Eagle3MetalOutput>,
    authoritative_fusion: Eagle3AuthoritativeFusionTelemetry,
}

fn stable_root_target_top_k(output: &Eagle3MetalOutput, count: usize) -> Result<Vec<u32>> {
    if count == 0 || count > crate::metal::EAGLE3_TOP_K_CANDIDATES {
        return Err(invalid(format!(
            "EAGLE-3 stable-root ranking count must be in 1..={}, got {count}",
            crate::metal::EAGLE3_TOP_K_CANDIDATES
        )));
    }
    if output.evaluated_vocab_rows != crate::metal::EAGLE3_DRAFT_VOCAB {
        return Err(invalid(format!(
            "EAGLE-3 stable-root ranking requires all {} draft rows, evaluated {}",
            crate::metal::EAGLE3_DRAFT_VOCAB,
            output.evaluated_vocab_rows
        )));
    }
    if output.top_candidates.len() < count {
        return Err(invalid(format!(
            "EAGLE-3 stable root retained only {} candidates, need {count}",
            output.top_candidates.len()
        )));
    }
    if output.top_candidates[0].target_token != output.target_token {
        return Err(invalid(format!(
            "EAGLE-3 stable root top-1 {} disagrees with selected token {}",
            output.top_candidates[0].target_token, output.target_token
        )));
    }
    let candidates: Vec<u32> = output
        .top_candidates
        .iter()
        .take(count)
        .map(|candidate| candidate.target_token)
        .collect();
    for (index, candidate) in candidates.iter().enumerate() {
        if candidates[..index].contains(candidate) {
            return Err(invalid(format!(
                "EAGLE-3 stable root target token {candidate} appears more than once"
            )));
        }
    }
    Ok(candidates)
}

impl Eagle3Drafter {
    fn forward_authoritative_last_output(
        &mut self,
        embeddings: &[f32],
        fused: &[f32],
        start: usize,
    ) -> Result<Eagle3MetalOutput> {
        if eagle3_full_authoritative_enabled() {
            return metal(self.head.forward_batch(embeddings, fused, start))?
                .pop()
                .ok_or_else(|| invalid("EAGLE-3 authoritative batch produced no output"));
        }
        if eagle3_batch_authoritative_kv_enabled() {
            return metal(
                self.head
                    .forward_batch_last_output_batched_kv(embeddings, fused, start),
            );
        }
        metal(
            self.head
                .forward_batch_last_output(embeddings, fused, start),
        )
    }

    fn accept_authoritative_features(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        features: &[f32],
        emitted: &[u32],
    ) -> Result<()> {
        if emitted.is_empty() {
            return Err(invalid(
                "an EAGLE-3 verify round must emit at least one token",
            ));
        }
        if !features.len().is_multiple_of(EAGLE3_AUX_WIDTH) {
            return Err(invalid(format!(
                "EAGLE-3 authoritative features have {} values, not a multiple of {EAGLE3_AUX_WIDTH}",
                features.len()
            )));
        }
        let rows = features.len() / EAGLE3_AUX_WIDTH;
        if emitted.len() > rows {
            return Err(invalid(format!(
                "EAGLE-3 verify emitted {} tokens but captured only {rows} target rows",
                emitted.len()
            )));
        }
        if eagle3_authoritative_cb_fusion_enabled() {
            validate_authoritative_cb_fusion_dependencies(
                true,
                eagle3_batch_authoritative_kv_enabled(),
                eagle3_full_authoritative_enabled(),
            )?;
            let embeddings = target_weights
                .token_embedding
                .embedding_lookup(emitted, "eagle3_authoritative_next_token_embeddings")?;
            let start = self.head.filled();
            let output = metal(self.head.forward_authoritative_features_last_output_fused(
                &embeddings.data,
                &features[..emitted.len() * EAGLE3_AUX_WIDTH],
                start,
            ))?;
            self.authoritative_fusion.note_fused_update(emitted.len());
            self.stable_seed = Some(output);
            return Ok(());
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

    /// Upload the validated head to Metal. `Eagle3MetalState` owns its uploaded
    /// copies, so serving may retain and share one host checkpoint across
    /// requests without cloning its hundreds of megabytes of buffers.
    pub fn new(model: &Eagle3DraftModel, max_positions: usize) -> Result<Self> {
        validate_drafter_capacity(model.config.sliding_window, max_positions)?;
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
            rope_theta: model.config.rope_theta,
            sliding_window: model.config.sliding_window,
        };
        let head = metal(Eagle3MetalState::new(weights, max_positions))?;
        Ok(Self {
            head,
            stable_seed: None,
            authoritative_fusion: Eagle3AuthoritativeFusionTelemetry::default(),
        })
    }

    pub fn filled(&self) -> usize {
        self.head.filled()
    }

    /// Narrow orchestration seam for the default-off target-overlapped E1 shadow. The target
    /// verifier may install a private diagnostic receipt but cannot advance this head's
    /// watermark, overwrite its cache, or replace `stable_seed`.
    pub fn authoritative_e1_shadow_head_mut(&mut self) -> &mut Eagle3MetalState {
        &mut self.head
    }

    /// Account for a deliberately skipped terminal serial update. E1 remains state-inert and
    /// cannot be compared without that oracle, so consume it as an explicit fallback instead of
    /// carrying the epoch into a later request.
    pub fn abandon_authoritative_e1_shadow_for_terminal_skip(&mut self) {
        if self.head.abandon_authoritative_e1_shadow() {
            let comparison = Eagle3AuthoritativeE1ShadowComparison::Fallback(
                crate::metal::Eagle3AuthoritativeE1ShadowFallbackReason::SerialOracleSkipped,
            );
            let counters = record_authoritative_e1_shadow(&comparison);
            eprintln!(
                "[eagle3-e1-shadow] outcome=fallback route=serial-authoritative \
                 reason=serial_oracle_skipped requested_total={} encoded_total={} \
                 matched_total={} mismatched_total={} fallback_total={}",
                counters.requested,
                counters.encoded,
                counters.matched,
                counters.mismatched,
                counters.fallback,
            );
        }
    }

    /// Capacity of the private one-layer head cache, in positions.
    pub fn max_positions(&self) -> usize {
        self.head.max_positions()
    }

    /// Forget every request-specific row so the uploaded head can serve another
    /// generation. The cache watermark, the stable root seed and the per-run fusion
    /// telemetry are exactly the state [`Self::seed_prompt`] demands of a fresh drafter;
    /// the Metal-resident weights and the cache allocation are untouched, so a reuse
    /// costs no upload. Stale cache bytes are unobservable: every read is bounded by the
    /// watermark and each admitted row overwrites its own slot first.
    pub fn reset_for_reuse(&mut self) {
        self.head.reset();
        self.stable_seed = None;
        self.authoritative_fusion = Eagle3AuthoritativeFusionTelemetry::default();
    }

    /// Decode-time telemetry for the benchmark-only authoritative command-buffer fusion.
    /// Prompt seeding is deliberately excluded and remains on the established control path.
    pub fn authoritative_fusion_telemetry(&self) -> Eagle3AuthoritativeFusionTelemetry {
        self.authoritative_fusion
    }

    /// Return the current stable root's target-vocabulary ranking without advancing the
    /// private EAGLE cache. This is intentionally a read-only seam for benchmark selection:
    /// the returned candidates were produced by the last prompt seed or target-authoritative
    /// update, before the next target verification exists.
    pub fn stable_root_target_top_k(&self, count: usize) -> Result<Vec<u32>> {
        if count == 0 || count > crate::metal::EAGLE3_TOP_K_CANDIDATES {
            return Err(invalid(format!(
                "EAGLE-3 stable-root ranking count must be in 1..={}, got {count}",
                crate::metal::EAGLE3_TOP_K_CANDIDATES
            )));
        }
        let seed = self
            .stable_seed
            .as_ref()
            .ok_or_else(|| invalid("EAGLE-3 must be seeded before reading its stable root"))?;
        stable_root_target_top_k(seed, count)
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
        self.draft_dynamic_frontier_impl(
            target_weights,
            anchor,
            config,
            None::<fn(Eagle3NextExpansionEvidence) -> bool>,
        )
    }

    /// Explore a dynamic frontier while giving a target-blind caller one causal admission point
    /// immediately before each non-root head expansion.
    ///
    /// Returning `false` stops materialization and returns the lattice accumulated so far. The
    /// callback receives only [`Eagle3NextExpansionEvidence`], which was computed from prior
    /// draft-head observations; it cannot inspect current-round target outcomes. The ordinary
    /// [`Self::draft_dynamic_frontier`] path supplies an always-admit gate and is unchanged.
    pub fn draft_dynamic_frontier_with_expansion_gate<F>(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        anchor: u32,
        config: Eagle3DynamicFrontierConfig,
        mut admit_next_expansion: F,
    ) -> Result<Eagle3DynamicFrontier>
    where
        F: FnMut(Eagle3NextExpansionEvidence) -> bool,
    {
        self.draft_dynamic_frontier_impl(
            target_weights,
            anchor,
            config,
            Some(&mut admit_next_expansion),
        )
    }

    fn draft_dynamic_frontier_impl<F>(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        anchor: u32,
        config: Eagle3DynamicFrontierConfig,
        mut expansion_gate: Option<F>,
    ) -> Result<Eagle3DynamicFrontier>
    where
        F: FnMut(Eagle3NextExpansionEvidence) -> bool,
    {
        let stable_seed = self
            .stable_seed
            .clone()
            .ok_or_else(|| invalid("EAGLE-3 must be seeded before dynamic drafting"))?;
        let stable = self.head.filled();
        let mut frontier = Eagle3DynamicFrontier::new(anchor, config)?;
        frontier.record_expansion(0, &stable_seed)?;
        // Benchmark-only: restore already-scored path rows by scatter instead of by forward.
        // Scheduling, admission, scoring, reranking, the verifier and acceptance are untouched;
        // the terminal cell of every path is still a full forward.
        let replay_scatter = eagle3_replay_scatter_enabled();
        // Confidence-gated early exit. Every X-derived sizing parameter (lattice budget, depth,
        // verifier width, head capacity) is fixed by `config` above and stays untouched; the
        // rule below can only end this loop before the expansion budget is spent.
        let early_exit_theta = eagle3_draft_early_exit_theta();

        // Lattice root zero is represented by `stable_seed`, not a private cache row. Each
        // successful non-root materialization below extends this cursor by exactly one row per
        // path node. Regardless of success, the outer cleanup restores the authoritative
        // watermark before returning.
        let mut cursor_path = vec![0usize];
        let materialization = (|| -> Result<Eagle3DynamicFrontier> {
            while let Some(parent) = frontier.next_parent() {
                if let Some(theta) = early_exit_theta {
                    if let Some(evidence) = frontier.early_exit_for_parent(parent, theta) {
                        frontier.record_early_exit(evidence)?;
                        break;
                    }
                }
                if let Some(admit_next_expansion) = expansion_gate.as_mut() {
                    let evidence =
                        frontier
                            .expansion_evidence_for_parent(parent)
                            .ok_or_else(|| {
                                invalid(
                                    "EAGLE-3 dynamic expansion evidence lost its scheduled parent",
                                )
                            })?;
                    if !admit_next_expansion(evidence) {
                        break;
                    }
                }
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
                    // Every path node before `parent` was expanded earlier this round -- that
                    // is how the next node on the path came to exist -- so its distribution is
                    // already in the lattice and only its cache row is needed here. With the
                    // gate armed, restore that row from the key/value retained when it was
                    // scored: one F16 scatter instead of a full head forward. The bytes are
                    // the ones that forward wrote, so the terminal cell below reads the same
                    // history either way.
                    if replay_scatter && source != parent {
                        let (key, value) = frontier.scored_kv(source)?;
                        metal(self.head.commit_scored_row(key, value, self.head.filled()))?;
                        frontier.replay_scatter_commits += 1;
                        continue;
                    }
                    let token = frontier.lattice.nodes()[source].token;
                    let recurrent = frontier.recurrent_g(source)?.to_vec();
                    let embedding = target_weights
                        .token_embedding
                        .embedding_lookup(&[token], "eagle3_dynamic_frontier_token_embedding")?;
                    if replay_scatter {
                        let row = metal(self.head.forward_token_scored(
                            &embedding.data,
                            &recurrent,
                            self.head.filled(),
                        ))?;
                        frontier.materialized_head_forwards += 1;
                        selected_output = Some(row);
                        continue;
                    }
                    let output = metal(self.head.forward_token(
                        &embedding.data,
                        &recurrent,
                        self.head.filled(),
                    ))?;
                    frontier.materialized_head_forwards += 1;
                    if source == parent {
                        selected_output = Some(Eagle3MetalScoredRow {
                            output,
                            key: Vec::new(),
                            value: Vec::new(),
                        });
                    }
                }
                let scored = selected_output.ok_or_else(|| {
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
                if replay_scatter {
                    frontier.record_scored_expansion(parent, &scored)?;
                } else {
                    frontier.record_expansion(parent, &scored.output)?;
                }
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
        let features = interleave_target_layer_inputs(captures)?;
        self.accept_authoritative_features(target_weights, &features, emitted)
    }

    /// Materialize all deferred target-authoritative rows and refresh `stable_seed` once.
    ///
    /// `forward_batch_last_output` appends K/V-only cells for every intermediate row and runs
    /// the complete EAGLE cell only for the newest row. This is exactly the state repeated
    /// authoritative updates would leave: K/V are byte-identical through their F16 scatter,
    /// and no intermediate head output feeds a target-authoritative successor. The buffer is
    /// cleared only after success, so callers can fail closed without losing accepted history.
    pub fn accept_authoritative_catchup(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        pending: &mut Eagle3AuthoritativeCatchup,
    ) -> Result<usize> {
        let rows = pending.pending_rows();
        if rows == 0 {
            return Ok(0);
        }
        self.accept_authoritative_features(
            target_weights,
            &pending.interleaved_features,
            &pending.emitted,
        )?;
        pending.clear();
        Ok(rows)
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
        self.accept_authoritative_forest_with_e1_receipt(
            target_weights,
            all_row_captures,
            acceptance,
        )
        .map(|_| ())
    }

    /// Receipt-returning twin used only by the selective-edge benchmark. The ordinary serving
    /// API above deliberately discards this diagnostic value and keeps its established shape.
    pub fn accept_authoritative_forest_with_e1_receipt(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        all_row_captures: &[CpuTensor],
        acceptance: &Eagle3ForestAcceptance,
    ) -> Result<Option<Eagle3AuthoritativeE1ShadowComparison>> {
        let serial_start = self.head.filled();
        let accepted_captures = acceptance.gather_layer_inputs(all_row_captures)?;
        self.accept_authoritative(
            target_weights,
            &accepted_captures,
            &acceptance.emitted_tokens,
        )?;
        let comparison = self
            .head
            .finish_authoritative_e1_shadow(&acceptance.capture_rows, serial_start);
        if let Some(comparison) = comparison.as_ref() {
            let counters = record_authoritative_e1_shadow(comparison);
            match comparison {
                Eagle3AuthoritativeE1ShadowComparison::Matched {
                    prepared_edges,
                    authoritative_edges,
                    prepared_hits,
                    prepared_misses,
                    authoritative_path_fully_covered,
                    theoretical_serial_edge_rows_displaced,
                    actually_reused_edge_rows,
                    compared_f16_values,
                    timing,
                } => eprintln!(
                    "[eagle3-e1-shadow] outcome=match route=target-tail-selective-or-wide-fc-private-kv-serial-oracle \
                     base={serial_start} prepared_edges={prepared_edges} authoritative_edges={authoritative_edges} \
                     prepared_hits={prepared_hits} prepared_misses={prepared_misses} \
                     authoritative_path_fully_covered={authoritative_path_fully_covered} \
                     theoretical_serial_edge_rows_displaced={theoretical_serial_edge_rows_displaced} \
                     actually_reused_edge_rows={actually_reused_edge_rows} \
                     compared_f16_values={compared_f16_values} target_gpu_us={} target_tail_gpu_us={} \
                     target_tail_baseline_us={:?} target_tail_penalty_us={:?} \
                     selective_path_budget={:?} portfolio_encode_us={} portfolio_commit_wait_us={} \
                     portfolio_gpu_us={} portfolio_kernel_window_us={} total_selective_prep_gpu_us={} \
                     e1_gpu_us={} overlap_us={} e1_encode_us={} e1_post_target_wait_us={} \
                     e1_readback_us={} oracle_compare_us={} \
                     target_prefix_interval={}..{} \
                     target_tail_interval={}..{} e1_interval={}..{} requested_total={} encoded_total={} \
                     matched_total={} mismatched_total={} fallback_total={} prepared_edges_total={} \
                     authoritative_edges_total={} prepared_hits_total={} prepared_misses_total={} \
                     fully_covered_rounds_total={}",
                    timing.target_gpu_us,
                    timing.target_tail_gpu_us,
                    timing.target_tail_baseline_us,
                    timing.target_tail_penalty_us,
                    timing.selective_path_budget,
                    timing.portfolio_encode_us,
                    timing.portfolio_commit_wait_us,
                    timing.portfolio_gpu_us,
                    timing.portfolio_kernel_window_us,
                    timing.total_selective_prep_gpu_us,
                    timing.e1_gpu_us,
                    timing.overlap_us,
                    timing.e1_encode_us,
                    timing.e1_post_target_wait_us,
                    timing.e1_readback_us,
                    timing.oracle_compare_us,
                    timing.target_prefix_start_us,
                    timing.target_prefix_end_us,
                    timing.target_tail_start_us,
                    timing.target_tail_end_us,
                    timing.e1_start_us,
                    timing.e1_end_us,
                    counters.requested,
                    counters.encoded,
                    counters.matched,
                    counters.mismatched,
                    counters.fallback,
                    counters.prepared_edges,
                    counters.authoritative_edges,
                    counters.prepared_hits,
                    counters.prepared_misses,
                    counters.fully_covered_rounds,
                ),
                Eagle3AuthoritativeE1ShadowComparison::Mismatched {
                    reason,
                    prepared_edges,
                    authoritative_edges,
                    prepared_hits,
                    prepared_misses,
                    authoritative_path_fully_covered,
                    theoretical_serial_edge_rows_displaced,
                    actually_reused_edge_rows,
                    timing,
                } => eprintln!(
                    "[eagle3-e1-shadow] outcome=mismatch route=serial-authoritative reason={reason:?} \
                     base={serial_start} prepared_edges={prepared_edges} authoritative_edges={authoritative_edges} \
                     prepared_hits={prepared_hits} prepared_misses={prepared_misses} \
                     authoritative_path_fully_covered={authoritative_path_fully_covered} \
                     theoretical_serial_edge_rows_displaced={theoretical_serial_edge_rows_displaced} \
                     actually_reused_edge_rows={actually_reused_edge_rows} \
                     target_gpu_us={} target_tail_gpu_us={} e1_gpu_us={} overlap_us={} \
                     target_tail_baseline_us={:?} target_tail_penalty_us={:?} \
                     selective_path_budget={:?} portfolio_encode_us={} portfolio_commit_wait_us={} \
                     portfolio_gpu_us={} portfolio_kernel_window_us={} total_selective_prep_gpu_us={} \
                     e1_post_target_wait_us={} e1_readback_us={} \
                     oracle_compare_us={} \
                     requested_total={} encoded_total={} matched_total={} mismatched_total={} \
                     fallback_total={}",
                    timing.target_gpu_us,
                    timing.target_tail_gpu_us,
                    timing.e1_gpu_us,
                    timing.overlap_us,
                    timing.target_tail_baseline_us,
                    timing.target_tail_penalty_us,
                    timing.selective_path_budget,
                    timing.portfolio_encode_us,
                    timing.portfolio_commit_wait_us,
                    timing.portfolio_gpu_us,
                    timing.portfolio_kernel_window_us,
                    timing.total_selective_prep_gpu_us,
                    timing.e1_post_target_wait_us,
                    timing.e1_readback_us,
                    timing.oracle_compare_us,
                    counters.requested,
                    counters.encoded,
                    counters.matched,
                    counters.mismatched,
                    counters.fallback,
                ),
                Eagle3AuthoritativeE1ShadowComparison::Fallback(reason) => eprintln!(
                    "[eagle3-e1-shadow] outcome=fallback route=serial-authoritative reason={} \
                     base={serial_start} requested_total={} encoded_total={} matched_total={} \
                     mismatched_total={} fallback_total={}",
                    reason.label(),
                    counters.requested,
                    counters.encoded,
                    counters.matched,
                    counters.mismatched,
                    counters.fallback,
                ),
            }
        }
        Ok(comparison)
    }

    /// Consume a descriptor-validated B4 device scratch after exact target acceptance.
    ///
    /// A successful candidate publishes only prepared prefix K/V rows, computes every miss and
    /// the terminal row through the authoritative lane, and updates `stable_seed` from that late
    /// terminal cell. An allowlisted precommit decline leaves the watermark at `serial_start`;
    /// the established fused authoritative update then overwrites the complete accepted range
    /// before returning a fallback receipt. Every other failure is fatal to the candidate run.
    pub fn accept_authoritative_forest_with_selective_edge_promotion(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        all_row_captures: &[CpuTensor],
        acceptance: &Eagle3ForestAcceptance,
    ) -> Result<Eagle3SelectiveEdgePromotionOutcome> {
        validate_authoritative_cb_fusion_dependencies(
            eagle3_authoritative_cb_fusion_enabled(),
            eagle3_batch_authoritative_kv_enabled(),
            eagle3_full_authoritative_enabled(),
        )?;
        if !eagle3_authoritative_cb_fusion_enabled() {
            return Err(invalid(
                "selective edge promotion requires authoritative command-buffer fusion",
            ));
        }
        let serial_start = self.head.filled();
        let accepted_captures = acceptance.gather_layer_inputs(all_row_captures)?;
        let features = interleave_target_layer_inputs(&accepted_captures)?;
        let admitted_values = acceptance
            .emitted_tokens
            .len()
            .checked_mul(EAGLE3_AUX_WIDTH)
            .ok_or_else(|| invalid("selective promotion feature length overflow"))?;
        if acceptance.emitted_tokens.is_empty() || admitted_values > features.len() {
            return Err(invalid(
                "selective promotion received an empty or truncated authoritative path",
            ));
        }
        let embeddings = target_weights.token_embedding.embedding_lookup(
            &acceptance.emitted_tokens,
            "eagle3_selective_promotion_next_token_embeddings",
        )?;
        match self
            .head
            .forward_authoritative_features_last_output_selective_promotion(
                &embeddings.data,
                &features[..admitted_values],
                &acceptance.emitted_tokens,
                &acceptance.capture_rows,
                serial_start,
            ) {
            Ok(Eagle3SelectiveEdgePromotionAttempt::Promoted { output, receipt }) => {
                self.authoritative_fusion
                    .note_fused_update(acceptance.emitted_tokens.len());
                self.stable_seed = Some(output);
                eprintln!(
                    "[eagle3-selective-promotion] outcome=promoted base={serial_start} \
                     authorization_generation={} authorized_stable_position={} \
                     prepared_edge_rows={:?} authoritative_path_rows={:?} \
                     promoted_edge_rows={:?} serial_miss_edge_rows={:?} \
                     prepared_edges={} authoritative_edges={} promoted_hits={} serial_misses={} \
                     full_coverage={} compacted_bytes={} additional_compaction_command_buffers={} \
                     compaction_encode_us={} compaction_gpu_us={:?} authoritative_update_gpu_us={} \
                     logical_serial_fc_rows_displaced={} saved_serial_kv_rows={} \
                     serial_fc_logical_columns={} serial_fc_physical_columns={} \
                     terminal_authoritative_rows={} proof_receipt_sha256={} \
                     e1_kv_host_readback=false",
                    receipt.authorization_generation,
                    receipt.authorized_stable_position,
                    receipt.prepared_edge_rows,
                    receipt.authoritative_path_rows,
                    receipt.promoted_edge_rows,
                    receipt.serial_miss_edge_rows,
                    receipt.prepared_edges,
                    receipt.authoritative_edges,
                    receipt.promoted_hits,
                    receipt.serial_misses,
                    receipt.authoritative_path_fully_covered,
                    receipt.compacted_bytes,
                    receipt.additional_compaction_command_buffers,
                    receipt.compaction_encode_us,
                    receipt.compaction_gpu_us,
                    receipt.authoritative_update_gpu_us,
                    receipt.logical_serial_fc_rows_displaced,
                    receipt.saved_serial_kv_rows,
                    receipt.serial_fc_logical_columns,
                    receipt.serial_fc_physical_columns,
                    receipt.terminal_authoritative_rows,
                    receipt.proof_receipt_sha256,
                );
                Ok(Eagle3SelectiveEdgePromotionOutcome::Promoted(receipt))
            }
            Ok(Eagle3SelectiveEdgePromotionAttempt::Declined { reason }) => {
                if self.head.filled() != serial_start {
                    return Err(invalid(format!(
                        "selective promotion decline advanced the EAGLE watermark from {serial_start} to {}",
                        self.head.filled()
                    )));
                }
                self.accept_authoritative(
                    target_weights,
                    &accepted_captures,
                    &acceptance.emitted_tokens,
                )?;
                eprintln!(
                    "[eagle3-selective-promotion] outcome=fallback route=serial-authoritative \
                     base={serial_start} reason={reason:?}"
                );
                Ok(Eagle3SelectiveEdgePromotionOutcome::Fallback { reason })
            }
            Err(reason) => Err(invalid(format!(
                "selective promotion failed outside its serial-fallback allowlist at EAGLE watermark {}/{}: {reason}",
                serial_start,
                self.head.filled(),
            ))),
        }
    }

    /// Consume an unobservable private promotion artifact when the request ends before the
    /// authoritative head update. The caller records the explicit terminal-skip fallback.
    pub fn abandon_selective_edge_promotion_for_terminal_skip(&mut self) -> bool {
        self.head.abandon_authoritative_e1_shadow()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn replay_scatter_gate_is_default_off_and_fails_closed() {
        assert!(!super::eagle3_replay_scatter_enabled_from(None));
        assert!(!super::eagle3_replay_scatter_enabled_from(Some("")));
        assert!(!super::eagle3_replay_scatter_enabled_from(Some("0")));
        assert!(!super::eagle3_replay_scatter_enabled_from(Some("true")));
        assert!(!super::eagle3_replay_scatter_enabled_from(Some("yes")));
        assert!(!super::eagle3_replay_scatter_enabled_from(Some("11")));
        assert!(super::eagle3_replay_scatter_enabled_from(Some("1")));
        assert!(super::eagle3_replay_scatter_enabled_from(Some(" 1 ")));
    }

    use super::*;
    use crate::metal::Eagle3DraftCandidate;

    #[test]
    fn authoritative_kv_batch_gate_is_explicit_and_fail_closed() {
        for enabled in ["1", "true", "TRUE", " on ", "Yes", "enabled"] {
            assert!(eagle3_batch_authoritative_kv_enabled_from(Some(enabled)));
        }
        for disabled in ["", "0", "false", "off", "no", "batch"] {
            assert!(!eagle3_batch_authoritative_kv_enabled_from(Some(disabled)));
        }
        assert!(!eagle3_batch_authoritative_kv_enabled_from(None));
    }

    #[test]
    fn authoritative_cb_fusion_gate_is_default_off_and_dependencies_are_strict() {
        for enabled in ["1", "true", "TRUE", " on ", "Yes", "enabled"] {
            assert!(eagle3_authoritative_cb_fusion_enabled_from(Some(enabled)));
        }
        for disabled in ["", "0", "false", "off", "no", "fuse", "garbage"] {
            assert!(!eagle3_authoritative_cb_fusion_enabled_from(Some(disabled)));
        }
        assert!(!eagle3_authoritative_cb_fusion_enabled_from(None));

        validate_authoritative_cb_fusion_dependencies(false, false, true).unwrap();
        validate_authoritative_cb_fusion_dependencies(true, true, false).unwrap();
        let missing_batch =
            validate_authoritative_cb_fusion_dependencies(true, false, false).unwrap_err();
        assert!(
            missing_batch
                .to_string()
                .contains("CAMELID_EAGLE3_BATCH_AUTHORITATIVE_KV=1"),
            "{missing_batch}"
        );
        let full = validate_authoritative_cb_fusion_dependencies(true, true, true).unwrap_err();
        assert!(
            full.to_string()
                .contains("CAMELID_EAGLE3_FULL_AUTHORITATIVE=1"),
            "{full}"
        );
    }

    #[test]
    fn authoritative_cb_fusion_telemetry_attests_one_buffer_per_update() {
        let mut telemetry = Eagle3AuthoritativeFusionTelemetry::default();
        telemetry.note_fused_update(1);
        telemetry.note_fused_update(4);
        assert_eq!(telemetry.fused_updates, 2);
        assert_eq!(telemetry.fused_rows, 5);
        assert_eq!(telemetry.command_buffers, telemetry.fused_updates);
    }

    #[test]
    fn sliding_window_limits_attention_span_not_cache_capacity() {
        for capacity in [1, 255, 256, 257, 2_048, 131_072] {
            validate_drafter_capacity(Some(256), capacity).unwrap();
        }
        validate_drafter_capacity(None, 131_072).unwrap();

        let error = validate_drafter_capacity(Some(0), 2_048).unwrap_err();
        assert!(error.to_string().contains("zero-position"), "{error}");
        assert!(validate_drafter_capacity(None, 0).is_err());
    }

    fn capture(name: &str, rows: usize, base: f32) -> CpuTensor {
        let data = (0..rows * HIDDEN_SIZE)
            .map(|index| base + index as f32)
            .collect();
        CpuTensor::from_f32(name, vec![rows, HIDDEN_SIZE], data).unwrap()
    }

    #[test]
    fn authoritative_catchup_keeps_only_each_emitted_capture_prefix_in_order() {
        let first = vec![
            capture("first_low", 3, 1.0),
            capture("first_middle", 3, 10_000.0),
            capture("first_high", 3, 20_000.0),
        ];
        let second = vec![
            capture("second_low", 2, 30_000.0),
            capture("second_middle", 2, 40_000.0),
            capture("second_high", 2, 50_000.0),
        ];
        let first_features = interleave_target_layer_inputs(&first).unwrap();
        let second_features = interleave_target_layer_inputs(&second).unwrap();

        let mut pending = Eagle3AuthoritativeCatchup::default();
        pending.push(&first, &[101, 102]).unwrap();
        pending.push(&second, &[103]).unwrap();

        let mut expected = first_features[..2 * EAGLE3_AUX_WIDTH].to_vec();
        expected.extend_from_slice(&second_features[..EAGLE3_AUX_WIDTH]);
        assert_eq!(pending.emitted, vec![101, 102, 103]);
        assert_eq!(pending.interleaved_features, expected);
        assert_eq!(pending.pending_rows(), 3);
        assert_eq!(pending.effective_filled(17).unwrap(), 20);
    }

    #[test]
    fn authoritative_catchup_rejects_invalid_rounds_transactionally() {
        let one_row = vec![
            capture("low", 1, 1.0),
            capture("middle", 1, 10_000.0),
            capture("high", 1, 20_000.0),
        ];
        let mut pending = Eagle3AuthoritativeCatchup::default();
        pending.push(&one_row, &[7]).unwrap();
        let before = pending.clone();

        assert!(pending.push(&one_row, &[8, 9]).is_err());
        assert_eq!(pending, before);
        assert!(pending.push(&one_row, &[]).is_err());
        assert_eq!(pending, before);
        assert!(pending.push(&one_row[..2], &[8]).is_err());
        assert_eq!(pending, before);
        assert!(pending.effective_filled(usize::MAX).is_err());
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

    #[test]
    fn stable_root_ranking_is_read_only_full_vocab_and_fail_closed() {
        let root = output(
            &[
                (101, 0.30),
                (102, 0.20),
                (103, 0.15),
                (104, 0.10),
                (105, 0.08),
                (106, 0.06),
                (107, 0.05),
                (108, 0.03),
            ],
            1.0,
        );
        assert_eq!(
            stable_root_target_top_k(&root, 8).unwrap(),
            vec![101, 102, 103, 104, 105, 106, 107, 108]
        );
        assert!(stable_root_target_top_k(&root, 0).is_err());

        let mut reduced = root.clone();
        reduced.evaluated_vocab_rows -= 1;
        assert!(stable_root_target_top_k(&reduced, 8).is_err());

        let mut mismatched_top1 = root.clone();
        mismatched_top1.target_token = 999;
        assert!(stable_root_target_top_k(&mismatched_top1, 8).is_err());

        let mut duplicate = root;
        duplicate.top_candidates[7].target_token = duplicate.top_candidates[0].target_token;
        assert!(stable_root_target_top_k(&duplicate, 8).is_err());
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
            adaptive_branching: false,
            certified_argmax_shadow: false,
        }
    }

    #[test]
    fn adaptive_branching_prunes_weak_siblings_when_top1_is_dominant() {
        let mut config = frontier_config(8, 16, 4);
        config.adaptive_branching = true;
        let mut frontier = Eagle3DynamicFrontier::new(10, config).unwrap();
        // Candidate 0 has probability ~0.80 (dominant). Weak siblings (0.05, 0.03, ...) should be pruned.
        let exp = output(&[(11, 0.80), (12, 0.05), (13, 0.03), (14, 0.02)], 1.0);
        let children = frontier.record_expansion(0, &exp).unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(frontier.lattice().nodes().len(), 2);
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
    fn certified_argmax_shadow_retains_unadmitted_candidates_without_changing_the_tree() {
        let mut off_config = frontier_config(2, 2, 1);
        off_config.candidates_per_parent = 1;
        let mut on_config = off_config;
        on_config.certified_argmax_shadow = true;
        let root = output(&[(13, 0.50), (11, 0.30), (12, 0.10)], 1.0);

        let mut off = Eagle3DynamicFrontier::new(10, off_config).unwrap();
        off.record_expansion(0, &root).unwrap();
        let off_forest = off.finish().unwrap();

        let mut on = Eagle3DynamicFrontier::new(10, on_config).unwrap();
        on.record_expansion(0, &root).unwrap();
        let selected = on
            .select_for_verifier_costs(&[Eagle3VerifierBudgetCost {
                max_nodes: 2,
                round_cost: 1.0,
            }])
            .unwrap();
        let on_forest = on.finish().unwrap();

        assert_eq!(off_forest.scored, on_forest.scored);
        assert_eq!(off_forest.packed_plan, on_forest.packed_plan);
        assert_eq!(
            off_forest.accept_target_predictions(&[13, 99]).unwrap(),
            on_forest.accept_target_predictions(&[13, 99]).unwrap()
        );
        assert!(off_forest.certified_argmax_shadow.is_none());
        assert_eq!(
            selected.forest.certified_argmax_shadow,
            on_forest.certified_argmax_shadow
        );
        let shadow = on_forest.certified_argmax_shadow.unwrap();
        assert_eq!(shadow.target_token_union(), vec![11, 12, 13]);
        assert_eq!(shadow.candidate_observations(), 3);
        assert_eq!(shadow.lattice_admitted_observations(), 1);
        assert_eq!(shadow.expansions.len(), 1);
        assert_eq!(shadow.expansions[0].parent_source_node, 0);
        assert_eq!(
            shadow.expansions[0].candidate_target_tokens,
            vec![13, 11, 12]
        );
        assert_eq!(
            shadow.expansions[0].lattice_admitted_target_tokens,
            vec![13]
        );
        assert_eq!(on_forest.scored.tree.tokens, vec![10, 13]);
    }

    #[test]
    fn certified_argmax_shadow_rejects_malformed_ids_before_mutating_the_frontier() {
        let mut config = frontier_config(2, 2, 1);
        config.candidates_per_parent = 1;
        config.certified_argmax_shadow = true;

        assert!(Eagle3DynamicFrontier::new(TARGET_VOCAB_SIZE as u32, config).is_err());

        let mut invalid_target = output(&[(11, 0.60), (12, 0.30)], 1.0);
        invalid_target.top_candidates[1].target_token = TARGET_VOCAB_SIZE as u32;
        let mut frontier = Eagle3DynamicFrontier::new(10, config).unwrap();
        assert!(frontier.record_expansion(0, &invalid_target).is_err());
        assert_eq!(frontier.lattice().nodes().len(), 1);
        assert_eq!(frontier.head_expansions(), 0);
        assert!(frontier
            .certified_argmax_shadow()
            .unwrap()
            .expansions
            .is_empty());

        let mut invalid_draft = output(&[(11, 0.60), (12, 0.30)], 1.0);
        invalid_draft.draft_token = EAGLE3_DRAFT_VOCAB as u32;
        let mut frontier = Eagle3DynamicFrontier::new(10, config).unwrap();
        assert!(frontier.record_expansion(0, &invalid_draft).is_err());
        assert_eq!(frontier.lattice().nodes().len(), 1);
        assert_eq!(frontier.head_expansions(), 0);
        assert!(frontier
            .certified_argmax_shadow()
            .unwrap()
            .expansions
            .is_empty());
    }

    #[test]
    fn next_expansion_evidence_is_causal_and_tracks_the_global_parent() {
        let mut frontier = Eagle3DynamicFrontier::new(10, frontier_config(6, 12, 4)).unwrap();
        let initial = frontier.next_expansion_evidence().unwrap();
        assert_eq!(initial.completed_head_expansions, 0);
        assert_eq!(initial.next_parent, 0);
        assert_eq!(initial.next_parent_depth, 0);
        assert_eq!(initial.next_parent_cumulative_probability, 1.0);

        frontier
            .record_expansion(0, &output(&[(11, 0.50), (12, 0.30)], 1.0))
            .unwrap();
        let after_root = frontier.next_expansion_evidence().unwrap();
        assert_eq!(after_root.completed_head_expansions, 1);
        assert_eq!(after_root.next_parent, 1);
        assert_eq!(after_root.next_parent_depth, 1);
        assert!((after_root.next_parent_cumulative_probability - 0.50).abs() < 1.0e-6);

        frontier
            .record_expansion(1, &output(&[(13, 0.40), (14, 0.20)], 2.0))
            .unwrap();
        frontier
            .record_expansion(2, &output(&[(15, 0.90)], 3.0))
            .unwrap();
        let before_fourth = frontier.next_expansion_evidence().unwrap();
        assert_eq!(before_fourth.completed_head_expansions, 3);
        assert_eq!(before_fourth.next_parent, 5);
        assert_eq!(before_fourth.next_parent_depth, 2);
        assert!((before_fourth.next_parent_cumulative_probability - 0.27).abs() < 1.0e-6);
        assert!((before_fourth.next_parent_cumulative_log_probability.exp() - 0.27).abs() < 1.0e-6);
    }

    #[test]
    fn draft_early_exit_gate_is_default_off_and_fails_closed() {
        use super::eagle3_draft_early_exit_theta_from as parse;
        assert_eq!(parse(None), Ok(None));
        assert_eq!(parse(Some("")), Ok(None));
        assert_eq!(parse(Some("   ")), Ok(None));
        assert_eq!(parse(Some("0")), Ok(None));
        assert_eq!(parse(Some("0.0")), Ok(None));
        assert_eq!(parse(Some("0.30")), Ok(Some(0.30)));
        assert_eq!(parse(Some(" 0.25 ")), Ok(Some(0.25)));
        assert_eq!(parse(Some("1")), Ok(Some(1.0)));
        for malformed in [
            "abc", "NaN", "inf", "-inf", "-0.1", "1.5", "true", "on", "0,3",
        ] {
            assert!(
                parse(Some(malformed)).is_err(),
                "{malformed:?} must be rejected so the gate fails closed"
            );
        }
    }

    /// Drive a frontier the way `draft_dynamic_frontier_impl` does, minus the Metal head:
    /// each scheduled parent is answered from `observations` in schedule order, and the
    /// confidence-gated early exit is consulted before every non-root expansion.
    fn drive_frontier(
        config: Eagle3DynamicFrontierConfig,
        observations: &[Eagle3MetalOutput],
        theta: Option<f64>,
    ) -> Eagle3DynamicFrontier {
        let mut frontier = Eagle3DynamicFrontier::new(10, config).unwrap();
        while let Some(parent) = frontier.next_parent() {
            if let Some(theta) = theta {
                if let Some(evidence) = frontier.early_exit_for_parent(parent, theta) {
                    frontier.record_early_exit(evidence).unwrap();
                    break;
                }
            }
            let observation = &observations[frontier.head_expansions()];
            frontier.record_expansion(parent, observation).unwrap();
        }
        frontier
    }

    /// Root .50/.30/.15/.05; under 11: .28/.15; under 12: .24/.06; under 21: .252/.028;
    /// under 41: .2268/.0252. The globally strongest unexpanded node before each non-root
    /// expansion is therefore 11 (.50), 12 (.30), 21 (.28), 41 (.252) in schedule order, with
    /// no probability ties anywhere so the expected schedule does not depend on rounding.
    fn study_observations() -> Vec<Eagle3MetalOutput> {
        vec![
            output(&[(11, 0.50), (12, 0.30), (13, 0.15), (14, 0.05)], 1.0),
            output(&[(21, 0.56), (22, 0.30)], 2.0),
            output(&[(31, 0.80), (32, 0.20)], 3.0),
            output(&[(41, 0.90), (42, 0.10)], 4.0),
            output(&[(51, 0.90), (52, 0.10)], 5.0),
        ]
    }

    #[test]
    fn early_exit_stops_exactly_when_best_unexpanded_joint_probability_is_below_theta() {
        let config = frontier_config(8, 21, 5);
        let mut frontier = Eagle3DynamicFrontier::new(10, config).unwrap();
        // The root distribution is free: no threshold, not even one, declines it.
        assert_eq!(frontier.next_parent(), Some(0));
        assert_eq!(frontier.early_exit_before_next_expansion(1.0), None);

        let observations = study_observations();
        frontier.record_expansion(0, &observations[0]).unwrap();
        // Before the first materialized expansion the scheduler wants 11 at joint .50.
        assert_eq!(frontier.next_parent(), Some(1));
        let scheduled = frontier.next_expansion_evidence().unwrap();
        let joint = f64::from(scheduled.next_parent_cumulative_probability);
        assert!((joint - 0.50).abs() < 1.0e-6);
        assert_eq!(frontier.early_exit_before_next_expansion(0.0), None);
        // Strictly below: equality keeps expanding, the next representable step stops.
        assert_eq!(frontier.early_exit_before_next_expansion(joint), None);
        let exit = frontier
            .early_exit_before_next_expansion(joint + 1.0e-9)
            .unwrap();
        assert_eq!(exit, scheduled);
        assert_eq!(exit.completed_head_expansions, 1);
        assert_eq!(exit.next_parent, 1);
        assert_eq!(exit.next_parent_depth, 1);
        // The rule is read-only: nothing about the frontier changed.
        assert_eq!(frontier.draft_early_exit(), None);
        assert_eq!(frontier.head_expansions(), 1);
        assert_eq!(frontier.next_parent(), Some(1));
        // Recording an exit fails closed unless it names exactly the scheduled expansion.
        let mut stale = scheduled;
        stale.completed_head_expansions += 1;
        assert!(frontier.record_early_exit(stale).is_err());
        let mut wrong_parent = scheduled;
        wrong_parent.next_parent = 2;
        assert!(frontier.record_early_exit(wrong_parent).is_err());
        assert_eq!(frontier.draft_early_exit(), None);

        // After expanding 11 (children .28/.15) the best unexpanded node is 12 at .30.
        frontier.record_expansion(1, &observations[1]).unwrap();
        assert_eq!(frontier.next_parent(), Some(2));
        assert_eq!(frontier.early_exit_before_next_expansion(0.29), None);
        let exit = frontier.early_exit_before_next_expansion(0.31).unwrap();
        assert_eq!(exit.completed_head_expansions, 2);
        assert_eq!(exit.next_parent, 2);
        assert!((exit.next_parent_cumulative_probability - 0.30).abs() < 1.0e-6);
    }

    #[test]
    fn early_exit_with_theta_zero_is_the_ungated_scheduler() {
        let config = frontier_config(8, 21, 5);
        let observations = study_observations();
        let ungated = drive_frontier(config, &observations, None);
        let armed_never_fires = drive_frontier(config, &observations, Some(0.0));
        assert_eq!(ungated.head_expansions(), 5);
        assert_eq!(ungated.draft_early_exit(), None);
        assert_eq!(armed_never_fires, ungated);
        assert_eq!(
            armed_never_fires.finish().unwrap().scored,
            ungated.finish().unwrap().scored
        );
        // Sanity: the same schedule with the existing (pre-gate) tests' lattices.
        let sparse = frontier_config(5, 7, 3);
        let sparse_observations = vec![
            output(&[(11, 0.50), (12, 0.30)], 1.0),
            output(&[(13, 0.50), (14, 0.40)], 2.0),
            output(&[(15, 0.50), (16, 0.25)], 3.0),
        ];
        assert_eq!(
            drive_frontier(sparse, &sparse_observations, Some(0.0)),
            drive_frontier(sparse, &sparse_observations, None)
        );
    }

    #[test]
    fn early_exit_truncates_the_x5_lattice_without_touching_its_sizing() {
        let x5 = frontier_config(8, 21, 5);
        let observations = study_observations();
        // theta .31 admits expansion two (11 at .50), declines expansion three (12 at .30).
        let exited = drive_frontier(x5, &observations, Some(0.31));
        assert_eq!(exited.head_expansions(), 2);
        assert_eq!(exited.config(), x5);
        let exit = exited.draft_early_exit().unwrap();
        assert_eq!(exit.completed_head_expansions, 2);
        assert_eq!(exit.next_parent, 2);
        assert!((exit.next_parent_cumulative_probability - 0.30).abs() < 1.0e-6);
        // The verifier sees exactly the lattice the offline study replays: the X5 run cut
        // after two expansions, re-admitted with the unchanged N and depth.
        let truncated = drive_frontier(frontier_config(8, 21, 2), &observations, None);
        assert_eq!(truncated.draft_early_exit(), None);
        assert_eq!(exited.lattice(), truncated.lattice());
        assert_eq!(
            exited.finish_borrowed().unwrap().scored,
            truncated.finish_borrowed().unwrap().scored
        );
        // Lower thresholds walk further (21 at .28, then 41 at .252); a threshold above every
        // joint stops before the first head forward.
        for (theta, expansions) in [(0.29, 3), (0.26, 4), (0.25, 5), (0.20, 5)] {
            assert_eq!(
                drive_frontier(x5, &observations, Some(theta)).head_expansions(),
                expansions,
                "theta {theta}"
            );
        }
        let immediate = drive_frontier(x5, &observations, Some(1.0));
        assert_eq!(immediate.head_expansions(), 1);
        assert_eq!(immediate.draft_early_exit().unwrap().next_parent, 1);
        assert_eq!(
            immediate.finish().unwrap().scored.tree.tokens,
            vec![10, 11, 12, 13, 14]
        );
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
    fn authoritative_precompute_maps_parent_captures_to_child_tokens() {
        let mut frontier = Eagle3DynamicFrontier::new(10, frontier_config(4, 4, 2)).unwrap();
        frontier
            .record_expansion(0, &output(&[(11, 0.60), (12, 0.40)], 1.0))
            .unwrap();
        frontier
            .record_expansion(1, &output(&[(13, 0.90)], 2.0))
            .unwrap();
        let forest = frontier.finish().unwrap();
        assert_eq!(forest.scored.tree.tokens, vec![10, 11, 12, 13]);

        let plan = forest
            .plan_authoritative_precompute(&[Some(99), Some(88), Some(99), Some(77)])
            .unwrap();
        assert_eq!(
            plan.verified_edges,
            vec![
                Eagle3AuthoritativePrecomputeCell {
                    verifier_row: 1,
                    token: 11,
                    capture_row: 0,
                    logical_position_offset: 0,
                    predecessor_edge_rows: vec![],
                },
                Eagle3AuthoritativePrecomputeCell {
                    verifier_row: 2,
                    token: 12,
                    capture_row: 0,
                    logical_position_offset: 0,
                    predecessor_edge_rows: vec![],
                },
                Eagle3AuthoritativePrecomputeCell {
                    verifier_row: 3,
                    token: 13,
                    capture_row: 1,
                    logical_position_offset: 1,
                    predecessor_edge_rows: vec![1],
                },
            ]
        );
        assert_eq!(
            plan.terminal_candidates[3],
            Some(Eagle3AuthoritativePrecomputeCell {
                verifier_row: 3,
                token: 77,
                capture_row: 3,
                logical_position_offset: 2,
                predecessor_edge_rows: vec![1, 3],
            })
        );

        // Root -> row 1 -> row 3, then target-only bonus 77. Every authoritative cell is
        // already represented: edge rows 1/3 followed by virtual terminal row 3.
        let acceptance = forest.accept_target_predictions(&[11, 13, 0, 77]).unwrap();
        assert_eq!(
            forest
                .resolve_authoritative_precompute(&plan, &acceptance)
                .unwrap(),
            Eagle3AuthoritativeCommitResolution::Complete {
                verified_edge_rows: vec![1, 3],
                terminal_candidate_row: 3,
            }
        );
    }

    #[test]
    fn authoritative_precompute_reuses_prefix_on_terminal_miss() {
        let mut frontier = Eagle3DynamicFrontier::new(10, frontier_config(4, 4, 2)).unwrap();
        frontier
            .record_expansion(0, &output(&[(11, 0.60), (12, 0.40)], 1.0))
            .unwrap();
        frontier
            .record_expansion(1, &output(&[(13, 0.90)], 2.0))
            .unwrap();
        let forest = frontier.finish().unwrap();
        let plan = forest
            .plan_authoritative_precompute(&[Some(98), Some(88), Some(98), None])
            .unwrap();

        // The target takes row 2 but predicts 99 there. Row 2's exact authoritative K/V is
        // reusable; only `(embedding[99], captures[row 2])` needs a post-target full cell.
        let acceptance = forest.accept_target_predictions(&[12, 0, 99, 0]).unwrap();
        assert_eq!(
            forest
                .resolve_authoritative_precompute(&plan, &acceptance)
                .unwrap(),
            Eagle3AuthoritativeCommitResolution::PrefixOnly {
                verified_edge_rows: vec![2],
                terminal_capture_row: 2,
                terminal_token: 99,
            }
        );

        // A terminal candidate must be outside the row's verified children. If it equals a
        // child, target acceptance would continue to that child and the virtual cell can never
        // be selected as the endpoint commit.
        assert!(forest
            .plan_authoritative_precompute(&[Some(11), None, None, None])
            .is_err());
        assert!(forest
            .plan_authoritative_precompute(&[None, None, None])
            .is_err());
    }

    fn transaction_portfolio_fixture() -> (TokenTree, ResidentIndexedHeadEarlySnapshot) {
        let tree = TokenTree {
            tokens: vec![10, 11, 12, 13],
            parent: vec![-1, 0, 0, 1],
            depth: vec![0, 1, 1, 2],
        };
        let candidate_union = vec![11, 12, 13, 90, 91];
        let rows = [
            [f32::INFINITY, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, f32::INFINITY, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, f32::INFINITY],
            [0.0, 0.0, 0.0, f32::INFINITY, 0.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(verifier_row, logits)| {
            let candidate_logit_bits = logits.map(f32::to_bits).to_vec();
            ResidentIndexedHeadEarlyRow {
                verifier_row,
                ranked_candidate_tokens: eagle3_early_ranked_tokens(
                    &candidate_union,
                    &candidate_logit_bits,
                ),
                candidate_logit_bits,
            }
        })
        .collect();
        (
            tree,
            ResidentIndexedHeadEarlySnapshot {
                layer_id: 25,
                compile_fast_math_enabled: false,
                candidate_union,
                encode_us: 11,
                commit_wait_us: 22,
                gpu_busy_us: 19,
                kernel_window_us: 20,
                fallback_reason: None,
                rows,
            },
        )
    }

    #[test]
    fn transaction_portfolio_freezes_joint_path_and_endpoint_rankings_without_truth() {
        let (tree, early) = transaction_portfolio_fixture();
        let portfolio = Eagle3EarlyTransactionPortfolio::freeze(&tree, &early).unwrap();
        assert_eq!(portfolio.transaction_candidates_considered, 1);
        assert_eq!(portfolio.path_candidates_considered, 1);
        assert_eq!(portfolio.endpoint_candidates_considered, 2);
        assert_eq!(portfolio.top_transactions[0].path_rows, vec![0, 1, 3]);
        assert_eq!(portfolio.top_transactions[0].terminal_token, 90);
        assert_eq!(portfolio.top_paths[0].path_rows, vec![0, 1, 3]);
        assert_eq!(portfolio.top_endpoints[0].verifier_row, 2);
        assert_eq!(portfolio.top_endpoints[0].terminal_token, 91);
        assert_eq!(portfolio.top_endpoints[1].verifier_row, 3);
        assert_eq!(portfolio.top_endpoints[1].terminal_token, 90);
        assert_eq!(portfolio.early_rows, early.rows);

        let authority = Eagle3ForestAcceptance {
            emitted_tokens: vec![11, 13, 90],
            leaf_row: 3,
            capture_rows: vec![0, 1, 3],
            source_nodes: Vec::new(),
        };
        let evaluated = portfolio.evaluate(&authority).unwrap();
        assert_eq!(evaluated.transaction_rank, Some(1));
        assert_eq!(evaluated.path_rank, Some(1));
        assert_eq!(evaluated.endpoint_rank, Some(2));
        assert_eq!(evaluated.terminal_rank_at_authoritative_leaf, Some(1));
        assert!(evaluated.coverage[0].transaction_covered);
        assert!(evaluated.coverage[0].path_covered);
        assert!(!evaluated.coverage[0].endpoint_covered);
        assert!(evaluated.coverage[0].terminal_at_authoritative_leaf_covered);
        assert!(evaluated.coverage[1].endpoint_covered);
    }

    #[test]
    fn transaction_portfolio_coverage_decomposes_path_and_terminal_misses() {
        let (tree, early) = transaction_portfolio_fixture();
        let portfolio = Eagle3EarlyTransactionPortfolio::freeze(&tree, &early).unwrap();

        // The early head ranks this terminal cell first globally but assigns zero candidate-
        // restricted probability to the edge leading there. Endpoint coverage therefore does
        // not imply either path or complete-transaction coverage.
        let endpoint_only = Eagle3ForestAcceptance {
            emitted_tokens: vec![12, 91],
            leaf_row: 2,
            capture_rows: vec![0, 2],
            source_nodes: Vec::new(),
        };
        let endpoint_only = portfolio.evaluate(&endpoint_only).unwrap();
        assert_eq!(endpoint_only.transaction_rank, None);
        assert_eq!(endpoint_only.path_rank, None);
        assert_eq!(endpoint_only.endpoint_rank, Some(1));
        assert_eq!(endpoint_only.terminal_rank_at_authoritative_leaf, Some(1));
        assert!(!endpoint_only.coverage[3].transaction_covered);
        assert!(!endpoint_only.coverage[3].path_covered);
        assert!(endpoint_only.coverage[0].endpoint_covered);
        assert!(endpoint_only.coverage[0].terminal_at_authoritative_leaf_covered);

        // A token absent from the frozen candidate union can retain path coverage, but it can
        // never be credited to the joint, endpoint, or truth-conditional terminal metrics.
        let path_only = Eagle3ForestAcceptance {
            emitted_tokens: vec![11, 13, 99],
            leaf_row: 3,
            capture_rows: vec![0, 1, 3],
            source_nodes: Vec::new(),
        };
        let path_only = portfolio.evaluate(&path_only).unwrap();
        assert_eq!(path_only.transaction_rank, None);
        assert_eq!(path_only.path_rank, Some(1));
        assert_eq!(path_only.endpoint_rank, None);
        assert_eq!(path_only.terminal_rank_at_authoritative_leaf, None);
        assert!(!path_only.coverage[3].transaction_covered);
        assert!(path_only.coverage[0].path_covered);
        assert!(!path_only.coverage[3].endpoint_covered);
        assert!(!path_only.coverage[3].terminal_at_authoritative_leaf_covered);
    }

    #[test]
    fn transaction_portfolio_reports_exact_ranks_beyond_retained_top_eight() {
        let tree = TokenTree {
            tokens: vec![1],
            parent: vec![-1],
            depth: vec![0],
        };
        let candidate_union = (10..20).collect::<Vec<u32>>();
        let candidate_logit_bits = (0..10)
            .rev()
            .map(|score| (score as f32).to_bits())
            .collect::<Vec<_>>();
        let early = ResidentIndexedHeadEarlySnapshot {
            layer_id: 25,
            compile_fast_math_enabled: false,
            candidate_union: candidate_union.clone(),
            encode_us: 0,
            commit_wait_us: 0,
            gpu_busy_us: 0,
            kernel_window_us: 0,
            fallback_reason: None,
            rows: vec![ResidentIndexedHeadEarlyRow {
                verifier_row: 0,
                ranked_candidate_tokens: eagle3_early_ranked_tokens(
                    &candidate_union,
                    &candidate_logit_bits,
                ),
                candidate_logit_bits,
            }],
        };
        let portfolio = Eagle3EarlyTransactionPortfolio::freeze(&tree, &early).unwrap();
        assert_eq!(portfolio.transaction_candidates_considered, 10);
        assert_eq!(portfolio.top_transactions.len(), 8);
        assert_eq!(portfolio.top_endpoints.len(), 8);

        let evaluated = portfolio
            .evaluate(&Eagle3ForestAcceptance {
                emitted_tokens: vec![19],
                leaf_row: 0,
                capture_rows: vec![0],
                source_nodes: Vec::new(),
            })
            .unwrap();
        assert_eq!(evaluated.transaction_rank, Some(10));
        assert_eq!(evaluated.path_rank, Some(1));
        assert_eq!(evaluated.endpoint_rank, Some(10));
        assert_eq!(evaluated.terminal_rank_at_authoritative_leaf, Some(10));
        assert!(!evaluated.coverage[3].transaction_covered);
        assert!(evaluated.coverage[0].path_covered);
        assert!(!evaluated.coverage[3].endpoint_covered);
        assert!(!evaluated.coverage[3].terminal_at_authoritative_leaf_covered);
    }

    #[test]
    fn transaction_portfolio_rejects_unranked_or_wrong_layer_evidence() {
        let (tree, early) = transaction_portfolio_fixture();
        let mut wrong_rank = early.clone();
        wrong_rank.rows[0].ranked_candidate_tokens.swap(0, 1);
        assert!(Eagle3EarlyTransactionPortfolio::freeze(&tree, &wrong_rank).is_err());

        let mut wrong_layer = early;
        wrong_layer.layer_id = 24;
        assert!(Eagle3EarlyTransactionPortfolio::freeze(&tree, &wrong_layer).is_err());

        let (_, valid_early) = transaction_portfolio_fixture();
        let portfolio = Eagle3EarlyTransactionPortfolio::freeze(&tree, &valid_early).unwrap();
        assert!(portfolio
            .evaluate(&Eagle3ForestAcceptance {
                emitted_tokens: vec![12, 13, 90],
                leaf_row: 3,
                capture_rows: vec![0, 1, 3],
                source_nodes: Vec::new(),
            })
            .is_err());
    }

    fn selective_edge_portfolio_fixture() -> Eagle3EarlyTransactionPortfolio {
        let paths = [
            vec![0, 1, 4, 6],
            vec![0, 1, 3],
            vec![0, 2, 5],
            vec![0, 1, 4, 7],
        ];
        Eagle3EarlyTransactionPortfolio {
            layer_id: 25,
            tree_tokens: vec![10, 11, 12, 13, 14, 15, 16, 17],
            tree_parent: vec![-1, 0, 0, 1, 1, 2, 4, 4],
            tree_depth: vec![0, 1, 1, 2, 2, 2, 3, 3],
            candidate_union: vec![11, 12, 13, 14, 15, 16, 17, 90],
            early_rows: Vec::new(),
            transaction_candidates_considered: 0,
            path_candidates_considered: paths.len(),
            endpoint_candidates_considered: 0,
            top_transactions: Vec::new(),
            top_paths: paths
                .into_iter()
                .enumerate()
                .map(|(rank, path_rows)| Eagle3EarlyPathCandidate {
                    path_rows,
                    log_probability_bits: (-(rank as f64)).to_bits(),
                })
                .collect(),
            top_endpoints: Vec::new(),
            indexed_head_encode_us: 0,
            indexed_head_commit_wait_us: 0,
            indexed_head_gpu_busy_us: 0,
            indexed_head_kernel_window_us: 0,
            ranked_transactions: Vec::new(),
            ranked_paths: Vec::new(),
            ranked_endpoints: Vec::new(),
        }
    }

    #[test]
    fn selective_edge_prep_unions_complete_top_paths_without_root_or_duplicates() {
        let portfolio = selective_edge_portfolio_fixture();
        let b1 = portfolio.plan_selective_edge_prep(1).unwrap();
        let b2 = portfolio.plan_selective_edge_prep(2).unwrap();
        let b4 = portfolio.plan_selective_edge_prep(4).unwrap();
        assert_eq!(b1.prepared_edge_rows, vec![1, 4, 6]);
        assert_eq!(b2.prepared_edge_rows, vec![1, 3, 4, 6]);
        assert_eq!(b4.prepared_edge_rows, vec![1, 2, 3, 4, 5, 6, 7]);
        assert!(b1.prepared_edge_rows.iter().all(|row| *row != 0));
        assert!(portfolio.plan_selective_edge_prep(3).is_err());

        let mut malformed = portfolio.clone();
        malformed.tree_parent[4] = 7;
        assert!(malformed.plan_selective_edge_prep(1).is_err());

        let mut out_of_range_path = portfolio.clone();
        out_of_range_path.top_paths[0].path_rows.push(99);
        assert!(out_of_range_path.plan_selective_edge_prep(1).is_err());
    }

    #[test]
    fn selective_edge_prep_reports_hits_misses_and_never_claims_shadow_reuse() {
        let portfolio = selective_edge_portfolio_fixture();
        let acceptance = Eagle3ForestAcceptance {
            emitted_tokens: vec![11, 14, 17, 90],
            leaf_row: 7,
            capture_rows: vec![0, 1, 4, 7],
            source_nodes: Vec::new(),
        };
        let b2 = portfolio
            .plan_selective_edge_prep(2)
            .unwrap()
            .evaluate(&acceptance)
            .unwrap();
        assert_eq!(b2.predicted_unique_edge_rows, 4);
        assert_eq!(b2.authoritative_edge_rows, 3);
        assert_eq!(b2.prepared_hits, 2);
        assert_eq!(b2.prepared_misses, 1);
        assert!(!b2.authoritative_path_fully_covered);
        assert_eq!(b2.theoretical_serial_edge_rows_displaced, 2);
        assert_eq!(b2.actually_reused_edge_rows, 0);

        let b4 = portfolio
            .plan_selective_edge_prep(4)
            .unwrap()
            .evaluate(&acceptance)
            .unwrap();
        assert_eq!(b4.prepared_hits, 3);
        assert_eq!(b4.prepared_misses, 0);
        assert!(b4.authoritative_path_fully_covered);
        assert_eq!(b4.actually_reused_edge_rows, 0);

        let mut ancestry_hole = portfolio.plan_selective_edge_prep(1).unwrap();
        ancestry_hole.prepared_edge_rows.remove(0);
        assert!(ancestry_hole.evaluate(&acceptance).is_err());
    }

    #[test]
    fn eagle3_device_acceptance_matches_target_oracle() {
        let mut frontier = Eagle3DynamicFrontier::new(10, frontier_config(8, 16, 4)).unwrap();
        frontier
            .record_expansion(0, &output(&[(11, 0.55), (12, 0.45)], 1.0))
            .unwrap();
        frontier
            .record_expansion(1, &output(&[(13, 0.60), (14, 0.40)], 2.0))
            .unwrap();
        frontier
            .record_expansion(2, &output(&[(15, 0.70)], 3.0))
            .unwrap();
        // The scheduler ranks source node 3 at .55 * .60 = .33, just ahead of source node 5
        // at .45 * .70 = .315.  Tests must follow that causal expansion order; verifier-row
        // numbering is frozen only by `finish` below.
        frontier
            .record_expansion(3, &output(&[(16, 0.65), (17, 0.35)], 4.0))
            .unwrap();
        let forest = frontier.finish().unwrap();
        let plan = forest.plan_device_acceptance().unwrap();
        let nodes = forest.scored.tree.nodes();
        assert_eq!(nodes, 8);
        assert_eq!(
            forest.scored.tree.tokens,
            vec![10, 11, 12, 13, 14, 15, 16, 17]
        );
        assert_eq!(forest.scored.tree.parent, vec![-1, 0, 0, 1, 1, 2, 3, 3]);
        assert_eq!(forest.scored.tree.depth, vec![0, 1, 1, 2, 2, 2, 3, 3]);

        let mut cases = vec![
            vec![99, 0, 0, 0, 0, 0, 0, 0],
            vec![11, 13, 0, 97, 0, 0, 0, 0],
            vec![11, 13, 0, 16, 0, 0, 96, 0],
            vec![11, 13, 0, 17, 0, 0, 0, 95],
            vec![11, 14, 0, 0, 93, 0, 0, 0],
            vec![12, 0, 15, 0, 0, 94, 0, 0],
            vec![u32::MAX, 0, 0, 0, 0, 0, 0, 0],
        ];
        let alphabet = [11, 12, 13, 14, 15, 16, 17, 98];
        for seed in 0..256u32 {
            let mut state = seed.wrapping_add(1);
            let mut predictions = Vec::with_capacity(nodes);
            for row in 0..nodes {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                predictions.push(alphabet[((state >> 24) as usize + row) % alphabet.len()]);
            }
            cases.push(predictions);
        }

        for predictions in cases {
            let selected = plan.select_reference(&predictions).unwrap();
            let (emitted, leaf) = forest.scored.tree.accept_longest_path(&predictions);
            let expected_path = forest
                .scored
                .tree
                .path_to(leaf)
                .into_iter()
                .map(|row| row as u32)
                .collect::<Vec<_>>();
            assert_eq!(selected.leaf_row as usize, leaf);
            assert_eq!(selected.selected_path(), expected_path.as_slice());
            assert_eq!(selected.emitted(), emitted.as_slice());
            assert_eq!(selected.terminal_token, *emitted.last().unwrap());
            assert_eq!(
                selected.terminal_depth,
                u32::from(forest.scored.tree.depth[leaf])
            );
            assert_eq!(
                selected.terminal_token_valid,
                selected.terminal_token != u32::MAX
            );
            assert_eq!(
                selected.safe_terminal_token,
                if selected.terminal_token_valid {
                    selected.terminal_token
                } else {
                    0
                }
            );
        }
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
