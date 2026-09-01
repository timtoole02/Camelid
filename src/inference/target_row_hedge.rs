//! Benchmark-only causal fusion of one prior target-row hedge into an EAGLE forest.
//!
//! The ordinary EAGLE dynamic reranker first builds its verifier-ready forest and records the
//! exact learned primary spine.  This module may exchange one *non-primary leaf* for one novel
//! top-1 candidate copied from a target row verified in an earlier round.  It never evicts the
//! EAGLE primary spine, always retains at least one learned neural hedge, deduplicates complete
//! root-to-node token paths, and never grows the fixed N=8 verifier batch.
//!
//! Proposal provenance has no authority over output.  The fused [`TokenTree`] is sent through the
//! same full target verification and [`TokenTree::accept_longest_path`] rule as the ordinary EAGLE
//! tree. The original hedge permits a token outside EAGLE's private draft lattice; the stricter
//! lattice-promotion entry point admits only an immediate child materialized under the exact
//! current primary source.

use std::collections::{HashMap, HashSet};

use super::spec_tree::{DynamicDraftLattice, ScoredTokenTree, TokenTree};

/// Mini2's fixed verifier-width speed lane, including the root anchor.
pub const TARGET_ROW_HEDGE_MAX_NODES: usize = 8;
/// One earlier exact target top-1 may at most double a current EAGLE candidate's score.
pub const TARGET_ROW_LATTICE_PRIOR_LOG_BONUS: f32 = std::f32::consts::LN_2;
/// Hard cap for benchmark-only score-gate sweeps; enforced again by the public core API.
pub const TARGET_ROW_LATTICE_MAX_PRIOR_LOG_BONUS: f32 = 16.0;

/// Provenance for one fused verifier row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetRowHedgeSource {
    Anchor,
    EaglePrimary,
    EagleNeuralHedge,
    PriorTargetTop1,
    EaglePrimaryAndPriorTargetTop1,
    EagleNeuralHedgeAndPriorTargetTop1,
}

impl TargetRowHedgeSource {
    fn with_prior_target(self) -> Self {
        match self {
            Self::Anchor => Self::Anchor,
            Self::EaglePrimary | Self::EaglePrimaryAndPriorTargetTop1 => {
                Self::EaglePrimaryAndPriorTargetTop1
            }
            Self::EagleNeuralHedge | Self::EagleNeuralHedgeAndPriorTargetTop1 => {
                Self::EagleNeuralHedgeAndPriorTargetTop1
            }
            Self::PriorTargetTop1 => Self::PriorTargetTop1,
        }
    }

    pub fn is_neural_hedge(self) -> bool {
        matches!(
            self,
            Self::EagleNeuralHedge | Self::EagleNeuralHedgeAndPriorTargetTop1
        )
    }
}

/// Why a causal fusion round did or did not add a target-row verifier slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetRowHedgeOutcome {
    Inserted,
    NoPriorTargetRow,
    NoPromotableLatticeChild,
    DuplicateOnly,
    BelowPromotionScore,
    NoReplaceableNeuralHedge,
}

/// Stable, model-free evidence for one fusion decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetRowHedgeDecision {
    pub outcome: TargetRowHedgeOutcome,
    pub supported_primary_rows: usize,
    pub full_path_duplicates: usize,
    pub candidate_token: Option<u32>,
    pub candidate_parent_depth: Option<usize>,
    pub candidate_depth: Option<usize>,
    /// Stable source id when the candidate was confirmed in the current materialized lattice.
    pub candidate_lattice_source_node: Option<usize>,
    pub candidate_cumulative_log_probability: Option<f32>,
    pub candidate_adjusted_log_score: Option<f32>,
    /// Cached primary rows whose prior target top-1 was absent under the exact current source.
    pub prior_rows_without_materialized_child: usize,
    /// Weakest ordinary leaf compared by the conservative lattice score gate. This row is only
    /// actually removed when `outcome == Inserted`.
    pub comparison_eagle_row: Option<usize>,
    pub comparison_cumulative_log_probability: Option<f32>,
    pub evicted_eagle_row: Option<usize>,
    pub neural_hedges_retained: usize,
}

/// One verifier-ready forest plus enough provenance to audit every row.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetRowHedgeForest {
    pub tree: TokenTree,
    pub source: Vec<TargetRowHedgeSource>,
    /// Original EAGLE verifier row for retained learned rows. The inserted candidate is `None`
    /// because it was not in ordinary N8, even when the lattice-promotion lane records its stable
    /// current-lattice source separately in the decision.
    pub eagle_source_row: Vec<Option<usize>>,
    /// Exact remapped learned primary spine, root first.
    pub primary_spine_rows: Vec<usize>,
    pub decision: TargetRowHedgeDecision,
}

/// Target-authoritative acceptance through the fused tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetRowHedgeAcceptance {
    pub emitted_tokens: Vec<u32>,
    pub leaf_row: usize,
    pub capture_rows: Vec<usize>,
}

/// Exact emitted-length effect of exchanging the two leaf edges in a promoted round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetRowHedgeCounterfactual {
    pub inserted_row_accepted: bool,
    pub evicted_edge_would_accept: bool,
    pub emitted_token_delta: i8,
}

#[derive(Debug, Clone, Copy)]
struct SelectedTargetRowCandidate {
    parent_row: usize,
    token: u32,
    lattice_source_node: Option<usize>,
    cumulative_log_probability: Option<f32>,
    adjusted_log_score: Option<f32>,
}

impl TargetRowHedgeForest {
    /// Fuse the shallowest reachable, novel prior-target top-1 candidate into an EAGLE forest.
    ///
    /// `prior_target_top1` must be a read-only snapshot of rows installed before this round.  The
    /// caller performs the current target verification and only then mutates its row cache.  A
    /// shallow candidate is preferred because every deeper hedge is conditional on surviving a
    /// longer EAGLE prefix.  Complete path identity, rather than token-id identity, controls
    /// deduplication, so the same token remains legal under different parents.
    pub fn fuse<F>(
        eagle: &ScoredTokenTree,
        max_nodes: usize,
        max_depth: usize,
        mut prior_target_top1: F,
    ) -> Result<Self, String>
    where
        F: FnMut(u32) -> Option<u32>,
    {
        validate_eagle(eagle, max_nodes, max_depth)?;
        let tree = &eagle.tree;
        let (mut source, paths, path_to_row) = initial_sources_and_paths(eagle)?;

        let mut supported_primary_rows = 0usize;
        let mut full_path_duplicates = 0usize;
        let mut selected_candidate = None;
        for &parent_row in &eagle.primary_spine_rows {
            let parent_depth = usize::from(tree.depth[parent_row]);
            if parent_depth >= max_depth {
                continue;
            }
            let Some(token) = prior_target_top1(tree.tokens[parent_row]) else {
                continue;
            };
            supported_primary_rows += 1;
            let mut path = paths[parent_row].clone();
            path.push(token);
            if let Some(&duplicate_row) = path_to_row.get(&path) {
                full_path_duplicates += 1;
                source[duplicate_row] = source[duplicate_row].with_prior_target();
                continue;
            }
            selected_candidate = Some(SelectedTargetRowCandidate {
                parent_row,
                token,
                lattice_source_node: None,
                cumulative_log_probability: None,
                adjusted_log_score: None,
            });
            break;
        }
        let empty_outcome = if supported_primary_rows == 0 {
            TargetRowHedgeOutcome::NoPriorTargetRow
        } else {
            TargetRowHedgeOutcome::DuplicateOnly
        };
        Self::splice_selected_candidate(
            eagle,
            max_nodes,
            source,
            selected_candidate,
            empty_outcome,
            supported_primary_rows,
            full_path_duplicates,
            0,
            false,
        )
    }

    /// Promote only an earlier target top-1 that is also an immediate child of the exact current
    /// primary lattice source and was pruned from ordinary N8.
    ///
    /// All matching pruned candidates are considered. Current cumulative EAGLE score ranks them;
    /// one prior exact target confirmation contributes a bounded `ln(2)` admission bonus. At full
    /// width the adjusted candidate must meet the score of the ordinary leaf it would displace.
    pub fn fuse_lattice_promoted<F>(
        eagle: &ScoredTokenTree,
        lattice: &DynamicDraftLattice,
        max_nodes: usize,
        max_depth: usize,
        prior_target_top1: F,
    ) -> Result<Self, String>
    where
        F: FnMut(u32) -> Option<u32>,
    {
        Self::fuse_lattice_promoted_with_bonus(
            eagle,
            lattice,
            max_nodes,
            max_depth,
            TARGET_ROW_LATTICE_PRIOR_LOG_BONUS,
            prior_target_top1,
        )
    }

    /// Experimental score-gate sweep for exact lattice promotion. The default entry point above
    /// remains byte-for-byte equivalent to `ln(2)`; callers must supply a finite bonus within the
    /// hard sweep cap, and target verification remains authoritative for every inserted row.
    pub fn fuse_lattice_promoted_with_bonus<F>(
        eagle: &ScoredTokenTree,
        lattice: &DynamicDraftLattice,
        max_nodes: usize,
        max_depth: usize,
        prior_log_bonus: f32,
        mut prior_target_top1: F,
    ) -> Result<Self, String>
    where
        F: FnMut(u32) -> Option<u32>,
    {
        if !prior_log_bonus.is_finite()
            || !(0.0..=TARGET_ROW_LATTICE_MAX_PRIOR_LOG_BONUS).contains(&prior_log_bonus)
        {
            return Err(format!(
                "target-row lattice prior log bonus must be finite and in 0..={}, got {prior_log_bonus}",
                TARGET_ROW_LATTICE_MAX_PRIOR_LOG_BONUS,
            ));
        }
        validate_eagle(eagle, max_nodes, max_depth)?;
        validate_lattice_alignment(eagle, lattice)?;
        let tree = &eagle.tree;
        let (mut source, paths, path_to_row) = initial_sources_and_paths(eagle)?;
        let selected_sources = eagle.source_node.iter().copied().collect::<HashSet<_>>();
        let mut supported_primary_rows = 0usize;
        let mut full_path_duplicates = 0usize;
        let mut prior_rows_without_materialized_child = 0usize;
        let mut candidates = Vec::new();

        for &parent_row in &eagle.primary_spine_rows {
            if usize::from(tree.depth[parent_row]) >= max_depth {
                continue;
            }
            let Some(token) = prior_target_top1(tree.tokens[parent_row]) else {
                continue;
            };
            supported_primary_rows += 1;
            let parent_source = eagle.source_node[parent_row];
            let materialized = lattice
                .nodes()
                .iter()
                .enumerate()
                .find(|(_, node)| node.parent == Some(parent_source) && node.token == token);
            let Some((lattice_source_node, lattice_node)) = materialized else {
                prior_rows_without_materialized_child += 1;
                continue;
            };

            let mut path = paths[parent_row].clone();
            path.push(token);
            if let Some(&duplicate_row) = path_to_row.get(&path) {
                full_path_duplicates += 1;
                source[duplicate_row] = source[duplicate_row].with_prior_target();
                continue;
            }
            if selected_sources.contains(&lattice_source_node) {
                return Err(format!(
                    "current EAGLE lattice source {lattice_source_node} is selected but its full path is absent from N8"
                ));
            }
            let adjusted_log_score = lattice_node.cumulative_log_probability + prior_log_bonus;
            if !adjusted_log_score.is_finite() {
                return Err(format!(
                    "target-row lattice adjusted score is not finite for source {lattice_source_node}"
                ));
            }
            candidates.push(SelectedTargetRowCandidate {
                parent_row,
                token,
                lattice_source_node: Some(lattice_source_node),
                cumulative_log_probability: Some(lattice_node.cumulative_log_probability),
                adjusted_log_score: Some(adjusted_log_score),
            });
        }

        candidates.sort_by(|left, right| {
            right
                .adjusted_log_score
                .expect("lattice candidate has adjusted score")
                .total_cmp(
                    &left
                        .adjusted_log_score
                        .expect("lattice candidate has adjusted score"),
                )
                .then_with(|| {
                    right
                        .cumulative_log_probability
                        .expect("lattice candidate has cumulative score")
                        .total_cmp(
                            &left
                                .cumulative_log_probability
                                .expect("lattice candidate has cumulative score"),
                        )
                })
                .then_with(|| tree.depth[left.parent_row].cmp(&tree.depth[right.parent_row]))
                .then_with(|| {
                    left.lattice_source_node
                        .expect("lattice candidate has source")
                        .cmp(
                            &right
                                .lattice_source_node
                                .expect("lattice candidate has source"),
                        )
                })
        });
        let selected_candidate = candidates.first().copied();
        let empty_outcome = if supported_primary_rows == 0 {
            TargetRowHedgeOutcome::NoPriorTargetRow
        } else if full_path_duplicates > 0
            && full_path_duplicates + prior_rows_without_materialized_child
                == supported_primary_rows
            && prior_rows_without_materialized_child == 0
        {
            TargetRowHedgeOutcome::DuplicateOnly
        } else {
            TargetRowHedgeOutcome::NoPromotableLatticeChild
        };
        Self::splice_selected_candidate(
            eagle,
            max_nodes,
            source,
            selected_candidate,
            empty_outcome,
            supported_primary_rows,
            full_path_duplicates,
            prior_rows_without_materialized_child,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn splice_selected_candidate(
        eagle: &ScoredTokenTree,
        max_nodes: usize,
        source: Vec<TargetRowHedgeSource>,
        selected_candidate: Option<SelectedTargetRowCandidate>,
        empty_outcome: TargetRowHedgeOutcome,
        supported_primary_rows: usize,
        full_path_duplicates: usize,
        prior_rows_without_materialized_child: usize,
        enforce_lattice_score_gate: bool,
    ) -> Result<Self, String> {
        let tree = &eagle.tree;
        let primary = eagle
            .primary_spine_rows
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let Some(candidate) = selected_candidate else {
            let neural_hedges_retained = source
                .iter()
                .filter(|&&item| item.is_neural_hedge())
                .count();
            return Ok(Self::unchanged(
                eagle,
                source,
                TargetRowHedgeDecision {
                    outcome: empty_outcome,
                    supported_primary_rows,
                    full_path_duplicates,
                    candidate_token: None,
                    candidate_parent_depth: None,
                    candidate_depth: None,
                    candidate_lattice_source_node: None,
                    candidate_cumulative_log_probability: None,
                    candidate_adjusted_log_score: None,
                    prior_rows_without_materialized_child,
                    comparison_eagle_row: None,
                    comparison_cumulative_log_probability: None,
                    evicted_eagle_row: None,
                    neural_hedges_retained,
                },
            ));
        };

        let candidate_decision = |outcome,
                                  comparison_eagle_row,
                                  comparison_score,
                                  evicted_eagle_row,
                                  neural_hedges_retained| {
            TargetRowHedgeDecision {
                outcome,
                supported_primary_rows,
                full_path_duplicates,
                candidate_token: Some(candidate.token),
                candidate_parent_depth: Some(usize::from(tree.depth[candidate.parent_row])),
                candidate_depth: Some(usize::from(tree.depth[candidate.parent_row]) + 1),
                candidate_lattice_source_node: candidate.lattice_source_node,
                candidate_cumulative_log_probability: candidate.cumulative_log_probability,
                candidate_adjusted_log_score: candidate.adjusted_log_score,
                prior_rows_without_materialized_child,
                comparison_eagle_row,
                comparison_cumulative_log_probability: comparison_score,
                evicted_eagle_row,
                neural_hedges_retained,
            }
        };

        let neural_hedge_count = source
            .iter()
            .filter(|&&item| item.is_neural_hedge())
            .count();
        if neural_hedge_count == 0 {
            return Ok(Self::unchanged(
                eagle,
                source,
                candidate_decision(
                    TargetRowHedgeOutcome::NoReplaceableNeuralHedge,
                    None,
                    None,
                    None,
                    0,
                ),
            ));
        }

        let comparison_eagle_row = if tree.nodes() < max_nodes {
            None
        } else {
            if neural_hedge_count < 2 {
                return Ok(Self::unchanged(
                    eagle,
                    source,
                    candidate_decision(
                        TargetRowHedgeOutcome::NoReplaceableNeuralHedge,
                        None,
                        None,
                        None,
                        neural_hedge_count,
                    ),
                ));
            }
            let has_child =
                |candidate: usize| tree.parent.iter().any(|&parent| parent == candidate as i32);
            let weakest_leaf = (1..tree.nodes())
                .filter(|row| {
                    !primary.contains(row)
                        && source[*row] == TargetRowHedgeSource::EagleNeuralHedge
                        && !has_child(*row)
                })
                .min_by(|&left, &right| {
                    eagle.cumulative_log_probability[left]
                        .total_cmp(&eagle.cumulative_log_probability[right])
                        .then_with(|| right.cmp(&left))
                });
            let Some(row) = weakest_leaf else {
                return Ok(Self::unchanged(
                    eagle,
                    source,
                    candidate_decision(
                        TargetRowHedgeOutcome::NoReplaceableNeuralHedge,
                        None,
                        None,
                        None,
                        neural_hedge_count,
                    ),
                ));
            };
            Some(row)
        };
        let comparison_score =
            comparison_eagle_row.map(|row| eagle.cumulative_log_probability[row]);
        if enforce_lattice_score_gate
            && comparison_score.is_some_and(|score| {
                candidate
                    .adjusted_log_score
                    .expect("score-gated candidate has adjusted score")
                    < score
            })
        {
            let retained = source
                .iter()
                .filter(|&&item| item.is_neural_hedge())
                .count();
            return Ok(Self::unchanged(
                eagle,
                source,
                candidate_decision(
                    TargetRowHedgeOutcome::BelowPromotionScore,
                    comparison_eagle_row,
                    comparison_score,
                    None,
                    retained,
                ),
            ));
        }

        let synthetic = tree.nodes();
        let mut retained = (0..tree.nodes())
            .filter(|row| Some(*row) != comparison_eagle_row)
            .collect::<Vec<_>>();
        retained.push(synthetic);
        retained.sort_by_key(|&row| {
            if row == synthetic {
                (tree.depth[candidate.parent_row].saturating_add(1), row)
            } else {
                (tree.depth[row], row)
            }
        });
        let mut remap = vec![usize::MAX; tree.nodes() + 1];
        for (new_row, &old_row) in retained.iter().enumerate() {
            remap[old_row] = new_row;
        }
        let mut tokens = Vec::with_capacity(retained.len());
        let mut parent = Vec::with_capacity(retained.len());
        let mut depth = Vec::with_capacity(retained.len());
        let mut output_source = Vec::with_capacity(retained.len());
        let mut eagle_source_row = Vec::with_capacity(retained.len());
        for &old_row in &retained {
            if old_row == synthetic {
                tokens.push(candidate.token);
                parent.push(
                    i32::try_from(remap[candidate.parent_row])
                        .map_err(|_| "target-row hedge verifier parent exceeds i32".to_string())?,
                );
                depth.push(
                    tree.depth[candidate.parent_row]
                        .checked_add(1)
                        .ok_or_else(|| "target-row hedge depth exceeds u16".to_string())?,
                );
                output_source.push(TargetRowHedgeSource::PriorTargetTop1);
                eagle_source_row.push(None);
            } else {
                tokens.push(tree.tokens[old_row]);
                parent.push(if old_row == 0 {
                    -1
                } else {
                    let old_parent = usize::try_from(tree.parent[old_row])
                        .map_err(|_| format!("EAGLE row {old_row} has a negative parent"))?;
                    i32::try_from(remap[old_parent])
                        .map_err(|_| "target-row hedge verifier row exceeds i32".to_string())?
                });
                depth.push(tree.depth[old_row]);
                output_source.push(source[old_row]);
                eagle_source_row.push(Some(old_row));
            }
        }
        let primary_spine_rows = eagle
            .primary_spine_rows
            .iter()
            .map(|&row| remap[row])
            .collect::<Vec<_>>();
        let fused = TokenTree {
            tokens,
            parent,
            depth,
        };
        validate_output(&fused, &output_source, &primary_spine_rows, max_nodes)?;
        let neural_hedges_retained = output_source
            .iter()
            .filter(|&&item| item.is_neural_hedge())
            .count();
        if neural_hedges_retained == 0 {
            return Err("target-row fusion removed every EAGLE neural hedge".to_string());
        }
        Ok(Self {
            tree: fused,
            source: output_source,
            eagle_source_row,
            primary_spine_rows,
            decision: candidate_decision(
                TargetRowHedgeOutcome::Inserted,
                comparison_eagle_row,
                comparison_score,
                comparison_eagle_row,
                neural_hedges_retained,
            ),
        })
    }

    fn unchanged(
        eagle: &ScoredTokenTree,
        source: Vec<TargetRowHedgeSource>,
        decision: TargetRowHedgeDecision,
    ) -> Self {
        Self {
            tree: eagle.tree.clone(),
            source,
            eagle_source_row: (0..eagle.tree.nodes()).map(Some).collect(),
            primary_spine_rows: eagle.primary_spine_rows.clone(),
            decision,
        }
    }

    /// Apply the unchanged target-greedy tree acceptance rule.
    pub fn accept_target_predictions(
        &self,
        predictions: &[u32],
    ) -> Result<TargetRowHedgeAcceptance, String> {
        if predictions.len() != self.tree.nodes() {
            return Err(format!(
                "target-row hedge forest has {} rows but target returned {} predictions",
                self.tree.nodes(),
                predictions.len()
            ));
        }
        let (emitted_tokens, leaf_row) = self.tree.accept_longest_path(predictions);
        let capture_rows = self.tree.path_to(leaf_row);
        if emitted_tokens.len() != capture_rows.len() {
            return Err(format!(
                "target-row hedge acceptance produced {} tokens from {} target rows",
                emitted_tokens.len(),
                capture_rows.len()
            ));
        }
        Ok(TargetRowHedgeAcceptance {
            emitted_tokens,
            leaf_row,
            capture_rows,
        })
    }

    /// Compute the exact emitted-length effect of the exchanged leaf edges without a shadow
    /// target row. The evicted node was a leaf, so its retained parent's already-verified greedy
    /// prediction is sufficient to determine whether ordinary N8 would have accepted it.
    pub fn exact_counterfactual(
        &self,
        eagle: &ScoredTokenTree,
        predictions: &[u32],
        acceptance: &TargetRowHedgeAcceptance,
    ) -> Result<TargetRowHedgeCounterfactual, String> {
        if predictions.len() != self.tree.nodes() {
            return Err("target-row hedge counterfactual predictions are misaligned".to_string());
        }
        let inserted_row = self
            .source
            .iter()
            .position(|&source| source == TargetRowHedgeSource::PriorTargetTop1);
        let inserted_row_accepted =
            inserted_row.is_some_and(|row| acceptance.capture_rows.contains(&row));
        let evicted_edge_would_accept = if let Some(evicted) = self.decision.evicted_eagle_row {
            if evicted >= eagle.tree.nodes() {
                return Err("target-row hedge evicted row is outside ordinary EAGLE".to_string());
            }
            let old_parent = usize::try_from(eagle.tree.parent[evicted])
                .map_err(|_| "target-row hedge cannot evict the root".to_string())?;
            let new_parent = self
                .eagle_source_row
                .iter()
                .position(|&source| source == Some(old_parent))
                .ok_or_else(|| "target-row hedge lost the evicted leaf parent".to_string())?;
            acceptance.capture_rows.contains(&new_parent)
                && predictions[new_parent] == eagle.tree.tokens[evicted]
        } else {
            false
        };
        if inserted_row_accepted && evicted_edge_would_accept {
            return Err(
                "inserted target row and evicted sibling cannot both be target-greedy".to_string(),
            );
        }
        Ok(TargetRowHedgeCounterfactual {
            inserted_row_accepted,
            evicted_edge_would_accept,
            emitted_token_delta: inserted_row_accepted as i8 - evicted_edge_would_accept as i8,
        })
    }
}

fn initial_sources_and_paths(
    eagle: &ScoredTokenTree,
) -> Result<
    (
        Vec<TargetRowHedgeSource>,
        Vec<Vec<u32>>,
        HashMap<Vec<u32>, usize>,
    ),
    String,
> {
    let tree = &eagle.tree;
    let primary = eagle
        .primary_spine_rows
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let source = (0..tree.nodes())
        .map(|row| {
            if row == 0 {
                TargetRowHedgeSource::Anchor
            } else if primary.contains(&row) {
                TargetRowHedgeSource::EaglePrimary
            } else {
                TargetRowHedgeSource::EagleNeuralHedge
            }
        })
        .collect::<Vec<_>>();
    let paths = tree_paths(tree)?;
    let mut path_to_row = HashMap::with_capacity(paths.len());
    for (row, path) in paths.iter().enumerate() {
        if path_to_row.insert(path.clone(), row).is_some() {
            return Err(format!(
                "EAGLE verifier tree repeats full token path at row {row}"
            ));
        }
    }
    Ok((source, paths, path_to_row))
}

fn validate_lattice_alignment(
    eagle: &ScoredTokenTree,
    lattice: &DynamicDraftLattice,
) -> Result<(), String> {
    let nodes = lattice.nodes();
    if nodes.is_empty()
        || nodes[0].parent.is_some()
        || nodes[0].depth != 0
        || !nodes[0].cumulative_log_probability.is_finite()
    {
        return Err("target-row promotion current EAGLE lattice root is malformed".to_string());
    }
    for (source, node) in nodes.iter().enumerate().skip(1) {
        let parent = node
            .parent
            .ok_or_else(|| format!("target-row promotion lattice source {source} has no parent"))?;
        if parent >= source
            || node.depth != nodes[parent].depth.saturating_add(1)
            || !node.cumulative_log_probability.is_finite()
        {
            return Err(format!(
                "target-row promotion lattice source {source} is not finite and parent-closed"
            ));
        }
    }

    let tree = &eagle.tree;
    let mut selected_sources = HashSet::new();
    for row in 0..tree.nodes() {
        let source = eagle.source_node[row];
        if source >= nodes.len() || !selected_sources.insert(source) {
            return Err(format!(
                "target-row promotion ordinary row {row} has invalid lattice source {source}"
            ));
        }
        let node = &nodes[source];
        if node.token != tree.tokens[row]
            || node.depth != tree.depth[row]
            || node.cumulative_log_probability.to_bits()
                != eagle.cumulative_log_probability[row].to_bits()
        {
            return Err(format!(
                "target-row promotion ordinary row {row} disagrees with lattice source {source}"
            ));
        }
        if row == 0 {
            if node.parent.is_some() {
                return Err(
                    "target-row promotion ordinary root has a lattice source parent".to_string(),
                );
            }
        } else {
            let parent = usize::try_from(tree.parent[row])
                .map_err(|_| format!("target-row promotion ordinary row {row} has no parent"))?;
            if node.parent != Some(eagle.source_node[parent]) {
                return Err(format!(
                    "target-row promotion ordinary row {row} disagrees with its lattice parent"
                ));
            }
        }
    }
    Ok(())
}

fn validate_eagle(
    eagle: &ScoredTokenTree,
    max_nodes: usize,
    max_depth: usize,
) -> Result<(), String> {
    let tree = &eagle.tree;
    if max_nodes == 0 || max_nodes > TARGET_ROW_HEDGE_MAX_NODES {
        return Err(format!(
            "target-row hedge node budget must be in 1..={TARGET_ROW_HEDGE_MAX_NODES}, got {max_nodes}"
        ));
    }
    if max_depth == 0 {
        return Err("target-row hedge needs a positive draft depth".to_string());
    }
    if tree.nodes() == 0 || tree.nodes() > max_nodes {
        return Err(format!(
            "target-row hedge EAGLE tree has {} rows for budget {max_nodes}",
            tree.nodes()
        ));
    }
    if tree.parent.len() != tree.nodes()
        || tree.depth.len() != tree.nodes()
        || eagle.cumulative_log_probability.len() != tree.nodes()
        || eagle.source_node.len() != tree.nodes()
    {
        return Err("target-row hedge EAGLE arrays have different lengths".to_string());
    }
    if tree.parent[0] != -1 || tree.depth[0] != 0 {
        return Err("target-row hedge EAGLE root is malformed".to_string());
    }
    for row in 1..tree.nodes() {
        let parent = usize::try_from(tree.parent[row])
            .map_err(|_| format!("target-row hedge EAGLE row {row} has a negative parent"))?;
        if parent >= row || tree.depth[row] != tree.depth[parent].saturating_add(1) {
            return Err(format!(
                "target-row hedge EAGLE row {row} is not causal and parent-closed"
            ));
        }
    }
    if eagle
        .cumulative_log_probability
        .iter()
        .any(|score| !score.is_finite())
    {
        return Err("target-row hedge EAGLE scores must be finite".to_string());
    }
    if eagle.primary_spine_rows.first() != Some(&0) {
        return Err("target-row hedge EAGLE primary spine must begin at root".to_string());
    }
    let mut seen = HashSet::new();
    for (slot, &row) in eagle.primary_spine_rows.iter().enumerate() {
        if row >= tree.nodes() || !seen.insert(row) {
            return Err(format!(
                "target-row hedge EAGLE primary spine row {row} is invalid"
            ));
        }
        if slot > 0 && tree.parent[row] != eagle.primary_spine_rows[slot - 1] as i32 {
            return Err("target-row hedge EAGLE primary spine is not a root path".to_string());
        }
    }
    Ok(())
}

fn validate_output(
    tree: &TokenTree,
    source: &[TargetRowHedgeSource],
    primary_spine_rows: &[usize],
    max_nodes: usize,
) -> Result<(), String> {
    if tree.nodes() > max_nodes
        || tree.parent.len() != tree.nodes()
        || tree.depth.len() != tree.nodes()
        || source.len() != tree.nodes()
    {
        return Err("target-row hedge output exceeds its cap or has misaligned arrays".to_string());
    }
    for row in 1..tree.nodes() {
        let parent = usize::try_from(tree.parent[row])
            .map_err(|_| format!("target-row hedge output row {row} has a negative parent"))?;
        if parent >= row || tree.depth[row] != tree.depth[parent].saturating_add(1) {
            return Err(format!(
                "target-row hedge output row {row} is not causal and parent-closed"
            ));
        }
    }
    if primary_spine_rows.first() != Some(&0)
        || primary_spine_rows
            .windows(2)
            .any(|rows| tree.parent[rows[1]] != rows[0] as i32)
    {
        return Err("target-row hedge output lost the EAGLE primary spine".to_string());
    }
    let paths = tree_paths(tree)?;
    let unique = paths.iter().collect::<HashSet<_>>();
    if unique.len() != paths.len() {
        return Err("target-row hedge output repeats a full token path".to_string());
    }
    Ok(())
}

fn tree_paths(tree: &TokenTree) -> Result<Vec<Vec<u32>>, String> {
    let mut paths = Vec::with_capacity(tree.nodes());
    for row in 0..tree.nodes() {
        let rows = tree.path_to(row);
        if rows.first() != Some(&0) || rows.last() != Some(&row) {
            return Err(format!("token tree row {row} has an invalid root path"));
        }
        paths.push(rows.into_iter().map(|index| tree.tokens[index]).collect());
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_tree::DraftCandidateScore;
    use crate::inference::token_recycling::TokenRecyclingDrafter;

    fn eagle_forest() -> ScoredTokenTree {
        // Primary: 10 -> 20 -> 21 -> 22 -> 23. Neural hedges: 10 -> 30,
        // 10 -> 40, and 10 -> 30 -> 31.
        ScoredTokenTree {
            tree: TokenTree {
                tokens: vec![10, 20, 30, 40, 21, 31, 22, 23],
                parent: vec![-1, 0, 0, 0, 1, 2, 4, 6],
                depth: vec![0, 1, 1, 1, 2, 2, 3, 4],
            },
            cumulative_log_probability: vec![0.0, -0.1, -0.3, -0.8, -0.2, -0.9, -0.3, -0.4],
            source_node: (0..8).collect(),
            primary_spine_rows: vec![0, 1, 4, 6, 7],
        }
    }

    fn path_tokens(tree: &TokenTree, row: usize) -> Vec<u32> {
        tree.path_to(row)
            .into_iter()
            .map(|index| tree.tokens[index])
            .collect()
    }

    fn promotion_lattice() -> (DynamicDraftLattice, ScoredTokenTree) {
        let mut lattice = DynamicDraftLattice::new(10);
        let root_children = lattice
            .expand(
                0,
                &[
                    DraftCandidateScore {
                        token: 20,
                        log_probability: -0.597_837,
                    },
                    DraftCandidateScore {
                        token: 30,
                        log_probability: -1.897_120,
                    },
                    DraftCandidateScore {
                        token: 40,
                        log_probability: -2.302_585,
                    },
                    DraftCandidateScore {
                        token: 99,
                        log_probability: -2.525_729,
                    },
                    DraftCandidateScore {
                        token: 77,
                        log_probability: -4.605_170,
                    },
                ],
            )
            .unwrap();
        let child_21 = lattice
            .expand(
                root_children[0],
                &[DraftCandidateScore {
                    token: 21,
                    log_probability: -0.223_144,
                }],
            )
            .unwrap()[0];
        let child_22 = lattice
            .expand(
                child_21,
                &[DraftCandidateScore {
                    token: 22,
                    log_probability: -0.223_144,
                }],
            )
            .unwrap()[0];
        let child_23 = lattice
            .expand(
                child_22,
                &[DraftCandidateScore {
                    token: 23,
                    log_probability: -0.223_144,
                }],
            )
            .unwrap()[0];
        lattice
            .expand(
                child_23,
                &[DraftCandidateScore {
                    token: 24,
                    log_probability: -0.223_144,
                }],
            )
            .unwrap();
        let eagle = lattice.rerank_connected(8, 15).unwrap();
        assert_eq!(eagle.primary_spine_rows.len(), 6);
        assert!(eagle.tree.tokens.contains(&30));
        assert!(eagle.tree.tokens.contains(&40));
        assert!(!eagle.tree.tokens.contains(&99));
        (lattice, eagle)
    }

    #[test]
    fn cache_updates_cannot_leak_into_the_same_round() {
        let eagle = eagle_forest();
        let mut cache = TokenRecyclingDrafter::default();
        let before =
            TargetRowHedgeForest::fuse(&eagle, 8, 15, |token| cache.prior_target_top1(token))
                .unwrap();
        assert_eq!(
            before.decision.outcome,
            TargetRowHedgeOutcome::NoPriorTargetRow
        );

        // This represents the caller's post-verification commit. Only a new fuse invocation can
        // observe it, so the current round can never draft from its own target result.
        cache.replace_target_candidates(10, &[99]);
        let next =
            TargetRowHedgeForest::fuse(&eagle, 8, 15, |token| cache.prior_target_top1(token))
                .unwrap();
        assert_eq!(next.decision.outcome, TargetRowHedgeOutcome::Inserted);
        assert_eq!(next.decision.candidate_token, Some(99));
    }

    #[test]
    fn primary_spine_is_never_evicted_and_a_neural_hedge_survives() {
        let eagle = eagle_forest();
        let fused =
            TargetRowHedgeForest::fuse(&eagle, 8, 15, |token| (token == 10).then_some(99)).unwrap();
        assert_eq!(fused.tree.nodes(), 8);
        for (slot, &old_row) in eagle.primary_spine_rows.iter().enumerate() {
            let new_row = fused.primary_spine_rows[slot];
            assert_eq!(
                path_tokens(&fused.tree, new_row),
                path_tokens(&eagle.tree, old_row)
            );
        }
        assert!(fused.decision.neural_hedges_retained >= 1);
        assert!(fused.source.iter().any(|source| source.is_neural_hedge()));
    }

    #[test]
    fn full_path_dedupe_parent_closure_and_n8_cap_hold() {
        let eagle = eagle_forest();
        let fused = TargetRowHedgeForest::fuse(&eagle, 8, 15, |token| match token {
            10 => Some(30), // Exact existing root-to-30 path: dedupe, do not stop searching.
            20 => Some(98), // Novel under the retained primary spine.
            _ => None,
        })
        .unwrap();
        assert_eq!(fused.decision.outcome, TargetRowHedgeOutcome::Inserted);
        assert_eq!(fused.decision.full_path_duplicates, 1);
        assert_eq!(fused.tree.nodes(), 8);
        assert!((1..fused.tree.nodes()).any(|row| path_tokens(&fused.tree, row) == [10, 20, 98]));
        assert_eq!(
            (1..fused.tree.nodes())
                .filter(|&row| path_tokens(&fused.tree, row) == [10, 30])
                .count(),
            1
        );
        assert!(fused
            .tree
            .parent
            .iter()
            .enumerate()
            .skip(1)
            .all(|(row, &parent)| parent >= 0 && parent < row as i32));
        assert!(fused.tree.depth.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn target_token_outside_eagle_lattice_needs_no_t2d_mapping() {
        let eagle = eagle_forest();
        let fused =
            TargetRowHedgeForest::fuse(&eagle, 8, 15, |token| (token == 10).then_some(999_999))
                .unwrap();
        let candidate_row = fused
            .source
            .iter()
            .position(|&source| source == TargetRowHedgeSource::PriorTargetTop1)
            .unwrap();
        assert_eq!(fused.eagle_source_row[candidate_row], None);
        assert!(!eagle.tree.tokens.contains(&999_999));

        let mut predictions = vec![777; fused.tree.nodes()];
        predictions[0] = 999_999;
        predictions[candidate_row] = 123;
        let accepted = fused.accept_target_predictions(&predictions).unwrap();
        assert_eq!(accepted.emitted_tokens, [999_999, 123]);
        assert_eq!(accepted.capture_rows, [0, candidate_row]);
    }

    #[test]
    fn mocked_target_oracle_remains_greedy_exact() {
        let eagle = eagle_forest();
        let fused = TargetRowHedgeForest::fuse(&eagle, 8, 15, |token| match token {
            10 => Some(20), // Dedupe the first primary edge.
            20 => Some(88), // Add a target-row alternative at depth two.
            _ => None,
        })
        .unwrap();
        let target_row = (1..fused.tree.nodes())
            .find(|&row| path_tokens(&fused.tree, row) == [10, 20, 88])
            .unwrap();
        let primary_20 = fused.tree.parent[target_row] as usize;
        let mut oracle = vec![u32::MAX; fused.tree.nodes()];
        oracle[0] = 20;
        oracle[primary_20] = 88;
        oracle[target_row] = 77;
        let accepted = fused.accept_target_predictions(&oracle).unwrap();
        assert_eq!(accepted.emitted_tokens, [20, 88, 77]);
        for (emitted, &row) in accepted.emitted_tokens.iter().zip(&accepted.capture_rows) {
            assert_eq!(*emitted, oracle[row]);
        }
    }

    #[test]
    fn lattice_promotion_is_parent_exact_n8_and_primary_safe() {
        let (lattice, eagle) = promotion_lattice();
        let fused = TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
            (token == 10).then_some(99)
        })
        .unwrap();
        assert_eq!(fused.decision.outcome, TargetRowHedgeOutcome::Inserted);
        assert_eq!(fused.decision.candidate_lattice_source_node, Some(4));
        assert_eq!(fused.tree.nodes(), 8);
        assert!(fused.tree.tokens.contains(&99));
        assert!(!fused.tree.tokens.contains(&40));
        for (slot, &old_row) in eagle.primary_spine_rows.iter().enumerate() {
            assert_eq!(
                path_tokens(&fused.tree, fused.primary_spine_rows[slot]),
                path_tokens(&eagle.tree, old_row)
            );
        }
        assert_eq!(fused.decision.neural_hedges_retained, 1);
        assert_eq!(
            fused
                .source
                .iter()
                .filter(|&&source| source.is_neural_hedge())
                .count(),
            1
        );
    }

    #[test]
    fn lattice_promotion_cache_is_visible_only_to_the_next_selection() {
        let (lattice, eagle) = promotion_lattice();
        let mut cache = TokenRecyclingDrafter::default();
        let before =
            TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
                cache.prior_target_top1(token)
            })
            .unwrap();
        assert_eq!(
            before.decision.outcome,
            TargetRowHedgeOutcome::NoPriorTargetRow
        );

        // Represents the caller's publication barrier after the completed target round.
        cache.replace_target_candidates(10, &[99]);
        let next = TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
            cache.prior_target_top1(token)
        })
        .unwrap();
        assert_eq!(next.decision.outcome, TargetRowHedgeOutcome::Inserted);
    }

    #[test]
    fn lattice_promotion_fails_closed_outside_exact_parent_source() {
        let (lattice, eagle) = promotion_lattice();
        let absent =
            TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
                (token == 10).then_some(999_999)
            })
            .unwrap();
        assert_eq!(
            absent.decision.outcome,
            TargetRowHedgeOutcome::NoPromotableLatticeChild
        );
        assert_eq!(absent.tree, eagle.tree);

        // Token 99 exists in the lattice, but only under root source 0, never under source 1.
        let wrong_parent =
            TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
                (token == 20).then_some(99)
            })
            .unwrap();
        assert_eq!(
            wrong_parent.decision.outcome,
            TargetRowHedgeOutcome::NoPromotableLatticeChild
        );
        assert_eq!(wrong_parent.tree, eagle.tree);
    }

    #[test]
    fn lattice_promotion_dedupes_selected_children_and_rejects_weak_rows() {
        let (lattice, eagle) = promotion_lattice();
        let duplicate =
            TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
                (token == 10).then_some(30)
            })
            .unwrap();
        assert_eq!(
            duplicate.decision.outcome,
            TargetRowHedgeOutcome::DuplicateOnly
        );
        assert_eq!(duplicate.decision.full_path_duplicates, 1);
        assert_eq!(duplicate.tree, eagle.tree);

        let weak = TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
            (token == 10).then_some(77)
        })
        .unwrap();
        assert_eq!(
            weak.decision.outcome,
            TargetRowHedgeOutcome::BelowPromotionScore
        );
        assert_eq!(weak.tree, eagle.tree);
        assert!(weak.decision.comparison_eagle_row.is_some());
        assert!(weak.decision.evicted_eagle_row.is_none());

        let admitted = TargetRowHedgeForest::fuse_lattice_promoted_with_bonus(
            &eagle,
            &lattice,
            8,
            15,
            16.0,
            |token| (token == 10).then_some(77),
        )
        .unwrap();
        assert_eq!(admitted.decision.outcome, TargetRowHedgeOutcome::Inserted);
        assert_eq!(admitted.decision.candidate_token, Some(77));
        assert!(admitted.decision.evicted_eagle_row.is_some());

        for invalid in [
            f32::NEG_INFINITY,
            -0.1,
            TARGET_ROW_LATTICE_MAX_PRIOR_LOG_BONUS + 0.001,
            f32::INFINITY,
            f32::NAN,
        ] {
            assert!(TargetRowHedgeForest::fuse_lattice_promoted_with_bonus(
                &eagle,
                &lattice,
                8,
                15,
                invalid,
                |_| None,
            )
            .is_err());
        }
    }

    #[test]
    fn lattice_promotion_ranks_all_supported_pruned_children_by_current_score() {
        let (mut lattice, eagle) = promotion_lattice();
        let stronger_source = lattice
            .expand(
                1,
                &[DraftCandidateScore {
                    token: 98,
                    log_probability: -0.100_000,
                }],
            )
            .unwrap()[0];
        let fused =
            TargetRowHedgeForest::fuse_lattice_promoted(
                &eagle,
                &lattice,
                8,
                15,
                |token| match token {
                    10 => Some(99),
                    20 => Some(98),
                    _ => None,
                },
            )
            .unwrap();
        assert_eq!(fused.decision.outcome, TargetRowHedgeOutcome::Inserted);
        assert_eq!(fused.decision.candidate_token, Some(98));
        assert_eq!(
            fused.decision.candidate_lattice_source_node,
            Some(stronger_source)
        );
    }

    #[test]
    fn lattice_promotion_score_gate_admits_equality() {
        let (mut lattice, eagle) = promotion_lattice();
        let weakest_score = eagle
            .tree
            .tokens
            .iter()
            .position(|&token| token == 40)
            .map(|row| eagle.cumulative_log_probability[row])
            .unwrap();
        lattice
            .expand(
                0,
                &[DraftCandidateScore {
                    token: 78,
                    log_probability: weakest_score - TARGET_ROW_LATTICE_PRIOR_LOG_BONUS,
                }],
            )
            .unwrap();
        let fused = TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
            (token == 10).then_some(78)
        })
        .unwrap();
        let explicit_default = TargetRowHedgeForest::fuse_lattice_promoted_with_bonus(
            &eagle,
            &lattice,
            8,
            15,
            TARGET_ROW_LATTICE_PRIOR_LOG_BONUS,
            |token| (token == 10).then_some(78),
        )
        .unwrap();
        assert_eq!(fused, explicit_default);
        assert_eq!(fused.decision.outcome, TargetRowHedgeOutcome::Inserted);
        assert_eq!(
            fused.decision.candidate_adjusted_log_score,
            fused.decision.comparison_cumulative_log_probability
        );
    }

    #[test]
    fn lattice_alignment_is_fail_closed() {
        let (lattice, mut eagle) = promotion_lattice();
        eagle.source_node[1] = lattice.nodes().len() + 1;
        let error = TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |_| None)
            .unwrap_err();
        assert!(error.contains("invalid lattice source"));
    }

    #[test]
    fn promoted_and_evicted_leaf_effects_are_counted_exactly() {
        let (lattice, eagle) = promotion_lattice();
        let fused = TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
            (token == 10).then_some(99)
        })
        .unwrap();
        let promoted_row = fused
            .source
            .iter()
            .position(|&source| source == TargetRowHedgeSource::PriorTargetTop1)
            .unwrap();

        let mut accepted_predictions = vec![u32::MAX; fused.tree.nodes()];
        accepted_predictions[0] = 99;
        accepted_predictions[promoted_row] = 123;
        let accepted = fused
            .accept_target_predictions(&accepted_predictions)
            .unwrap();
        assert_eq!(accepted.emitted_tokens, [99, 123]);
        assert_eq!(
            fused
                .exact_counterfactual(&eagle, &accepted_predictions, &accepted)
                .unwrap(),
            TargetRowHedgeCounterfactual {
                inserted_row_accepted: true,
                evicted_edge_would_accept: false,
                emitted_token_delta: 1,
            }
        );

        let mut evicted_predictions = vec![u32::MAX; fused.tree.nodes()];
        evicted_predictions[0] = 40;
        let rejected = fused
            .accept_target_predictions(&evicted_predictions)
            .unwrap();
        assert_eq!(rejected.emitted_tokens, [40]);
        assert_eq!(
            fused
                .exact_counterfactual(&eagle, &evicted_predictions, &rejected)
                .unwrap(),
            TargetRowHedgeCounterfactual {
                inserted_row_accepted: false,
                evicted_edge_would_accept: true,
                emitted_token_delta: -1,
            }
        );
    }

    #[test]
    fn evicted_leaf_counterfactual_follows_a_remapped_deeper_parent() {
        let (mut lattice, _) = promotion_lattice();
        lattice
            .expand(
                6,
                &[DraftCandidateScore {
                    token: 88,
                    log_probability: -1.279_019,
                }],
            )
            .unwrap();
        let eagle = lattice.rerank_connected(8, 15).unwrap();
        let evicted = eagle
            .tree
            .tokens
            .iter()
            .position(|&token| token == 88)
            .unwrap();
        assert_ne!(eagle.tree.parent[evicted], 0);
        let fused = TargetRowHedgeForest::fuse_lattice_promoted(&eagle, &lattice, 8, 15, |token| {
            (token == 10).then_some(99)
        })
        .unwrap();
        assert_eq!(fused.decision.evicted_eagle_row, Some(evicted));
        let old_parent = usize::try_from(eagle.tree.parent[evicted]).unwrap();
        let new_parent = fused
            .eagle_source_row
            .iter()
            .position(|&source| source == Some(old_parent))
            .unwrap();
        assert_ne!(old_parent, new_parent);

        let mut predictions = vec![u32::MAX; fused.tree.nodes()];
        for rows in fused.primary_spine_rows.windows(2).take(2) {
            predictions[rows[0]] = fused.tree.tokens[rows[1]];
        }
        predictions[new_parent] = 88;
        let acceptance = fused.accept_target_predictions(&predictions).unwrap();
        assert_eq!(
            fused
                .exact_counterfactual(&eagle, &predictions, &acceptance)
                .unwrap(),
            TargetRowHedgeCounterfactual {
                inserted_row_accepted: false,
                evicted_edge_would_accept: true,
                emitted_token_delta: -1,
            }
        );
    }
}
