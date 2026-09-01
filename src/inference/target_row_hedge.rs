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
//! tree.  In particular, a prior-target token need not exist in EAGLE's private draft lattice.

use std::collections::{HashMap, HashSet};

use super::spec_tree::{ScoredTokenTree, TokenTree};

/// Mini2's fixed verifier-width speed lane, including the root anchor.
pub const TARGET_ROW_HEDGE_MAX_NODES: usize = 8;

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
    DuplicateOnly,
    NoReplaceableNeuralHedge,
}

/// Stable, model-free evidence for one fusion decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetRowHedgeDecision {
    pub outcome: TargetRowHedgeOutcome,
    pub supported_primary_rows: usize,
    pub full_path_duplicates: usize,
    pub candidate_token: Option<u32>,
    pub candidate_parent_depth: Option<usize>,
    pub candidate_depth: Option<usize>,
    pub evicted_eagle_row: Option<usize>,
    pub neural_hedges_retained: usize,
}

/// One verifier-ready forest plus enough provenance to audit every row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetRowHedgeForest {
    pub tree: TokenTree,
    pub source: Vec<TargetRowHedgeSource>,
    /// Original EAGLE verifier row for retained learned rows.  The inserted target-row candidate
    /// is deliberately `None`: it is safe even when its token has no EAGLE lattice/T2D entry.
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
        let primary: HashSet<usize> = eagle.primary_spine_rows.iter().copied().collect();
        let mut source = (0..tree.nodes())
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
            selected_candidate = Some((parent_row, token, path));
            break;
        }

        let Some((candidate_parent, candidate_token, _candidate_path)) = selected_candidate else {
            let outcome = if supported_primary_rows == 0 {
                TargetRowHedgeOutcome::NoPriorTargetRow
            } else {
                TargetRowHedgeOutcome::DuplicateOnly
            };
            return Ok(Self {
                tree: tree.clone(),
                source,
                eagle_source_row: (0..tree.nodes()).map(Some).collect(),
                primary_spine_rows: eagle.primary_spine_rows.clone(),
                decision: TargetRowHedgeDecision {
                    outcome,
                    supported_primary_rows,
                    full_path_duplicates,
                    candidate_token: None,
                    candidate_parent_depth: None,
                    candidate_depth: None,
                    evicted_eagle_row: None,
                    neural_hedges_retained: tree
                        .nodes()
                        .saturating_sub(eagle.primary_spine_rows.len()),
                },
            });
        };

        let evicted_eagle_row = if tree.nodes() < max_nodes {
            None
        } else {
            let hedge_count = tree.nodes().saturating_sub(primary.len());
            if hedge_count < 2 {
                let neural_hedges_retained = source
                    .iter()
                    .filter(|&&item| item.is_neural_hedge())
                    .count();
                return Ok(Self {
                    tree: tree.clone(),
                    source,
                    eagle_source_row: (0..tree.nodes()).map(Some).collect(),
                    primary_spine_rows: eagle.primary_spine_rows.clone(),
                    decision: TargetRowHedgeDecision {
                        outcome: TargetRowHedgeOutcome::NoReplaceableNeuralHedge,
                        supported_primary_rows,
                        full_path_duplicates,
                        candidate_token: Some(candidate_token),
                        candidate_parent_depth: Some(usize::from(tree.depth[candidate_parent])),
                        candidate_depth: Some(usize::from(tree.depth[candidate_parent]) + 1),
                        evicted_eagle_row: None,
                        neural_hedges_retained,
                    },
                });
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
                        // Equal confidence evicts the later stable expansion row.
                        .then_with(|| right.cmp(&left))
                });
            let Some(row) = weakest_leaf else {
                let neural_hedges_retained = source
                    .iter()
                    .filter(|&&item| item.is_neural_hedge())
                    .count();
                return Ok(Self {
                    tree: tree.clone(),
                    source,
                    eagle_source_row: (0..tree.nodes()).map(Some).collect(),
                    primary_spine_rows: eagle.primary_spine_rows.clone(),
                    decision: TargetRowHedgeDecision {
                        outcome: TargetRowHedgeOutcome::NoReplaceableNeuralHedge,
                        supported_primary_rows,
                        full_path_duplicates,
                        candidate_token: Some(candidate_token),
                        candidate_parent_depth: Some(usize::from(tree.depth[candidate_parent])),
                        candidate_depth: Some(usize::from(tree.depth[candidate_parent]) + 1),
                        evicted_eagle_row: None,
                        neural_hedges_retained,
                    },
                });
            };
            Some(row)
        };

        let synthetic = tree.nodes();
        let mut retained = (0..tree.nodes())
            .filter(|row| Some(*row) != evicted_eagle_row)
            .collect::<Vec<_>>();
        retained.push(synthetic);
        retained.sort_by_key(|&row| {
            if row == synthetic {
                (tree.depth[candidate_parent].saturating_add(1), row)
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
                tokens.push(candidate_token);
                parent.push(
                    i32::try_from(remap[candidate_parent])
                        .map_err(|_| "target-row hedge verifier parent exceeds i32".to_string())?,
                );
                depth.push(
                    tree.depth[candidate_parent]
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
            decision: TargetRowHedgeDecision {
                outcome: TargetRowHedgeOutcome::Inserted,
                supported_primary_rows,
                full_path_duplicates,
                candidate_token: Some(candidate_token),
                candidate_parent_depth: Some(usize::from(tree.depth[candidate_parent])),
                candidate_depth: Some(usize::from(tree.depth[candidate_parent]) + 1),
                evicted_eagle_row,
                neural_hedges_retained,
            },
        })
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
}
