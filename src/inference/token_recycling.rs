//! Token recycling (Lane A).
//!
//! Maintains a sparse per-vocabulary adjacency map — for each token, the
//! top-`k` tokens most often observed to follow it — and BFS-expands the
//! anchor's adjacency into a draft tree. The name (Token Recycling, Luo et al.
//! 2024) is from reusing the model's own observed/accepted token transitions
//! as the draft source: no separate draft model, no model forward.
//!
//! Model-free, all-Rust, GPU-free, lossless (the target verify is
//! authoritative). The adjacency is sparse (a `HashMap<u32, ...>` keyed on the
//! tokens actually seen), so memory is O(distinct tokens × k), not O(vocab²).
//!
//! Learning scope: the default drafter learns from the *accepted token stream*
//! alone — `observe`/`learn` feed realized history transitions. The experimental
//! benchmark lane can additionally call `replace_target_candidates` with the
//! resident verifier's compact target top-k rows. No logits are read back and
//! the target verifier remains authoritative in either mode.

use std::collections::HashMap;

use crate::inference::spec_tree::{TokenTree, TreeDrafter};

/// Fixed-point scale for structural Token Recycling admission evidence.
pub const TOKEN_RECYCLING_CONFIDENCE_Q16_ONE: u32 = 1 << 16;
/// A hybrid round needs two already-supported rank-0 edges before replacing EAGLE drafting.
pub const TOKEN_RECYCLING_MIN_PRIMARY_DEPTH: usize = 2;
/// At least half of the possible parent rows above the proposed tree's deepest level must have
/// exact target rows. This rejects a deep primary path when most competing parent slots are cold.
pub const TOKEN_RECYCLING_MIN_KNOWN_PARENT_SHARE_Q16: u32 = TOKEN_RECYCLING_CONFIDENCE_Q16_ONE / 2;

/// Deterministic, model-free evidence used by the benchmark EAGLE/TR hybrid.
///
/// Every field is computed from target rows that were installed before the current target
/// verification. There are no logits, probabilities, future outcomes, or accepted-stream counts
/// in this decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenRecyclingTreeEvidence {
    pub root_known: bool,
    pub nodes: usize,
    pub max_depth: usize,
    /// Depth of the first-child path. Target rows preserve rank order, so this is the depth of
    /// the candidate chain formed by the previously observed target top-1 at each step.
    pub primary_depth: usize,
    /// Proposed nodes above the deepest level: these are the rows that could have expanded the
    /// known forest further during this draft.
    pub parent_rows: usize,
    pub known_parent_rows: usize,
    pub known_parent_share_q16: u32,
    pub admitted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenRecyclingTreeProposal {
    pub tree: TokenTree,
    pub evidence: TokenRecyclingTreeEvidence,
}

/// Per-token successor counts: `succ[token]` maps a following token to how
/// often it has been observed in that position.
#[derive(Debug, Clone)]
pub struct TokenRecyclingDrafter {
    /// token -> (successor token -> count).
    succ: HashMap<u32, HashMap<u32, u32>>,
    /// Latest verifier top-k row for a token, preserving target rank order.
    ///
    /// The Token Recycling update is row replacement (`matrix[input] = top_k(logits)`), not a
    /// frequency accumulator. Keeping this separate from `succ` preserves the accepted-stream
    /// fallback while making verifier evidence deterministic and immediately authoritative.
    target_rows: HashMap<u32, Vec<u32>>,
    /// Successors kept per token when building a tree.
    pub topk: usize,
    /// Branching factor at each tree node (≤ topk).
    pub branch: usize,
    /// Cap on distinct successors retained per token (bounds memory).
    pub max_succ_per_token: usize,
}

impl TokenRecyclingDrafter {
    pub fn new() -> Self {
        Self {
            succ: HashMap::new(),
            target_rows: HashMap::new(),
            topk: 4,
            branch: 2,
            max_succ_per_token: 16,
        }
    }

    /// Record a single observed transition `from -> to`.
    pub fn observe(&mut self, from: u32, to: u32) {
        let entry = self.succ.entry(from).or_default();
        *entry.entry(to).or_insert(0) += 1;
        // Bound memory: if a token accrues too many distinct successors, drop
        // the least-frequent one (keep the hot set).
        if entry.len() > self.max_succ_per_token {
            if let Some((&victim, _)) = entry.iter().min_by_key(|(_, &c)| c) {
                entry.remove(&victim);
            }
        }
    }

    /// Replace `from`'s adjacency with one verified target candidate row.
    ///
    /// This matches Token Recycling's adjacency-matrix update: the latest target row overwrites
    /// the previous row and retains target rank order. Duplicates are removed, the verifier's
    /// `u32::MAX` exhausted-rank sentinel is ignored, and the configured per-token cap is applied.
    /// Returns the number of valid distinct candidates stored for benchmark telemetry.
    pub fn replace_target_candidates(&mut self, from: u32, candidates: &[u32]) -> usize {
        let mut row = Vec::with_capacity(candidates.len().min(self.max_succ_per_token));
        for (rank, &to) in candidates.iter().enumerate() {
            if to == u32::MAX || candidates[..rank].contains(&to) {
                continue;
            }
            if row.len() == self.max_succ_per_token {
                break;
            }
            row.push(to);
        }
        let stored = row.len();
        if row.is_empty() {
            self.target_rows.remove(&from);
        } else {
            self.target_rows.insert(from, row);
        }
        stored
    }

    /// Whether this token already has an exact target candidate row in the current run.
    pub fn has_target_candidates(&self, token: u32) -> bool {
        self.target_rows.contains_key(&token)
    }

    /// Target greedy top-1 saved by a completed, earlier verifier row.
    ///
    /// This accessor is read-only so a caller can snapshot causal hedge proposals before its
    /// current target verification.  Newly verified rows must still be installed later through
    /// [`Self::replace_target_candidates`].
    pub fn prior_target_top1(&self, token: u32) -> Option<u32> {
        self.target_rows
            .get(&token)
            .and_then(|row| row.first())
            .copied()
    }

    /// Build and assess a tree using only exact target rows already known before this round.
    ///
    /// Unlike [`TreeDrafter::draft_tree`], this is read-only and never consults or mutates the
    /// accepted-stream frequency fallback. It is therefore suitable for a deterministic hybrid
    /// admission decision: EAGLE remains the cold/low-support fallback, while Token Recycling is
    /// admitted only once its in-session target-row forest is both deep and sufficiently covered.
    pub fn draft_known_target_tree(
        &self,
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> TokenRecyclingTreeProposal {
        let root_known = self.has_target_candidates(anchor);
        let mut tree = TokenTree::linear(anchor, &[]);
        if max_nodes > 1 && max_depth > 0 && self.branch > 0 {
            let mut frontier = vec![0usize];
            while let Some(node) = pop_front(&mut frontier) {
                if tree.nodes() >= max_nodes {
                    break;
                }
                let node_depth = tree.depth[node] as usize;
                if node_depth >= max_depth {
                    continue;
                }
                let Some(successors) = self.target_rows.get(&tree.tokens[node]) else {
                    continue;
                };
                for &token in successors.iter().take(self.branch.min(self.topk)) {
                    if tree.nodes() >= max_nodes {
                        break;
                    }
                    let child = tree.nodes();
                    tree.tokens.push(token);
                    tree.parent.push(node as i32);
                    tree.depth.push((node_depth + 1) as u16);
                    frontier.push(child);
                }
            }
        }

        let proposed_max_depth = tree.max_depth();
        let mut primary_depth = 0usize;
        let mut primary_parent = 0usize;
        loop {
            let Some(child) = ((primary_parent + 1)..tree.nodes())
                .find(|&node| tree.parent[node] == primary_parent as i32)
            else {
                break;
            };
            primary_depth += 1;
            primary_parent = child;
        }
        let parent_rows = tree
            .depth
            .iter()
            .filter(|&&depth| (depth as usize) < proposed_max_depth)
            .count();
        let known_parent_rows = tree
            .tokens
            .iter()
            .zip(&tree.depth)
            .filter(|(token, depth)| {
                (**depth as usize) < proposed_max_depth && self.has_target_candidates(**token)
            })
            .count();
        let known_parent_share_q16 = if parent_rows == 0 {
            0
        } else {
            ((known_parent_rows as u64 * TOKEN_RECYCLING_CONFIDENCE_Q16_ONE as u64)
                / parent_rows as u64) as u32
        };
        let admitted = root_known
            && primary_depth >= TOKEN_RECYCLING_MIN_PRIMARY_DEPTH
            && known_parent_share_q16 >= TOKEN_RECYCLING_MIN_KNOWN_PARENT_SHARE_Q16;
        let evidence = TokenRecyclingTreeEvidence {
            root_known,
            nodes: tree.nodes(),
            max_depth: proposed_max_depth,
            primary_depth,
            parent_rows,
            known_parent_rows,
            known_parent_share_q16,
            admitted,
        };
        TokenRecyclingTreeProposal { tree, evidence }
    }

    /// Learn every adjacent transition in an observed token stream (the
    /// accepted/history stream). Idempotent only in the sense that repeated
    /// calls accumulate counts — call once per newly-committed segment.
    pub fn learn(&mut self, stream: &[u32]) {
        for w in stream.windows(2) {
            self.observe(w[0], w[1]);
        }
    }

    /// Top successors of `token`, most-frequent first (ties by lower id), up to
    /// `n`.
    fn top_successors(&self, token: u32, n: usize) -> Vec<u32> {
        if let Some(row) = self.target_rows.get(&token) {
            return row.iter().copied().take(n).collect();
        }
        match self.succ.get(&token) {
            None => Vec::new(),
            Some(map) => {
                let mut items: Vec<(u32, u32)> = map.iter().map(|(&t, &c)| (t, c)).collect();
                items.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                items.into_iter().take(n).map(|(t, _)| t).collect()
            }
        }
    }
}

impl Default for TokenRecyclingDrafter {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeDrafter for TokenRecyclingDrafter {
    fn draft_tree(
        &mut self,
        history: &[u32],
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> TokenTree {
        // Opportunistically learn the most recent transition so a fresh
        // drafter still proposes something on repetitive streams. (The
        // adjacency persists across calls.)
        if history.len() >= 2 {
            let n = history.len();
            self.observe(history[n - 2], history[n - 1]);
        }
        let mut tree = TokenTree::linear(anchor, &[]);
        if max_nodes <= 1 || max_depth == 0 || self.branch == 0 {
            return tree;
        }
        let mut frontier: Vec<usize> = vec![0];
        while let Some(node) = pop_front(&mut frontier) {
            if tree.nodes() >= max_nodes {
                break;
            }
            let node_depth = tree.depth[node] as usize;
            if node_depth >= max_depth {
                continue;
            }
            let parent_token = tree.tokens[node];
            let succ = self.top_successors(parent_token, self.branch.min(self.topk));
            for tok in succ {
                if tree.nodes() >= max_nodes {
                    break;
                }
                let child = tree.nodes();
                tree.tokens.push(tok);
                tree.parent.push(node as i32);
                tree.depth.push((node_depth + 1) as u16);
                frontier.push(child);
            }
        }
        tree
    }
}

/// BFS front pop (small frontier; keeps BFS order so `parent[i] < i`).
fn pop_front(frontier: &mut Vec<usize>) -> Option<usize> {
    if frontier.is_empty() {
        None
    } else {
        Some(frontier.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_tree::TREE_MAX_NODES;

    #[test]
    fn learns_and_drafts_top_successor() {
        let mut d = TokenRecyclingDrafter::new();
        // 5 -> 6 observed three times, 5 -> 7 once.
        d.learn(&[5, 6, 5, 6, 5, 6, 5, 7]);
        let tree = d.draft_tree(&[9, 9, 5], 5, TREE_MAX_NODES, 3);
        assert!(tree.nodes() >= 2);
        assert_eq!(tree.tokens[0], 5);
        // 6 is the more-frequent successor and must come first.
        let first_child = (1..tree.nodes())
            .find(|&i| tree.parent[i] == 0)
            .expect("a child exists");
        assert_eq!(tree.tokens[first_child], 6);
    }

    #[test]
    fn empty_adjacency_yields_anchor_only() {
        let mut d = TokenRecyclingDrafter::new();
        let tree = d.draft_tree(&[1], 1, TREE_MAX_NODES, 3);
        assert_eq!(tree.nodes(), 1);
    }

    #[test]
    fn bounds_distinct_successors() {
        let mut d = TokenRecyclingDrafter::new();
        d.max_succ_per_token = 3;
        // Feed 5 distinct successors of token 1; only 3 should remain.
        for to in [10u32, 11, 12, 13, 14] {
            d.observe(1, to);
        }
        assert!(d.succ.get(&1).unwrap().len() <= 3);
    }

    #[test]
    fn latest_target_row_bootstraps_and_overwrites_deterministically() {
        let mut d = TokenRecyclingDrafter::new();
        d.branch = 3;
        let cold = d.draft_tree(&[], 10, 4, 1);
        assert_eq!(cold.nodes(), 1, "cold row starts anchor-only");
        assert_eq!(
            d.replace_target_candidates(10, &[5, 3, 5, u32::MAX, 7]),
            3,
            "duplicate ids and exhausted-rank sentinels are not double-counted"
        );
        let tree = d.draft_tree(&[], 10, 4, 1);
        assert_eq!(
            tree.tokens,
            [10, 5, 3, 7],
            "a seed row immediately drafts in target rank order"
        );
        assert_eq!(tree.parent, [-1, 0, 0, 0]);

        assert_eq!(d.replace_target_candidates(10, &[7, 9]), 2);
        let tree = d.draft_tree(&[], 10, 4, 1);
        assert_eq!(
            tree.tokens,
            [10, 7, 9],
            "latest target evidence replaces, rather than accumulates with, the old row"
        );
    }

    #[test]
    fn known_target_forest_admission_is_pure_deep_and_coverage_gated() {
        let mut d = TokenRecyclingDrafter::new();
        d.branch = 2;

        let cold = d.draft_known_target_tree(10, 7, 3);
        assert_eq!(cold.tree, TokenTree::linear(10, &[]));
        assert!(!cold.evidence.root_known);
        assert!(!cold.evidence.admitted);

        d.replace_target_candidates(10, &[20, 30]);
        let shallow = d.draft_known_target_tree(10, 7, 3);
        assert!(shallow.evidence.root_known);
        assert_eq!(shallow.evidence.primary_depth, 1);
        assert!(!shallow.evidence.admitted);

        d.replace_target_candidates(20, &[40, 50]);
        let deep = d.draft_known_target_tree(10, 7, 3);
        assert_eq!(deep.evidence.primary_depth, 2);
        assert_eq!(deep.evidence.parent_rows, 3);
        assert_eq!(deep.evidence.known_parent_rows, 2);
        assert_eq!(deep.evidence.known_parent_share_q16, 43_690);
        assert!(deep.evidence.admitted);
        assert_eq!(deep, d.draft_known_target_tree(10, 7, 3));
    }

    #[test]
    fn known_target_forest_rejects_supported_alternative_without_primary_depth() {
        let mut d = TokenRecyclingDrafter::new();
        d.branch = 2;
        d.replace_target_candidates(10, &[20, 30]);
        d.replace_target_candidates(30, &[40, 50]);

        let proposal = d.draft_known_target_tree(10, 7, 3);
        assert_eq!(proposal.evidence.max_depth, 2);
        assert_eq!(proposal.evidence.primary_depth, 1);
        assert_eq!(proposal.evidence.known_parent_rows, 2);
        assert!(!proposal.evidence.admitted);
    }

    #[test]
    fn known_target_forest_rejects_low_known_parent_share() {
        let mut d = TokenRecyclingDrafter::new();
        d.branch = 4;
        d.topk = 4;
        d.replace_target_candidates(10, &[20, 30, 40, 50]);
        d.replace_target_candidates(20, &[60]);

        let proposal = d.draft_known_target_tree(10, 9, 3);
        assert_eq!(proposal.evidence.primary_depth, 2);
        assert_eq!(proposal.evidence.parent_rows, 5);
        assert_eq!(proposal.evidence.known_parent_rows, 2);
        assert!(
            proposal.evidence.known_parent_share_q16 < TOKEN_RECYCLING_MIN_KNOWN_PARENT_SHARE_Q16
        );
        assert!(!proposal.evidence.admitted);
    }

    #[test]
    fn branches_and_recurses() {
        let mut d = TokenRecyclingDrafter::new();
        d.branch = 2;
        // 1 -> {2,3}, 2 -> {4}, 3 -> {5}
        d.learn(&[1, 2, 4, 9, 1, 2, 4, 9, 1, 3, 5, 9, 1, 3, 5]);
        let tree = d.draft_tree(&[0, 1], 1, TREE_MAX_NODES, 3);
        // Root 1 should branch to 2 and 3 (both frequent), each recursing.
        let depth1: Vec<u32> = (1..tree.nodes())
            .filter(|&i| tree.parent[i] == 0)
            .map(|i| tree.tokens[i])
            .collect();
        assert!(depth1.contains(&2) || depth1.contains(&3));
        assert!(tree.max_depth() >= 1);
        for i in 1..tree.nodes() {
            assert!(tree.parent[i] < i as i32);
        }
    }

    #[test]
    fn respects_caps() {
        let mut d = TokenRecyclingDrafter::new();
        d.branch = 3;
        d.learn(&[1, 2, 1, 3, 1, 4, 2, 5, 2, 6, 3, 7, 3, 8]);
        let tree = d.draft_tree(&[0, 1], 1, 4, 2);
        assert!(tree.nodes() <= 4);
        assert!(tree.max_depth() <= 2);
    }
}
