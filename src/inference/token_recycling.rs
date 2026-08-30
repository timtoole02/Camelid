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
