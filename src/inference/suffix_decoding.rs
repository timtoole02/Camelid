//! Model-free suffix decoding (Lane A).
//!
//! Drafts a tree of likely continuations from the *observed* token stream
//! (prompt + generated history) alone — no model forward. The idea (Suffix
//! Decoding, Oliaro et al. 2024): find where the current suffix recurred
//! earlier and propose the tokens that followed those earlier occurrences,
//! preferring the most frequent continuation. Where the n-gram drafter takes a
//! single most-recent match, suffix decoding aggregates *all* matches into a
//! frequency-weighted tree and follows the most-frequent path(s) to adaptive
//! depth.
//!
//! All-Rust, GPU-free, lossless: the target verify is authoritative, so a
//! wrong draft only costs a wasted verify row, never a wrong emission.
//!
//! Memory is bounded: we scan the (bounded) history window for matches and
//! build a small frequency map keyed by the matched continuation tokens; the
//! emitted tree is capped by `max_nodes`/`max_depth`. We do not retain a
//! persistent automaton across calls (the history is rescanned each draft), so
//! peak memory is O(history_window) and the tree is O(max_nodes).

use std::collections::HashMap;

use crate::inference::spec_tree::{TokenTree, TreeDrafter};

/// Model-free suffix-decoding tree drafter.
#[derive(Debug, Clone)]
pub struct SuffixDecodingDrafter {
    /// Longest suffix length to try matching (descending). A longer match is
    /// higher precision.
    pub max_match: usize,
    /// Shortest suffix length worth matching.
    pub min_match: usize,
    /// Cap on the history window scanned for matches (most recent tokens).
    pub window: usize,
    /// Branching factor: at each tree node, expand at most this many distinct
    /// most-frequent successor tokens.
    pub branch: usize,
}

impl Default for SuffixDecodingDrafter {
    fn default() -> Self {
        Self {
            max_match: 4,
            min_match: 2,
            // Recent context carries essentially all the recurrence signal and
            // keeps the per-draft scan O(window) cheap. Bounded by design.
            window: 512,
            branch: 2,
        }
    }
}

impl SuffixDecodingDrafter {
    /// Policy for a linear verifier. Longer matches disambiguate repeated labels
    /// in code/tables; the bounded window retains the source passage while the
    /// answer grows. Tree callers keep their existing, smaller search policy.
    pub fn for_chain() -> Self {
        Self {
            max_match: 32,
            min_match: 3,
            window: 8192,
            branch: 1,
        }
    }

    /// Follow the most frequent continuation at each step, filling a linear
    /// verification window without spending its budget on sibling branches.
    /// Selecting the deepest leaf of a BFS tree is not equivalent: tied depths
    /// can select a less frequent sibling and branching shortens the proposal.
    pub fn draft_chain(&self, history: &[u32], max_tokens: usize) -> Vec<u32> {
        let hist = &history[history.len().saturating_sub(self.window)..];
        if self.min_match == 0 || hist.len() <= self.min_match {
            return Vec::new();
        }
        let mut path = Vec::new();
        for _ in 0..max_tokens {
            let max_n = self
                .max_match
                .min(hist.len().saturating_sub(1) + path.len());
            let next = (self.min_match..=max_n).rev().find_map(|n| {
                let pattern = build_pattern(hist, &path, n);
                if pattern.len() < n {
                    return None;
                }
                let freq = Self::successor_freqs(hist, &pattern);
                freq.into_iter()
                    .max_by(|(ta, ca), (tb, cb)| ca.cmp(cb).then_with(|| tb.cmp(ta)))
                    .map(|(token, _)| token)
            });
            match next {
                Some(token) => path.push(token),
                None => break,
            }
        }
        path
    }

    /// Of all earlier occurrences of `pattern` within `hist`, collect the token
    /// that immediately follows each, with its frequency. `pattern` is a suffix
    /// of the full history; matches whose continuation index falls inside the
    /// pattern's own trailing occurrence are excluded by construction (we only
    /// scan starts strictly before `hist.len() - pattern.len()`).
    fn successor_freqs(hist: &[u32], pattern: &[u32]) -> HashMap<u32, u32> {
        let mut freq: HashMap<u32, u32> = HashMap::new();
        let n = pattern.len();
        if n == 0 || hist.len() <= n {
            return freq;
        }
        let limit = hist.len() - n; // exclude the trailing occurrence (the suffix)
        for start in 0..limit {
            if &hist[start..start + n] == pattern {
                let follow = start + n;
                if follow < hist.len() {
                    *freq.entry(hist[follow]).or_insert(0) += 1;
                }
            }
        }
        freq
    }

    /// Pick up to `branch` most-frequent successors (ties broken by lower token
    /// id, for determinism). Returns successors that occurred at least once.
    fn top_successors(freq: &HashMap<u32, u32>, branch: usize) -> Vec<u32> {
        let mut items: Vec<(u32, u32)> = freq.iter().map(|(&t, &c)| (t, c)).collect();
        // Sort by frequency desc, then token id asc.
        items.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        items.into_iter().take(branch).map(|(t, _)| t).collect()
    }
}

impl TreeDrafter for SuffixDecodingDrafter {
    fn draft_tree(
        &mut self,
        history: &[u32],
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> TokenTree {
        let mut tree = TokenTree::linear(anchor, &[]); // anchor-only root
        if max_nodes <= 1 || max_depth == 0 || self.branch == 0 {
            return tree;
        }
        // Bounded history window (most recent `window` tokens).
        let start = history.len().saturating_sub(self.window);
        let hist = &history[start..];
        if hist.len() <= self.min_match {
            return tree;
        }

        // BFS-expand the tree. For each frontier node we form the "match
        // context" = the tokens along its root-to-node path (excluding the
        // anchor's own value is fine; the anchor IS part of the suffix we
        // search for). We then look for the longest suffix of (history-context
        // + path tokens) that recurred earlier, and attach its top successors.
        //
        // To keep this bounded and model-free, the context we match on is the
        // history suffix extended by the path tokens drafted so far.
        let mut frontier: Vec<usize> = vec![0]; // node indices to expand
                                                // path_tokens[node] = tokens drafted from anchor down to (and
                                                // including) this node, used to extend the match context.
        let mut path_tokens: HashMap<usize, Vec<u32>> = HashMap::new();
        path_tokens.insert(0, Vec::new());

        while let Some(node) = frontier_pop(&mut frontier) {
            if tree.nodes() >= max_nodes {
                break;
            }
            let node_depth = tree.depth[node] as usize;
            if node_depth >= max_depth {
                continue;
            }
            // Build the context to match: history suffix followed by the path
            // tokens drafted so far. The anchor is the last real history token,
            // so the search pattern is a suffix of `hist` extended by the path.
            let path = path_tokens.get(&node).cloned().unwrap_or_default();
            // Try the longest match length down to min_match.
            let max_n = self
                .max_match
                .min(hist.len().saturating_sub(1) + path.len());
            let mut chosen: Vec<u32> = Vec::new();
            for n in (self.min_match..=max_n).rev() {
                let pattern = build_pattern(hist, &path, n);
                if pattern.len() < n {
                    continue;
                }
                let freq = Self::successor_freqs_ctx(hist, &path, &pattern);
                if !freq.is_empty() {
                    chosen = Self::top_successors(&freq, self.branch);
                    break;
                }
            }
            // Attach chosen successors as children of `node`.
            for tok in chosen {
                if tree.nodes() >= max_nodes {
                    break;
                }
                let child = tree.nodes();
                tree.tokens.push(tok);
                tree.parent.push(node as i32);
                tree.depth.push((node_depth + 1) as u16);
                let mut child_path = path.clone();
                child_path.push(tok);
                path_tokens.insert(child, child_path);
                frontier.push(child);
            }
        }
        tree
    }
}

impl SuffixDecodingDrafter {
    /// Successor frequencies for a pattern that is a suffix of (`hist` ++
    /// `path`). All earlier occurrences live in `hist` (the path is just the
    /// few tokens drafted this round at the very tail), so we scan `hist`
    /// alone — no per-call `hist ++ path` allocation — and count the token in
    /// `hist` that follows each match of `pattern`. When `path` is non-empty
    /// the pattern's tail includes the drafted path tokens, so only history
    /// occurrences that actually continued the drafted continuation match,
    /// which is exactly the narrowing we want. O(window) per call.
    fn successor_freqs_ctx(hist: &[u32], path: &[u32], pattern: &[u32]) -> HashMap<u32, u32> {
        let _ = path; // pattern already encodes the path tail
        Self::successor_freqs(hist, pattern)
    }
}

/// Pop from the front (BFS). Small frontier, O(n) shift is fine and keeps BFS
/// order so `parent[i] < i` holds in the built tree.
fn frontier_pop(frontier: &mut Vec<usize>) -> Option<usize> {
    if frontier.is_empty() {
        None
    } else {
        Some(frontier.remove(0))
    }
}

/// The length-`n` suffix of (`hist` ++ `path`).
fn build_pattern(hist: &[u32], path: &[u32], n: usize) -> Vec<u32> {
    let total = hist.len() + path.len();
    if total < n {
        return Vec::new();
    }
    let mut combined: Vec<u32> = Vec::with_capacity(n);
    let want = n;
    if path.len() >= want {
        combined.extend_from_slice(&path[path.len() - want..]);
    } else {
        let from_hist = want - path.len();
        combined.extend_from_slice(&hist[hist.len() - from_hist..]);
        combined.extend_from_slice(path);
    }
    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_tree::TREE_MAX_NODES;

    #[test]
    fn anchor_only_when_no_repeat() {
        let mut d = SuffixDecodingDrafter::default();
        let history = vec![1, 2, 3, 4, 5];
        let tree = d.draft_tree(&history, 5, TREE_MAX_NODES, 6);
        assert_eq!(tree.nodes(), 1); // root only
        assert_eq!(tree.tokens, vec![5]);
    }

    #[test]
    fn drafts_most_frequent_continuation() {
        // Suffix [3,4] recurs; after it we mostly see 7 (twice) and once 9.
        // history: 3 4 7 ... 3 4 7 ... 3 4 9 ... 3 4  (trailing suffix)
        let mut d = SuffixDecodingDrafter {
            max_match: 2,
            min_match: 2,
            window: 4096,
            branch: 1,
        };
        let history = vec![3, 4, 7, 0, 3, 4, 7, 0, 3, 4, 9, 0, 3, 4];
        let anchor = 4;
        let tree = d.draft_tree(&history, anchor, TREE_MAX_NODES, 4);
        // Root + most-frequent successor 7.
        assert!(tree.nodes() >= 2);
        assert_eq!(tree.tokens[0], 4);
        assert_eq!(tree.tokens[1], 7, "7 follows [3,4] more often than 9");
        // BFS invariant.
        for i in 1..tree.nodes() {
            assert!(tree.parent[i] < i as i32);
        }
    }

    #[test]
    fn branching_expands_multiple_successors() {
        let mut d = SuffixDecodingDrafter {
            max_match: 2,
            min_match: 2,
            window: 4096,
            branch: 2,
        };
        // [3,4] -> 7 (x2) and 9 (x1): branch=2 should attach both.
        let history = vec![3, 4, 7, 0, 3, 4, 7, 0, 3, 4, 9, 0, 3, 4];
        let tree = d.draft_tree(&history, 4, TREE_MAX_NODES, 1);
        // depth capped at 1, so root + (up to) 2 children.
        let children: Vec<u32> = (1..tree.nodes())
            .filter(|&i| tree.parent[i] == 0)
            .map(|i| tree.tokens[i])
            .collect();
        assert!(children.contains(&7));
        assert!(children.contains(&9));
        assert_eq!(tree.max_depth(), 1);
    }

    #[test]
    fn respects_node_and_depth_caps() {
        let mut d = SuffixDecodingDrafter::default();
        let history = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2];
        let tree = d.draft_tree(&history, 2, 5, 2);
        assert!(tree.nodes() <= 5, "node cap respected");
        assert!(tree.max_depth() <= 2, "depth cap respected");
    }

    #[test]
    fn chain_prefers_frequent_branch_and_uses_the_whole_budget() {
        let d = SuffixDecodingDrafter {
            max_match: 2,
            min_match: 2,
            window: 512,
            branch: 2,
        };
        let history = [3, 4, 7, 8, 10, 0, 3, 4, 7, 8, 10, 0, 3, 4, 9, 0, 3, 4];
        assert_eq!(d.draft_chain(&history, 3), [7, 8, 10]);
        assert!(d.draft_chain(&history, 0).is_empty());
    }

    #[test]
    fn chain_uses_long_context_to_disambiguate_common_record_suffixes() {
        let d = SuffixDecodingDrafter::for_chain();
        let history = [10, 1, 2, 3, 4, 90, 20, 1, 2, 3, 4, 80, 20, 1, 2, 3, 4];
        assert_eq!(d.draft_chain(&history, 1), [80]);
    }

    #[test]
    fn chain_retains_source_beyond_old_512_token_window_but_honors_its_bound() {
        let mut d = SuffixDecodingDrafter::for_chain();
        let mut history = vec![11, 12, 13, 14, 15];
        history.extend(100..700);
        history.extend([11, 12, 13]);
        assert_eq!(d.draft_chain(&history, 2), [14, 15]);
        d.window = 512;
        assert!(d.draft_chain(&history, 2).is_empty());
    }
}
