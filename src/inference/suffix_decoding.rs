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

/// Fixed-point scale used by suffix admission evidence. Keeping the decision in
/// integer arithmetic makes the same token history take the same route on every
/// host, independent of floating-point library details.
pub const SUFFIX_CONFIDENCE_Q16_ONE: u32 = 1 << 16;
/// A proposed token remains in the chain only while its conservatively shrunk
/// prefix survival is at least 25%.
pub const SUFFIX_MIN_PREFIX_SURVIVAL_Q16: u32 = SUFFIX_CONFIDENCE_Q16_ONE / 4;
/// The admitted prefix must predict at least 1.25 accepted draft tokens. This
/// lies between the measured weak-prose suffix yield (1.93 emitted = 0.93
/// drafts after removing the target bonus) and useful-code yield (2.92 emitted
/// = 1.92 drafts). The target's guaranteed bonus token is deliberately excluded.
pub const SUFFIX_MIN_EXPECTED_ACCEPTED_Q16: u32 = SUFFIX_CONFIDENCE_Q16_ONE * 5 / 4;
/// A suffix verify must cover at least three draft rows to compete with the
/// learned-tree fallback. Shorter evidence falls through without target work.
pub const SUFFIX_MIN_CONFIDENT_DEPTH: usize = 3;

/// Model-free evidence attached to one flattened suffix-chain decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SuffixAdmissionEvidence {
    /// Deepest path the legacy suffix tree could have returned.
    pub raw_depth: usize,
    /// Depth of the best prefix that cleared the fixed-point survival floor.
    pub confident_depth: usize,
    /// Match length used to choose the first token on the selected/raw path.
    pub root_match_len: usize,
    /// Number of earlier occurrences supporting any root successor.
    pub root_support: u32,
    /// Occurrences supporting the selected first token.
    pub root_branch_count: u32,
    /// Sum of conservative prefix-survival mass along the selected path.
    pub expected_accepted_q16: u32,
    /// Conservative probability mass remaining at the selected terminal token.
    pub terminal_survival_q16: u32,
    /// True only when every admission threshold cleared.
    pub admitted: bool,
}

/// A suffix chain plus the deterministic evidence that admitted or declined it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SuffixChainProposal {
    pub tokens: Vec<u32>,
    pub evidence: SuffixAdmissionEvidence,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SuffixNodeEvidence {
    root_match_len: usize,
    root_support: u32,
    root_branch_count: u32,
    root_admissible: bool,
    survival_q16: u32,
    expected_accepted_q16: u32,
}

struct ScoredSuffixTree {
    tree: TokenTree,
    evidence: Vec<SuffixNodeEvidence>,
}

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
    /// id, for determinism), retaining counts for confidence-qualified chains.
    fn top_successors(freq: &HashMap<u32, u32>, branch: usize) -> Vec<(u32, u32)> {
        let mut items: Vec<(u32, u32)> = freq.iter().map(|(&t, &c)| (t, c)).collect();
        // Sort by frequency desc, then token id asc.
        items.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        items.into_iter().take(branch).collect()
    }

    /// Short matches need substantially more evidence. A length-2 suffix may
    /// still be valuable for syntax/code recurrence, but only when at least four
    /// observations agree at 75% or better. Longer contexts need two agreeing
    /// observations; the path-survival score handles any remaining ambiguity.
    fn root_evidence_admissible(match_len: usize, support: u32, branch_count: u32) -> bool {
        if branch_count < 2 {
            return false;
        }
        match match_len {
            0 | 1 => false,
            2 => support >= 4 && u64::from(branch_count) * 4 >= u64::from(support) * 3,
            _ => true,
        }
    }

    fn child_evidence(
        parent: SuffixNodeEvidence,
        parent_index: usize,
        match_len: usize,
        support: u32,
        branch_count: u32,
    ) -> SuffixNodeEvidence {
        // Reserve one pseudo-observation for an unseen continuation. This is not
        // a probability claim; it is deterministic shrinkage that prevents a
        // singleton (share=100%, entropy=0) from masquerading as a strong chain.
        let survival_q16 =
            ((parent.survival_q16 as u64 * branch_count as u64) / (support as u64 + 1)) as u32;
        let expected_accepted_q16 = parent.expected_accepted_q16.saturating_add(survival_q16);
        let (root_match_len, root_support, root_branch_count, root_admissible) =
            if parent_index == 0 {
                (
                    match_len,
                    support,
                    branch_count,
                    Self::root_evidence_admissible(match_len, support, branch_count),
                )
            } else {
                (
                    parent.root_match_len,
                    parent.root_support,
                    parent.root_branch_count,
                    parent.root_admissible,
                )
            };
        SuffixNodeEvidence {
            root_match_len,
            root_support,
            root_branch_count,
            root_admissible,
            survival_q16,
            expected_accepted_q16,
        }
    }

    fn draft_scored_tree(
        &mut self,
        history: &[u32],
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> ScoredSuffixTree {
        let mut tree = TokenTree::linear(anchor, &[]); // anchor-only root
        let mut evidence = vec![SuffixNodeEvidence {
            survival_q16: SUFFIX_CONFIDENCE_Q16_ONE,
            ..SuffixNodeEvidence::default()
        }];
        if max_nodes <= 1 || max_depth == 0 || self.branch == 0 {
            return ScoredSuffixTree { tree, evidence };
        }
        // Bounded history window (most recent `window` tokens).
        let start = history.len().saturating_sub(self.window);
        let hist = &history[start..];
        if hist.len() <= self.min_match {
            return ScoredSuffixTree { tree, evidence };
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
            let mut chosen: Vec<(u32, u32)> = Vec::new();
            let mut chosen_match_len = 0usize;
            let mut chosen_support = 0u32;
            for n in (self.min_match..=max_n).rev() {
                let pattern = build_pattern(hist, &path, n);
                if pattern.len() < n {
                    continue;
                }
                let freq = Self::successor_freqs_ctx(hist, &path, &pattern);
                if !freq.is_empty() {
                    chosen_match_len = n;
                    chosen_support = freq.values().copied().sum();
                    chosen = Self::top_successors(&freq, self.branch);
                    break;
                }
            }
            // Attach chosen successors as children of `node`.
            for (tok, branch_count) in chosen {
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
                evidence.push(Self::child_evidence(
                    evidence[node],
                    node,
                    chosen_match_len,
                    chosen_support,
                    branch_count,
                ));
                frontier.push(child);
            }
        }
        debug_assert_eq!(tree.nodes(), evidence.len());
        ScoredSuffixTree { tree, evidence }
    }

    fn best_confident_node(scored: &ScoredSuffixTree) -> usize {
        let mut best = 0usize;
        for node in 1..scored.tree.nodes() {
            let node_evidence = scored.evidence[node];
            if !node_evidence.root_admissible
                || node_evidence.survival_q16 < SUFFIX_MIN_PREFIX_SURVIVAL_Q16
            {
                continue;
            }
            let best_evidence = scored.evidence[best];
            let better = best == 0
                || node_evidence.expected_accepted_q16 > best_evidence.expected_accepted_q16
                || (node_evidence.expected_accepted_q16 == best_evidence.expected_accepted_q16
                    && scored.tree.depth[node] > scored.tree.depth[best])
                || (node_evidence.expected_accepted_q16 == best_evidence.expected_accepted_q16
                    && scored.tree.depth[node] == scored.tree.depth[best]
                    && node_evidence.survival_q16 > best_evidence.survival_q16);
            if better {
                best = node;
            }
        }
        best
    }

    /// Build the legacy suffix tree, then flatten only the strongest
    /// confidence-qualified path. Generic branching-tree callers keep using
    /// [`TreeDrafter::draft_tree`] and therefore retain their existing behavior.
    pub fn draft_confident_chain(
        &mut self,
        history: &[u32],
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> SuffixChainProposal {
        let scored = self.draft_scored_tree(history, anchor, max_nodes, max_depth);
        let mut raw_leaf = 0usize;
        for node in 1..scored.tree.nodes() {
            if scored.tree.depth[node] > scored.tree.depth[raw_leaf] {
                raw_leaf = node;
            }
        }
        let raw_depth = scored.tree.depth[raw_leaf] as usize;

        // Compare path utility, not raw depth: a rare deep branch must not beat
        // a shorter, much better-supported continuation merely because it copied
        // more tokens from one old occurrence.
        let best = Self::best_confident_node(&scored);

        let selected = if best == 0 {
            scored.evidence[raw_leaf]
        } else {
            scored.evidence[best]
        };
        let confident_depth = if best == 0 {
            0
        } else {
            scored.tree.depth[best] as usize
        };
        let admitted = confident_depth >= SUFFIX_MIN_CONFIDENT_DEPTH
            && selected.expected_accepted_q16 >= SUFFIX_MIN_EXPECTED_ACCEPTED_Q16;
        let tokens = if admitted {
            scored
                .tree
                .path_to(best)
                .into_iter()
                .skip(1)
                .map(|node| scored.tree.tokens[node])
                .take(max_depth)
                .collect()
        } else {
            Vec::new()
        };
        SuffixChainProposal {
            tokens,
            evidence: SuffixAdmissionEvidence {
                raw_depth,
                confident_depth,
                root_match_len: selected.root_match_len,
                root_support: selected.root_support,
                root_branch_count: selected.root_branch_count,
                expected_accepted_q16: selected.expected_accepted_q16,
                terminal_survival_q16: selected.survival_q16,
                admitted,
            },
        }
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
        self.draft_scored_tree(history, anchor, max_nodes, max_depth)
            .tree
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
        assert_eq!(children, [7, 9], "legacy frequency ordering changed");
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

    #[test]
    fn confident_chain_retains_long_exact_recurrence() {
        let cycle = [10, 11, 12, 13];
        let history: Vec<u32> = cycle.into_iter().cycle().take(24 * cycle.len()).collect();
        let expected: Vec<u32> = cycle.into_iter().cycle().take(15).collect();
        let mut d = SuffixDecodingDrafter::default();

        let proposal = d.draft_confident_chain(&history, 13, 16, 15);

        assert_eq!(proposal.tokens, expected);
        assert!(proposal.evidence.admitted);
        assert_eq!(proposal.evidence.raw_depth, 15);
        assert_eq!(proposal.evidence.confident_depth, 15);
        assert_eq!(proposal.evidence.root_match_len, 4);
        assert_eq!(proposal.evidence.root_support, 23);
        assert_eq!(proposal.evidence.root_branch_count, 23);
        assert!(
            proposal.evidence.terminal_survival_q16 >= SUFFIX_MIN_PREFIX_SURVIVAL_Q16,
            "the four-token recurrence ceiling must survive the full verify depth"
        );
    }

    #[test]
    fn confident_chain_declines_a_single_copied_tail() {
        // The trailing four-token suffix occurred once and has a long copied
        // continuation. Raw suffix depth alone would eagerly verify it.
        let history = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 90, 1, 2, 3, 4];
        let mut d = SuffixDecodingDrafter::default();

        let proposal = d.draft_confident_chain(&history, 4, 16, 8);

        assert!(proposal.evidence.raw_depth >= 3);
        assert_eq!(proposal.evidence.root_support, 1);
        assert_eq!(proposal.evidence.root_branch_count, 1);
        assert_eq!(proposal.evidence.confident_depth, 0);
        assert!(!proposal.evidence.admitted);
        assert!(proposal.tokens.is_empty());
    }

    #[test]
    fn confident_chain_admits_two_agreeing_long_contexts() {
        // Two independent length-4 matches agree for three continuations. The
        // Q16 survival sequence is 2/3, 4/9, 8/27, whose mass exceeds 1.25.
        let history = vec![1, 2, 3, 4, 5, 6, 7, 90, 1, 2, 3, 4, 5, 6, 7, 91, 1, 2, 3, 4];
        let mut d = SuffixDecodingDrafter::default();

        let proposal = d.draft_confident_chain(&history, 4, 4, 3);

        assert_eq!(proposal.tokens, [5, 6, 7]);
        assert!(proposal.evidence.admitted);
        assert_eq!(proposal.evidence.confident_depth, 3);
        assert_eq!(proposal.evidence.root_match_len, 4);
        assert_eq!(proposal.evidence.root_support, 2);
        assert_eq!(proposal.evidence.root_branch_count, 2);
        assert!(proposal.evidence.expected_accepted_q16 >= SUFFIX_MIN_EXPECTED_ACCEPTED_Q16);
    }

    #[test]
    fn short_match_requires_four_observations_and_three_quarters_top_share() {
        assert!(!SuffixDecodingDrafter::root_evidence_admissible(2, 3, 3));
        assert!(!SuffixDecodingDrafter::root_evidence_admissible(2, 4, 2));
        assert!(SuffixDecodingDrafter::root_evidence_admissible(2, 4, 3));
        assert!(SuffixDecodingDrafter::root_evidence_admissible(3, 2, 2));
    }

    #[test]
    fn singleton_survival_hits_then_falls_below_the_exact_q16_floor() {
        let root = SuffixNodeEvidence {
            survival_q16: SUFFIX_CONFIDENCE_Q16_ONE,
            ..SuffixNodeEvidence::default()
        };
        let first = SuffixDecodingDrafter::child_evidence(root, 0, 4, 1, 1);
        let second = SuffixDecodingDrafter::child_evidence(first, 1, 4, 1, 1);
        let third = SuffixDecodingDrafter::child_evidence(second, 2, 4, 1, 1);

        assert_eq!(first.survival_q16, SUFFIX_CONFIDENCE_Q16_ONE / 2);
        assert_eq!(second.survival_q16, SUFFIX_MIN_PREFIX_SURVIVAL_Q16);
        assert_eq!(third.survival_q16, SUFFIX_CONFIDENCE_Q16_ONE / 8);
        assert!(second.survival_q16 >= SUFFIX_MIN_PREFIX_SURVIVAL_Q16);
        assert!(third.survival_q16 < SUFFIX_MIN_PREFIX_SURVIVAL_Q16);
    }

    #[test]
    fn predicted_mass_beats_raw_depth_when_selecting_a_branch() {
        // The weak branch is one token deeper, but its 25% survival contributes
        // only 1.0 token of predicted mass. The supported depth-3 branch has
        // more than 2.0 and must win independently of insertion depth.
        let tree = TokenTree {
            tokens: vec![0, 10, 20, 11, 21, 12, 22, 23],
            parent: vec![-1, 0, 0, 1, 2, 3, 4, 6],
            depth: vec![0, 1, 1, 2, 2, 3, 3, 4],
        };
        let eligible = |survival_q16, expected_accepted_q16| SuffixNodeEvidence {
            root_match_len: 4,
            root_support: 8,
            root_branch_count: 6,
            root_admissible: true,
            survival_q16,
            expected_accepted_q16,
        };
        let scored = ScoredSuffixTree {
            tree,
            evidence: vec![
                SuffixNodeEvidence {
                    survival_q16: SUFFIX_CONFIDENCE_Q16_ONE,
                    ..SuffixNodeEvidence::default()
                },
                eligible(
                    SUFFIX_CONFIDENCE_Q16_ONE * 3 / 4,
                    SUFFIX_CONFIDENCE_Q16_ONE * 3 / 4,
                ),
                eligible(SUFFIX_CONFIDENCE_Q16_ONE / 4, SUFFIX_CONFIDENCE_Q16_ONE / 4),
                eligible(
                    SUFFIX_CONFIDENCE_Q16_ONE * 11 / 16,
                    SUFFIX_CONFIDENCE_Q16_ONE * 23 / 16,
                ),
                eligible(SUFFIX_CONFIDENCE_Q16_ONE / 4, SUFFIX_CONFIDENCE_Q16_ONE / 2),
                eligible(
                    SUFFIX_CONFIDENCE_Q16_ONE * 10 / 16,
                    SUFFIX_CONFIDENCE_Q16_ONE * 33 / 16,
                ),
                eligible(
                    SUFFIX_CONFIDENCE_Q16_ONE / 4,
                    SUFFIX_CONFIDENCE_Q16_ONE * 3 / 4,
                ),
                eligible(SUFFIX_CONFIDENCE_Q16_ONE / 4, SUFFIX_CONFIDENCE_Q16_ONE),
            ],
        };

        assert_eq!(SuffixDecodingDrafter::best_confident_node(&scored), 5);
        assert_eq!(scored.tree.path_to(5), [0, 1, 3, 5]);
        assert_eq!(scored.tree.max_depth(), 4);
    }

    #[test]
    fn admitted_chain_is_a_path_in_the_unchanged_branching_tree() {
        let history: Vec<u32> = [4, 5, 6, 7].into_iter().cycle().take(12 * 4).collect();
        let mut tree_drafter = SuffixDecodingDrafter::default();
        let mut chain_drafter = tree_drafter.clone();
        let tree = tree_drafter.draft_tree(&history, 7, 9, 8);
        let proposal = chain_drafter.draft_confident_chain(&history, 7, 9, 8);

        assert!(proposal.evidence.admitted);
        assert!((1..tree.nodes())
            .map(|node| {
                tree.path_to(node)
                    .into_iter()
                    .skip(1)
                    .map(|node| tree.tokens[node])
                    .collect::<Vec<_>>()
            })
            .any(|path| path == proposal.tokens));
    }
}
