//! CPU tree-speculation substrate (Lane A).
//!
//! Today's lossless speculation drafts a single linear chain of tokens and
//! verifies it in one batched forward (see
//! [`crate::inference::speculative`] and the `verify_drafts_gpu` loop in
//! `inference.rs`). A *tree* of drafts generalizes that: several candidate
//! continuations share a prefix, so one batched forward can verify multiple
//! branches and accept the longest matching path. This module is the CPU-side
//! seam a future GPU tree-verify kernel and model-free drafters plug into. It
//! is all-Rust, GPU-free, and unit-testable without a GPU.
//!
//! Losslessness is preserved by construction: every emitted token is the
//! target model's own greedy argmax along the accepted path
//! ([`TokenTree::accept_longest_path`]). The degenerate single-branch tree
//! ([`TokenTree::linear`]) reproduces today's linear chain exactly and serves
//! as the oracle the tree path is checked against.

/// Maximum nodes in a draft tree (including the root anchor). The verify GEMM
/// batches one row per node; on the GPU `launch_gemm_batched`
/// (`cuda_resident.rs` ~1255) clamps `warps_per_block` by the shared-mem
/// budget, and for big-FFN models the per-block occupancy collapses once the
/// batch width N grows past ~16-20, so widening the tree past this stops
/// paying for itself. Kept SEPARATE from `cuda_resident::MAX_VERIFY_K` (= 8,
/// the linear-chain verify cap): a tree of N nodes has at most depth N-1 but
/// usually much less, so the node cap and the depth cap are different limits.
/// Cap at 16 (hard ceiling 20).
pub const TREE_MAX_NODES: usize = 16;

/// A draft token tree.
///
/// Node 0 is the root = the anchor: the last committed token, the one the
/// target is guaranteed to produce a +1 bonus prediction for. Nodes are stored
/// in BFS order so a parent always precedes its children (`parent[i] < i` for
/// `i > 0`). `parent[0] == -1` and `depth[0] == 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenTree {
    /// Token id at each node. `tokens[0]` is the anchor (last committed token).
    pub tokens: Vec<u32>,
    /// Parent index per node (`-1` for the root). BFS order ⇒ `parent[i] < i`.
    pub parent: Vec<i32>,
    /// Depth per node (root = 0).
    pub depth: Vec<u16>,
}

impl TokenTree {
    /// A single-branch (linear) tree: the EXACT reproduction of today's draft
    /// chain. `anchor` is the last committed token; `drafts` are the proposed
    /// continuation. This is the oracle the tree path is validated against.
    ///
    /// Layout: node 0 = anchor, node i = `drafts[i-1]` with parent `i-1`,
    /// depth `i`. A chain of `drafts.len()` edges.
    pub fn linear(anchor: u32, drafts: &[u32]) -> Self {
        let n = drafts.len() + 1;
        let mut tokens = Vec::with_capacity(n);
        let mut parent = Vec::with_capacity(n);
        let mut depth = Vec::with_capacity(n);
        tokens.push(anchor);
        parent.push(-1);
        depth.push(0);
        for (i, &tok) in drafts.iter().enumerate() {
            tokens.push(tok);
            parent.push(i as i32); // previous node
            depth.push((i + 1) as u16);
        }
        Self {
            tokens,
            parent,
            depth,
        }
    }

    /// Number of nodes (including the root anchor).
    pub fn nodes(&self) -> usize {
        self.tokens.len()
    }

    /// Maximum depth of any node (0 for an anchor-only tree).
    pub fn max_depth(&self) -> usize {
        self.depth.iter().copied().max().unwrap_or(0) as usize
    }

    /// The node indices from the root to `leaf` inclusive (root first).
    pub fn path_to(&self, leaf: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut cur = leaf as i32;
        while cur >= 0 {
            path.push(cur as usize);
            cur = self.parent[cur as usize];
        }
        path.reverse();
        path
    }

    // --- GPU-interface builders ---------------------------------------------
    //
    // Host data the future tree-verify kernel consumes. Built and unit-tested
    // now so the kernel has a validated contract to read against.

    /// Per-node depth as `i32` (root = 0). The kernel uses depth to index the
    /// RoPE tables for each node's position (`base + depth`).
    pub fn node_depth(&self) -> Vec<i32> {
        self.depth.iter().map(|&d| d as i32).collect()
    }

    /// Per-node KV slot relative to `base` (the current KV position). Every
    /// node gets a UNIQUE slot = `base + BFS index`, so the batched verify can
    /// scatter each node's K/V into its own cache row without collision. The
    /// per-node attention mask (which earlier slots a node may read) is the
    /// ancestor bitset, not the slot itself.
    pub fn node_kvslot(&self, base: usize) -> Vec<i32> {
        (0..self.nodes()).map(|i| (base + i) as i32).collect()
    }

    /// Ancestor bitset: `ceil(N/32)` u32 words per node. For node `i`, bit `j`
    /// (word `j/32`, bit `j%32`) is set iff node `j` is an ancestor of `i` or
    /// `i` itself. This is the causal attention mask for tree verify — a node
    /// attends only to the tokens on its own root-to-node path. Returns the
    /// flat bitset and the stride (words per node).
    pub fn ancestor_bitset(&self) -> (Vec<u32>, usize) {
        let n = self.nodes();
        let words = n.div_ceil(32);
        let mut bits = vec![0u32; n * words];
        for i in 0..n {
            // Walk i and all its ancestors, setting the corresponding bits.
            let mut cur = i as i32;
            while cur >= 0 {
                let c = cur as usize;
                bits[i * words + c / 32] |= 1u32 << (c % 32);
                cur = self.parent[c];
            }
        }
        (bits, words)
    }

    /// Compact root-to-row paths for a packed-forest verifier.
    ///
    /// Unlike the generic tree-attention lane, a packed-forest verifier can keep each layer's
    /// node K/V in a small `[node][kv]` scratch buffer.  Immediately before running the ordinary
    /// lossless linear attention kernel for row `r`, it copies that row's CSR slice from
    /// `path_rows` into the canonical contiguous future KV slots.
    /// Thus the proven linear kernel sees exactly `[committed prefix || root-to-r path]`, while
    /// every large target projection still batches all forest rows and reads target weights
    /// once.  `u32` matches Metal buffer/scalar indexing and supports trees larger than the
    /// current production cap without changing this ABI.
    pub fn packed_forest_plan(&self) -> PackedForestPlan {
        let mut path_offsets = Vec::with_capacity(self.nodes() + 1);
        let mut path_rows = Vec::new();
        path_offsets.push(0);
        for row in 0..self.nodes() {
            path_rows.extend(self.path_to(row).into_iter().map(|node| node as u32));
            path_offsets.push(path_rows.len() as u32);
        }
        PackedForestPlan {
            path_offsets,
            path_rows,
        }
    }

    /// Lossless greedy-exact tree accept.
    ///
    /// `predicted[i]` is the target model's greedy argmax for node `i` (i.e.
    /// the token that should follow node `i` along its path). The root is
    /// always accepted: `predicted[0]` is the guaranteed +1 bonus. At an
    /// accepted node `c` we look for the child `ch` whose token equals
    /// `predicted[c]` (the model's own next token); if one exists we accept and
    /// emit it and recurse from `ch`. When no child matches we stop and emit
    /// `predicted[c]` as the final bonus token (the model's greedy next token,
    /// which no draft branch covered).
    ///
    /// Returns `(emitted_tokens, leaf_node)` where `leaf_node` is the last
    /// accepted DRAFT node (its KV is committed by the caller advancing the
    /// position). For an anchor-only tree the leaf is the root (0).
    ///
    /// Every emitted token is `predicted[c]` for some accepted node `c`, i.e.
    /// the target's own greedy choice — losslessly identical to vanilla decode.
    pub fn accept_longest_path(&self, predicted: &[u32]) -> (Vec<u32>, usize) {
        debug_assert_eq!(
            predicted.len(),
            self.nodes(),
            "predicted must have one entry per tree node"
        );
        let mut emitted = Vec::new();
        let mut c = 0usize; // root always accepted
        loop {
            // predicted[c] is the model's greedy next token after node c.
            let next = predicted[c];
            // Find a child of c whose token == next. Children of c are the
            // nodes whose parent is c; BFS order is not required here.
            let mut matched: Option<usize> = None;
            for ch in (c + 1)..self.nodes() {
                if self.parent[ch] == c as i32 && self.tokens[ch] == next {
                    matched = Some(ch);
                    break;
                }
            }
            match matched {
                Some(ch) => {
                    // Accept and emit the model's token (== the child's token),
                    // then continue verifying from the child.
                    emitted.push(next);
                    c = ch;
                }
                None => {
                    // No draft branch covered the model's next token: emit it
                    // as the final bonus and stop. `c` is the last accepted
                    // draft node (root if nothing was accepted).
                    emitted.push(next);
                    return (emitted, c);
                }
            }
        }
    }
}

/// CSR-style ancestry rows consumed by the packed-forest materialization kernel described in
/// [`TokenTree::packed_forest_plan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedForestPlan {
    /// `nodes + 1` offsets into `path_rows`.
    pub path_offsets: Vec<u32>,
    /// Concatenated root-to-node verifier row ids, including each row itself.
    pub path_rows: Vec<u32>,
}

impl PackedForestPlan {
    pub fn path(&self, row: usize) -> Option<&[u32]> {
        let start = usize::try_from(*self.path_offsets.get(row)?).ok()?;
        let end = usize::try_from(*self.path_offsets.get(row + 1)?).ok()?;
        self.path_rows.get(start..end)
    }
}

/// One conditional candidate produced while expanding a learned draft lattice.
///
/// `log_probability` is the draft model's normalized log probability for this token
/// conditioned on `parent`'s path.  Keeping the score normalized (rather than accepting a
/// raw logit) gives the lattice a useful invariant: cumulative path scores never increase,
/// so a global top-N rerank is parent-closed when ties prefer shallower nodes.  That is the
/// small but important property used by EAGLE-2-style dynamic trees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftCandidateScore {
    pub token: u32,
    pub log_probability: f32,
}

/// Convert retained top-k raw logits into normalized candidate scores using the log-sum-exp
/// over the **entire** draft vocabulary.
///
/// The caller must supply `full_vocab_logsumexp`, not a normalizer computed from `top_logits`.
/// Normalizing only the retained candidates would redistribute omitted probability mass and
/// systematically over-rank wide branches.  Token/logit input order is preserved so the
/// learned head's deterministic tie-break remains available to later reranking.
pub fn normalize_draft_top_logits(
    top_logits: &[(u32, f32)],
    full_vocab_logsumexp: f32,
) -> Result<Vec<DraftCandidateScore>, String> {
    if !full_vocab_logsumexp.is_finite() {
        return Err(format!(
            "draft full-vocabulary logsumexp must be finite, got {full_vocab_logsumexp}"
        ));
    }
    top_logits
        .iter()
        .enumerate()
        .map(|(slot, &(token, logit))| {
            if !logit.is_finite() {
                return Err(format!("draft top-k logit {slot} is not finite: {logit}"));
            }
            let log_probability = logit - full_vocab_logsumexp;
            if log_probability > 1.0e-5 {
                return Err(format!(
                    "draft top-k logit {slot} exceeds the full-vocabulary logsumexp"
                ));
            }
            Ok(DraftCandidateScore {
                token,
                // A tiny positive value is possible if the all-vocabulary reduction and the
                // retained scalar were rounded along different f32 paths.  Probability one is
                // the conservative normalized ceiling.
                log_probability: log_probability.min(0.0),
            })
        })
        .collect()
}

/// A node in the expansion-time candidate lattice.  Indices are stable and parents always
/// precede children; the final verifier layout is produced separately by
/// [`DynamicDraftLattice::rerank_connected`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredDraftNode {
    pub token: u32,
    pub parent: Option<usize>,
    pub depth: u16,
    /// Sum of conditional log probabilities from the root to this node.  The root is 0.
    pub cumulative_log_probability: f32,
}

/// CPU-side EAGLE-2-style dynamic draft lattice.
///
/// Expansion and verification are deliberately separate.  A learned head may grow a wider
/// temporary lattice, then [`rerank_connected`](Self::rerank_connected) keeps only the nodes
/// with the greatest estimated chance of surviving target verification.  The result is an
/// ordinary [`TokenTree`], so target-authoritative [`TokenTree::accept_longest_path`] remains
/// the sole acceptance rule.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicDraftLattice {
    nodes: Vec<ScoredDraftNode>,
}

/// Connected verifier-ready subset selected from a [`DynamicDraftLattice`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredTokenTree {
    pub tree: TokenTree,
    /// Cumulative log probability in the same BFS row order as `tree`.
    pub cumulative_log_probability: Vec<f32>,
    /// Stable expansion-lattice index for every verifier row.  Useful for gathering the
    /// corresponding recurrent state/KV row without relying on token ids being unique.
    pub source_node: Vec<usize>,
}

impl ScoredTokenTree {
    /// Expected target-emitted tokens per verify round under calibrated draft confidence.
    ///
    /// Every selected non-root node contributes one token iff its entire path is accepted,
    /// whose estimated probability is its cumulative path probability.  The target's final
    /// authoritative bonus token contributes one unconditionally.
    pub fn estimated_emitted_tokens(&self) -> f64 {
        1.0 + self
            .cumulative_log_probability
            .iter()
            .skip(1)
            .map(|&score| f64::from(score).exp())
            .sum::<f64>()
    }
}

impl DynamicDraftLattice {
    pub fn new(anchor: u32) -> Self {
        Self {
            nodes: vec![ScoredDraftNode {
                token: anchor,
                parent: None,
                depth: 0,
                cumulative_log_probability: 0.0,
            }],
        }
    }

    pub fn nodes(&self) -> &[ScoredDraftNode] {
        &self.nodes
    }

    /// Append one normalized top-k expansion under `parent`, returning stable node indices.
    ///
    /// Sibling probabilities must sum to at most one (a top-k subset may sum to less).  A
    /// repeated token under the same parent is rejected because it would spend a verifier row
    /// without adding a distinct greedy path.  Errors leave the lattice unchanged.
    pub fn expand(
        &mut self,
        parent: usize,
        candidates: &[DraftCandidateScore],
    ) -> Result<Vec<usize>, String> {
        let Some(parent_node) = self.nodes.get(parent) else {
            return Err(format!("draft lattice parent {parent} is out of range"));
        };
        let depth = parent_node
            .depth
            .checked_add(1)
            .ok_or_else(|| "draft lattice depth exceeds u16".to_string())?;
        let parent_score = parent_node.cumulative_log_probability;
        let mut probability_sum = 0.0f64;
        for (slot, candidate) in candidates.iter().enumerate() {
            if !candidate.log_probability.is_finite() || candidate.log_probability > 0.0 {
                return Err(format!(
                    "draft candidate {slot} has invalid normalized log probability {}",
                    candidate.log_probability
                ));
            }
            probability_sum += f64::from(candidate.log_probability).exp();
            if candidates[..slot]
                .iter()
                .any(|earlier| earlier.token == candidate.token)
                || self
                    .nodes
                    .iter()
                    .any(|node| node.parent == Some(parent) && node.token == candidate.token)
            {
                return Err(format!(
                    "draft lattice parent {parent} has duplicate child token {}",
                    candidate.token
                ));
            }
        }
        // Accommodate the final few ulps of a correctly normalized f32 softmax, but reject
        // raw-logit inputs and otherwise malformed sibling distributions.
        if probability_sum > 1.0 + 1.0e-5 {
            return Err(format!(
                "draft candidate probabilities sum to {probability_sum}, above one"
            ));
        }

        let start = self.nodes.len();
        self.nodes
            .extend(candidates.iter().map(|candidate| ScoredDraftNode {
                token: candidate.token,
                parent: Some(parent),
                depth,
                cumulative_log_probability: parent_score + candidate.log_probability,
            }));
        Ok((start..self.nodes.len()).collect())
    }

    /// Select the globally strongest `max_nodes` rows while preserving parent closure, then
    /// lay them out in deterministic BFS order for target verification.
    ///
    /// Ranking is cumulative probability descending, depth ascending, then stable expansion
    /// index ascending.  Because every conditional log probability is <= 0, a parent always
    /// ranks before its child (including exact-probability-one ties); therefore truncating this
    /// order cannot orphan a selected node.
    pub fn rerank_connected(
        &self,
        max_nodes: usize,
        max_depth: usize,
    ) -> Result<ScoredTokenTree, String> {
        if max_nodes == 0 {
            return Err("a verifier tree must retain at least its root".to_string());
        }
        let mut selected: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| (usize::from(node.depth) <= max_depth).then_some(index))
            .collect();
        selected.sort_by(|&left, &right| {
            self.nodes[right]
                .cumulative_log_probability
                .total_cmp(&self.nodes[left].cumulative_log_probability)
                .then_with(|| self.nodes[left].depth.cmp(&self.nodes[right].depth))
                .then_with(|| left.cmp(&right))
        });
        selected.truncate(max_nodes);
        // The root is score 0/depth 0 and every child is <= it, so it must survive any
        // non-empty truncation.  Keep this a checked contract rather than a debug-only claim.
        if !selected.contains(&0) {
            return Err("dynamic draft rerank dropped the root".to_string());
        }
        for &node in &selected {
            if let Some(parent) = self.nodes[node].parent {
                if !selected.contains(&parent) {
                    return Err(format!(
                        "dynamic draft rerank selected node {node} without parent {parent}"
                    ));
                }
            }
        }

        // Matrix rows may be ranked in any order, but the shared tree substrate promises BFS.
        // Restore level order while retaining stable expansion order inside each level.
        selected.sort_by_key(|&index| (self.nodes[index].depth, index));
        let mut remap = vec![usize::MAX; self.nodes.len()];
        for (row, &source) in selected.iter().enumerate() {
            remap[source] = row;
        }

        let mut tokens = Vec::with_capacity(selected.len());
        let mut parent = Vec::with_capacity(selected.len());
        let mut depth = Vec::with_capacity(selected.len());
        let mut cumulative_log_probability = Vec::with_capacity(selected.len());
        for &source in &selected {
            let node = &self.nodes[source];
            tokens.push(node.token);
            parent.push(match node.parent {
                None => -1,
                Some(source_parent) => i32::try_from(remap[source_parent])
                    .map_err(|_| "verifier tree row exceeds i32".to_string())?,
            });
            depth.push(node.depth);
            cumulative_log_probability.push(node.cumulative_log_probability);
        }
        Ok(ScoredTokenTree {
            tree: TokenTree {
                tokens,
                parent,
                depth,
            },
            cumulative_log_probability,
            source_node: selected,
        })
    }
}

/// A drafter that proposes a tree of candidate continuations.
///
/// `history` is the full token sequence so far (prompt + generated, including
/// the trailing committed token). `anchor` is that trailing token (the tree
/// root). `max_nodes` and `max_depth` bound the returned tree. Implementations
/// must return a tree with node 0's token == `anchor`.
pub trait TreeDrafter {
    fn draft_tree(
        &mut self,
        history: &[u32],
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> TokenTree;
}

/// Adapts the existing [`NGramDrafter`](crate::inference::speculative::NGramDrafter)
/// as a degenerate single-branch tree drafter. The tree is exactly
/// [`TokenTree::linear`] of the n-gram chain, so wrapping it changes no
/// behavior — the linear accept and the tree accept coincide (proven in
/// tests).
pub struct NGramTreeDrafter {
    pub inner: crate::inference::speculative::NGramDrafter,
}

impl NGramTreeDrafter {
    pub fn new(inner: crate::inference::speculative::NGramDrafter) -> Self {
        Self { inner }
    }
}

impl TreeDrafter for NGramTreeDrafter {
    fn draft_tree(
        &mut self,
        history: &[u32],
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> TokenTree {
        // A single chain consumes (chain_len) nodes plus the root; bound by
        // both the node budget and the depth budget.
        let max_drafts = max_nodes.saturating_sub(1).min(max_depth);
        let drafts = self.inner.draft(history, max_drafts);
        TokenTree::linear(anchor, &drafts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::speculative::NGramDrafter;

    /// Build a small hand tree:
    ///        0 (anchor=10)
    ///       / \
    ///      1   2        (tokens 11, 12)
    ///     /   / \
    ///    3   4   5      (tokens 13, 14, 15)
    fn branchy_tree() -> TokenTree {
        TokenTree {
            tokens: vec![10, 11, 12, 13, 14, 15],
            parent: vec![-1, 0, 0, 1, 2, 2],
            depth: vec![0, 1, 1, 2, 2, 2],
        }
    }

    #[test]
    fn linear_layout_matches_chain() {
        let t = TokenTree::linear(10, &[11, 12, 13]);
        assert_eq!(t.tokens, vec![10, 11, 12, 13]);
        assert_eq!(t.parent, vec![-1, 0, 1, 2]);
        assert_eq!(t.depth, vec![0, 1, 2, 3]);
        assert_eq!(t.nodes(), 4);
        assert_eq!(t.max_depth(), 3);
        assert_eq!(t.path_to(3), vec![0, 1, 2, 3]);
    }

    #[test]
    fn linear_anchor_only() {
        let t = TokenTree::linear(7, &[]);
        assert_eq!(t.tokens, vec![7]);
        assert_eq!(t.parent, vec![-1]);
        assert_eq!(t.depth, vec![0]);
        assert_eq!(t.max_depth(), 0);
    }

    #[test]
    fn parent_before_child_bfs_invariant() {
        let t = branchy_tree();
        for i in 1..t.nodes() {
            assert!(t.parent[i] < i as i32, "parent[{i}] must precede child");
        }
        assert_eq!(t.parent[0], -1);
        assert_eq!(t.depth[0], 0);
    }

    #[test]
    fn path_to_walks_to_root() {
        let t = branchy_tree();
        assert_eq!(t.path_to(3), vec![0, 1, 3]);
        assert_eq!(t.path_to(5), vec![0, 2, 5]);
        assert_eq!(t.path_to(0), vec![0]);
    }

    #[test]
    fn node_depth_matches() {
        let t = branchy_tree();
        assert_eq!(t.node_depth(), vec![0, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn node_kvslot_unique_from_base() {
        let t = branchy_tree();
        assert_eq!(t.node_kvslot(100), vec![100, 101, 102, 103, 104, 105]);
        // Slots are unique (no two nodes collide).
        let slots = t.node_kvslot(0);
        let mut sorted = slots.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(slots.len(), sorted.len());
    }

    #[test]
    fn ancestor_bitset_marks_path() {
        let t = branchy_tree();
        let (bits, words) = t.ancestor_bitset();
        assert_eq!(words, 1); // 6 nodes ⇒ ceil(6/32) = 1 word
        let isset = |node: usize, anc: usize| -> bool {
            (bits[node * words + anc / 32] >> (anc % 32)) & 1 == 1
        };
        // Node 3 (path 0->1->3): ancestors-or-self = {0,1,3}.
        assert!(isset(3, 0) && isset(3, 1) && isset(3, 3));
        assert!(!isset(3, 2) && !isset(3, 4) && !isset(3, 5));
        // Node 5 (path 0->2->5): {0,2,5}.
        assert!(isset(5, 0) && isset(5, 2) && isset(5, 5));
        assert!(!isset(5, 1) && !isset(5, 3) && !isset(5, 4));
        // Root: only itself.
        assert!(isset(0, 0));
        assert!(!isset(0, 1));
    }

    #[test]
    fn ancestor_bitset_multiword_stride() {
        // 33-node linear chain ⇒ stride 2 words.
        let drafts: Vec<u32> = (1..=32).collect();
        let t = TokenTree::linear(0, &drafts);
        let (bits, words) = t.ancestor_bitset();
        assert_eq!(t.nodes(), 33);
        assert_eq!(words, 2);
        // Node 32 is the leaf of the chain: every node 0..=32 is an ancestor.
        let isset = |node: usize, anc: usize| -> bool {
            (bits[node * words + anc / 32] >> (anc % 32)) & 1 == 1
        };
        for anc in 0..=32 {
            assert!(isset(32, anc), "node 32 should have ancestor {anc}");
        }
    }

    #[test]
    fn packed_forest_plan_lists_each_root_to_row_path() {
        let t = branchy_tree();
        let packed = t.packed_forest_plan();
        assert_eq!(packed.path_offsets, vec![0, 1, 3, 5, 8, 11, 14]);
        assert_eq!(packed.path(0), Some(&[0][..]));
        assert_eq!(packed.path(1), Some(&[0, 1][..]));
        assert_eq!(packed.path(2), Some(&[0, 2][..]));
        assert_eq!(packed.path(3), Some(&[0, 1, 3][..]));
        assert_eq!(packed.path(4), Some(&[0, 2, 4][..]));
        assert_eq!(packed.path(5), Some(&[0, 2, 5][..]));
        assert_eq!(packed.path(6), None);
    }

    // --- accept_longest_path == linear accept (the oracle) ------------------

    /// Today's linear accept, reproduced verbatim from the verify_drafts_gpu /
    /// assert_speculative_matches_vanilla loop: accept the longest prefix of
    /// `drafts` that the model confirms, plus the bonus at the first mismatch.
    /// Returns the emitted tokens (predicted[0..=accepted]).
    fn linear_accept_reference(drafts: &[u32], predicted: &[u32]) -> Vec<u32> {
        let mut accepted = vec![predicted[0]];
        let mut j = 0usize;
        while j < drafts.len() && drafts[j] == predicted[j] {
            accepted.push(predicted[j + 1]);
            j += 1;
        }
        accepted
    }

    #[test]
    fn accept_longest_path_reproduces_linear_accept() {
        // Exhaustively check many (drafts, predicted) pairs: the tree accept on
        // a linear() tree must emit exactly what the linear accept emits.
        let cases: &[(&[u32], &[u32])] = &[
            // full accept
            (&[1, 2, 3], &[9, 1, 2, 3]),
            // accept first two, mismatch at third
            (&[1, 2, 3], &[9, 1, 2, 99]),
            // accept first, mismatch at second
            (&[1, 2, 3], &[9, 1, 88, 0]),
            // reject all drafts (only bonus emitted)
            (&[1, 2, 3], &[9, 88, 0, 0]),
            // no drafts: bonus only
            (&[], &[42]),
            // single draft accepted
            (&[5], &[9, 5]),
            // single draft rejected
            (&[5], &[9, 6]),
        ];
        for (drafts, predicted) in cases {
            let tree = TokenTree::linear(0, drafts);
            let (emitted, _leaf) = tree.accept_longest_path(predicted);
            let reference = linear_accept_reference(drafts, predicted);
            assert_eq!(
                emitted, reference,
                "tree accept must equal linear accept for drafts={drafts:?} predicted={predicted:?}"
            );
        }
    }

    #[test]
    fn accept_longest_path_leaf_tracks_accepted_count() {
        // drafts=[1,2,3] live at nodes 1,2,3. predicted[c] is the model's
        // argmax after node c. To accept drafts[0]=1 then drafts[1]=2 then
        // mismatch: predicted = [1, 2, 99, _]. Emitted = the model's tokens
        // along the accepted path plus the final bonus: [1, 2, 99].
        let tree = TokenTree::linear(0, &[1, 2, 3]);
        let (emitted, leaf) = tree.accept_longest_path(&[1, 2, 99, 0]);
        assert_eq!(emitted, vec![1, 2, 99]); // 2 accepted + bonus
        assert_eq!(leaf, 2); // node index of the 2nd accepted draft
                             // reject at root ⇒ leaf is root, only bonus emitted.
        let (emitted2, leaf2) = tree.accept_longest_path(&[88, 0, 0, 0]);
        assert_eq!(emitted2, vec![88]);
        assert_eq!(leaf2, 0);
    }

    #[test]
    fn accept_longest_path_picks_matching_branch() {
        //        0 (anchor=10)
        //       / \
        //      1   2     (tokens 11, 12)
        //     /   / \
        //    3   4   5   (tokens 13, 14, 15)
        let t = branchy_tree();
        // predicted: at root the model wants 12 (node 2's token), at node 2 it
        // wants 15 (node 5's token), at node 5 it wants 100 (no child).
        // predicted indexed by node: [12, _, 15, _, _, 100]
        let predicted = vec![12, 0, 15, 0, 0, 100];
        let (emitted, leaf) = t.accept_longest_path(&predicted);
        // root accepted -> emit 12 (matches child 2) -> emit 15 (matches child 5)
        // -> no child -> emit 100 bonus.
        assert_eq!(emitted, vec![12, 15, 100]);
        assert_eq!(leaf, 5);
    }

    #[test]
    fn accept_longest_path_stops_when_no_branch_matches() {
        let t = branchy_tree();
        // Model wants 99 at root: no child has token 99, emit bonus, stop.
        let predicted = vec![99, 0, 0, 0, 0, 0];
        let (emitted, leaf) = t.accept_longest_path(&predicted);
        assert_eq!(emitted, vec![99]);
        assert_eq!(leaf, 0);
    }

    #[test]
    fn dynamic_lattice_reranks_by_global_path_probability_and_stays_connected() {
        let mut lattice = DynamicDraftLattice::new(10);
        let first = lattice
            .expand(
                0,
                &[
                    DraftCandidateScore {
                        token: 11,
                        log_probability: 0.6f32.ln(),
                    },
                    DraftCandidateScore {
                        token: 12,
                        log_probability: 0.4f32.ln(),
                    },
                ],
            )
            .unwrap();
        let under_11 = lattice
            .expand(
                first[0],
                &[
                    DraftCandidateScore {
                        token: 13,
                        log_probability: 0.5f32.ln(), // global 0.30
                    },
                    DraftCandidateScore {
                        token: 14,
                        log_probability: 0.4f32.ln(), // global 0.24; pruned
                    },
                ],
            )
            .unwrap();
        let under_12 = lattice
            .expand(
                first[1],
                &[DraftCandidateScore {
                    token: 15,
                    log_probability: 0.9f32.ln(), // global 0.36
                }],
            )
            .unwrap();

        // Root + both depth-1 nodes + the two strongest depth-2 nodes.  The global rerank
        // prefers token 15 (0.36) over token 13 (0.30), but verifier rows are restored to BFS
        // expansion order and remain parent-before-child.
        let plan = lattice.rerank_connected(5, 8).unwrap();
        assert_eq!(plan.tree.tokens, vec![10, 11, 12, 13, 15]);
        assert_eq!(plan.tree.parent, vec![-1, 0, 0, 1, 2]);
        assert_eq!(plan.tree.depth, vec![0, 1, 1, 2, 2]);
        assert_eq!(
            plan.source_node,
            vec![0, first[0], first[1], under_11[0], under_12[0]]
        );
        for row in 1..plan.tree.nodes() {
            assert!(plan.tree.parent[row] < row as i32);
        }
        let expected = 1.0 + 0.6 + 0.4 + 0.30 + 0.36;
        assert!((plan.estimated_emitted_tokens() - expected).abs() < 1.0e-6);
    }

    #[test]
    fn full_vocab_normalizer_preserves_probability_mass_omitted_by_top_k() {
        // Full distribution is [0.50, 0.30, 0.20], while only its top two entries survive.
        let normalized = normalize_draft_top_logits(
            &[(11, 0.5f32.ln()), (12, 0.3f32.ln())],
            0.0, // ln(0.50 + 0.30 + 0.20)
        )
        .unwrap();
        assert_eq!(normalized[0].token, 11);
        assert_eq!(normalized[1].token, 12);
        let retained_mass: f32 = normalized
            .iter()
            .map(|candidate| candidate.log_probability.exp())
            .sum();
        assert!((retained_mass - 0.8).abs() < 1.0e-6);
        assert!(retained_mass < 1.0, "top-k must not be renormalized");
    }

    #[test]
    fn dynamic_lattice_probability_one_ties_still_keep_parent_before_child() {
        let mut lattice = DynamicDraftLattice::new(1);
        let child = lattice
            .expand(
                0,
                &[DraftCandidateScore {
                    token: 2,
                    log_probability: 0.0,
                }],
            )
            .unwrap()[0];
        lattice
            .expand(
                child,
                &[DraftCandidateScore {
                    token: 3,
                    log_probability: 0.0,
                }],
            )
            .unwrap();

        let root_only = lattice.rerank_connected(1, 8).unwrap();
        assert_eq!(root_only.tree.tokens, vec![1]);
        let root_and_child = lattice.rerank_connected(2, 8).unwrap();
        assert_eq!(root_and_child.tree.tokens, vec![1, 2]);
        assert_eq!(root_and_child.tree.parent, vec![-1, 0]);
    }

    #[test]
    fn dynamic_lattice_rejects_unnormalized_or_duplicate_siblings_without_mutation() {
        let mut lattice = DynamicDraftLattice::new(1);
        let before = lattice.clone();
        assert!(lattice
            .expand(
                0,
                &[
                    DraftCandidateScore {
                        token: 2,
                        log_probability: 0.8f32.ln(),
                    },
                    DraftCandidateScore {
                        token: 3,
                        log_probability: 0.8f32.ln(),
                    },
                ],
            )
            .is_err());
        assert_eq!(lattice, before);

        assert!(lattice
            .expand(
                0,
                &[
                    DraftCandidateScore {
                        token: 2,
                        log_probability: 0.5f32.ln(),
                    },
                    DraftCandidateScore {
                        token: 2,
                        log_probability: 0.4f32.ln(),
                    },
                ],
            )
            .is_err());
        assert_eq!(lattice, before);
    }

    #[test]
    fn ngram_tree_drafter_is_degenerate_linear() {
        // Wrapping NGramDrafter as a tree drafter yields exactly linear() of
        // the same n-gram chain — zero behavior change.
        let mut drafter = NGramTreeDrafter::new(NGramDrafter::default());
        let history = vec![1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3, 4];
        let anchor = *history.last().unwrap();
        let max_nodes = TREE_MAX_NODES;
        let max_depth = 8;
        let tree = drafter.draft_tree(&history, anchor, max_nodes, max_depth);
        // The wrapper draws max_nodes-1 capped by max_depth drafts.
        let max_drafts = (max_nodes - 1).min(max_depth);
        let chain = NGramDrafter::default().draft(&history, max_drafts);
        let expected = TokenTree::linear(anchor, &chain);
        assert_eq!(tree, expected);
        // The tree is a single chain (no branching).
        for i in 1..tree.nodes() {
            assert_eq!(tree.parent[i], (i - 1) as i32);
        }
    }
}
