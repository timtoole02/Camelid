//! Radix-Tree Multi-Turn Prompt Prefix Caching.
//!
//! Replaces single-slot or naive prefix caching with a dynamic prefix tree (compact trie)
//! of token sequences backed by shared physical KV blocks from `PagedKvBlockPool`.
//!
//! Multi-turn chat sessions and shared system prompts frequently share long common prefixes
//! (e.g. system instructions, tool definitions, earlier conversation turns). By indexing
//! KV cache blocks in a Radix Tree:
//! 1. Prompt prefill can match the longest existing prefix and immediately resume from it
//!    without recomputing any attention or feed-forward operations.
//! 2. Shared physical blocks are reference-counted, saving significant memory.
//! 3. LRU eviction guarantees bounded cache size under memory pressure.

use std::collections::HashMap;
use std::time::Instant;

use crate::inference::paged_kv::{PagedKvBlockPool, PhysicalBlockId, KV_BLOCK_TOKENS};
use crate::Result;

/// A node in the Radix Tree representing a sequence of tokens and associated physical blocks.
#[derive(Debug, Clone)]
pub struct RadixNode {
    pub id: usize,
    pub parent: Option<usize>,
    /// Child transitions keyed by the FIRST token of the child's edge sequence.
    pub children: HashMap<u32, usize>,
    /// Token sequence on this edge. Always a whole number of KV blocks — see the
    /// block-alignment invariant on [`RadixPrefixCache`].
    pub tokens: Vec<u32>,
    /// Physical blocks holding the KV activations for these tokens. Exactly
    /// `tokens.len() / KV_BLOCK_TOKENS` entries.
    pub blocks: Vec<PhysicalBlockId>,
    /// Last access timestamp for LRU ordering.
    pub last_accessed: Instant,
}

impl RadixNode {
    pub fn new(
        id: usize,
        parent: Option<usize>,
        tokens: Vec<u32>,
        blocks: Vec<PhysicalBlockId>,
    ) -> Self {
        debug_assert_eq!(
            tokens.len() / KV_BLOCK_TOKENS,
            blocks.len(),
            "radix node must hold exactly one block per {KV_BLOCK_TOKENS} tokens"
        );
        Self {
            id,
            parent,
            children: HashMap::new(),
            tokens,
            blocks,
            last_accessed: Instant::now(),
        }
    }
}

/// Statistics for prefix cache performance monitoring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RadixCacheStats {
    pub queries: u64,
    pub hits: u64,
    pub tokens_queried: u64,
    pub tokens_saved: u64,
    pub evicted_nodes: u64,
}

/// Dynamic Radix-Tree Prefix Cache for multi-turn conversations.
///
/// # Block-alignment invariant
///
/// KV lives in fixed [`KV_BLOCK_TOKENS`]-token physical blocks, so a prefix is only
/// reusable in whole blocks: a 37-token match backed by 2 full blocks can reuse 32
/// tokens, not 37. Every node therefore carries a whole number of blocks, and
/// [`Self::insert_sequence`] truncates its input to a block boundary. That makes
/// `blocks.len() * KV_BLOCK_TOKENS == tokens.len()` an invariant which
/// [`Self::match_longest_prefix`] can rely on, so a reported hit length is always
/// fully backed by the blocks returned alongside it.
///
/// Without this, splitting a node mid-block dropped the remainder's blocks while
/// keeping its tokens, and a subsequent query reported a full hit while handing back
/// KV for only part of it — a silent generation corruption.
///
/// # Reference-counting contract
///
/// - [`Self::insert_sequence`] retains one pool reference per block for the tree. The
///   caller should release its own reference afterwards.
/// - [`Self::match_longest_prefix`] is a **read**: it retains nothing. A caller that
///   will use the blocks across a generation must call [`Self::pin_match`] and later
///   [`Self::unpin_match`], which is also what makes those blocks safe from eviction.
#[derive(Debug)]
pub struct RadixPrefixCache {
    nodes: Vec<Option<RadixNode>>,
    root_id: usize,
    free_node_ids: Vec<usize>,
    pub max_nodes: usize,
    pub stats: RadixCacheStats,
}

impl RadixPrefixCache {
    pub fn new(max_nodes: usize) -> Self {
        let mut cache = Self {
            nodes: Vec::new(),
            root_id: 0,
            free_node_ids: Vec::new(),
            max_nodes,
            stats: RadixCacheStats::default(),
        };

        // Initialize root node representing empty prefix
        let root = RadixNode::new(0, None, Vec::new(), Vec::new());
        cache.nodes.push(Some(root));
        cache
    }

    fn alloc_node_id(&mut self) -> usize {
        if let Some(id) = self.free_node_ids.pop() {
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(None);
            id
        }
    }

    /// Match the longest cached prefix of `tokens`.
    ///
    /// Returns `(blocks, matched_token_count)` where the blocks fully back every
    /// reported token: `blocks.len() * KV_BLOCK_TOKENS == matched_token_count`.
    ///
    /// This is a read and retains nothing — see the reference-counting contract on
    /// [`RadixPrefixCache`]. Call [`Self::pin_match`] to hold the result.
    pub fn match_longest_prefix(&mut self, tokens: &[u32]) -> (Vec<PhysicalBlockId>, usize) {
        self.stats.queries += 1;
        self.stats.tokens_queried += tokens.len() as u64;

        let mut current_id = self.root_id;
        let mut token_idx = 0;
        let mut matched_blocks = Vec::new();
        let now = Instant::now();

        while token_idx < tokens.len() {
            let next_token = tokens[token_idx];
            let child_id = match self.nodes[current_id]
                .as_ref()
                .and_then(|n| n.children.get(&next_token).copied())
            {
                Some(id) => id,
                None => break,
            };

            let child = self.nodes[child_id].as_mut().unwrap();
            child.last_accessed = now;

            // How many tokens match along this edge?
            let edge_tokens = &child.tokens;
            let mut edge_matched = 0;
            while edge_matched < edge_tokens.len()
                && (token_idx + edge_matched) < tokens.len()
                && edge_tokens[edge_matched] == tokens[token_idx + edge_matched]
            {
                edge_matched += 1;
            }

            // Reuse only whole blocks. On a full-edge match the invariant makes this
            // the entire block list; on a partial match it truncates to the blocks
            // that are actually covered.
            let usable_blocks = edge_matched / KV_BLOCK_TOKENS;
            matched_blocks.extend_from_slice(&child.blocks[..usable_blocks]);
            token_idx += usable_blocks * KV_BLOCK_TOKENS;

            if edge_matched < edge_tokens.len() {
                break;
            }
            current_id = child_id;
        }

        debug_assert_eq!(matched_blocks.len() * KV_BLOCK_TOKENS, token_idx);
        if token_idx > 0 {
            self.stats.hits += 1;
            self.stats.tokens_saved += token_idx as u64;
        }

        (matched_blocks, token_idx)
    }

    /// Hold a match's blocks across a generation, protecting them from eviction.
    pub fn pin_match(blocks: &[PhysicalBlockId], pool: &mut PagedKvBlockPool) {
        for &b in blocks {
            pool.retain(b);
        }
    }

    /// Release a pin taken by [`Self::pin_match`].
    pub fn unpin_match(blocks: &[PhysicalBlockId], pool: &mut PagedKvBlockPool) {
        for &b in blocks {
            pool.release(b);
        }
    }

    /// True when any of a node's blocks is held by someone other than the tree itself.
    fn node_is_pinned(node: &RadixNode, pool: &PagedKvBlockPool) -> bool {
        node.blocks
            .iter()
            .any(|b| pool.block(*b).is_some_and(|blk| blk.ref_count > 1))
    }

    /// Insert a sequence and its physical blocks into the tree.
    ///
    /// The input is truncated to a whole number of [`KV_BLOCK_TOKENS`]-token blocks.
    /// A partial trailing block cannot be reused by a later prefix match, so caching it
    /// would only reintroduce the token/block mismatch this structure exists to avoid;
    /// a sequence shorter than one block caches nothing and returns `Ok(())`.
    ///
    /// Retains one pool reference per stored block on the tree's behalf. The caller
    /// should release its own reference afterwards.
    pub fn insert_sequence(
        &mut self,
        tokens: &[u32],
        blocks: &[PhysicalBlockId],
        pool: &mut PagedKvBlockPool,
    ) -> Result<()> {
        let whole_blocks = (tokens.len() / KV_BLOCK_TOKENS).min(blocks.len());
        if whole_blocks == 0 {
            return Ok(());
        }
        let tokens = &tokens[..whole_blocks * KV_BLOCK_TOKENS];
        let blocks = &blocks[..whole_blocks];

        self.enforce_capacity(pool);

        let mut current_id = self.root_id;
        let mut token_offset = 0usize;
        let now = Instant::now();

        while token_offset < tokens.len() {
            // Block-aligned by construction, so the block cursor is derived from the
            // token cursor rather than accumulated separately — the two used to be
            // tracked independently and could drift apart (and index out of bounds).
            let block_offset = token_offset / KV_BLOCK_TOKENS;
            let next_token = tokens[token_offset];
            let child_id_opt = self.nodes[current_id]
                .as_ref()
                .and_then(|n| n.children.get(&next_token).copied());

            let Some(child_id) = child_id_opt else {
                // No child on this token: hang the whole remainder off `current_id`.
                let new_leaf_id = self.alloc_node_id();
                let leaf = RadixNode::new(
                    new_leaf_id,
                    Some(current_id),
                    tokens[token_offset..].to_vec(),
                    blocks[block_offset..].to_vec(),
                );
                for &b in &leaf.blocks {
                    pool.retain(b);
                }
                self.nodes[new_leaf_id] = Some(leaf);
                self.nodes[current_id]
                    .as_mut()
                    .unwrap()
                    .children
                    .insert(next_token, new_leaf_id);
                return Ok(());
            };

            let (common_len, edge_len) = {
                let child = self.nodes[child_id].as_ref().unwrap();
                let mut common = 0;
                while common < child.tokens.len()
                    && (token_offset + common) < tokens.len()
                    && child.tokens[common] == tokens[token_offset + common]
                {
                    common += 1;
                }
                (common, child.tokens.len())
            };

            if common_len == edge_len {
                // Whole edge matched: descend and keep walking.
                token_offset += edge_len;
                current_id = child_id;
                self.nodes[current_id].as_mut().unwrap().last_accessed = now;
                continue;
            }

            // Divergence inside this edge. Only a block-aligned prefix is shareable,
            // so round the split point down to a block boundary.
            let split_at = (common_len / KV_BLOCK_TOKENS) * KV_BLOCK_TOKENS;
            if split_at == 0 {
                // Under one block in common: there is nothing to share and splitting
                // here would create a node holding tokens but no blocks.
                return Ok(());
            }

            let split_id = self.alloc_node_id();
            let (prefix_tokens, prefix_blocks) = {
                let child = self.nodes[child_id].as_mut().unwrap();
                child.parent = Some(split_id);
                let remaining_tokens = child.tokens.split_off(split_at);
                let remaining_blocks = child.blocks.split_off(split_at / KV_BLOCK_TOKENS);
                (
                    std::mem::replace(&mut child.tokens, remaining_tokens),
                    std::mem::replace(&mut child.blocks, remaining_blocks),
                )
            };
            let first_remaining = self.nodes[child_id].as_ref().unwrap().tokens[0];

            let mut split_node =
                RadixNode::new(split_id, Some(current_id), prefix_tokens, prefix_blocks);
            split_node.children.insert(first_remaining, child_id);
            self.nodes[split_id] = Some(split_node);
            self.nodes[current_id]
                .as_mut()
                .unwrap()
                .children
                .insert(next_token, split_id);

            // Whether the new sequence's own tail can also be cached depends on where
            // it left the old edge:
            //
            // - `split_at == common_len` — the two diverge exactly on a block boundary,
            //   so `tokens[split_at]` differs from the remainder child's first token and
            //   the tail attaches as a sibling under the split node.
            // - `split_at < common_len` — they still agree for the sub-block remainder,
            //   so both branches would key on the SAME token and the new leaf would
            //   silently displace the existing child in the `children` map. There is no
            //   block-aligned point to attach at; only the shared prefix is cached.
            //
            // `split_at` is an offset within the edge, so it has to be rebased onto
            // `token_offset` before indexing the caller's arrays.
            let tail_start = token_offset + split_at;
            if split_at == common_len && tail_start < tokens.len() {
                let new_leaf_id = self.alloc_node_id();
                let leaf = RadixNode::new(
                    new_leaf_id,
                    Some(split_id),
                    tokens[tail_start..].to_vec(),
                    blocks[tail_start / KV_BLOCK_TOKENS..].to_vec(),
                );
                for &b in &leaf.blocks {
                    pool.retain(b);
                }
                let key = leaf.tokens[0];
                self.nodes[new_leaf_id] = Some(leaf);
                self.nodes[split_id]
                    .as_mut()
                    .unwrap()
                    .children
                    .insert(key, new_leaf_id);
            }
            return Ok(());
        }

        Ok(())
    }

    /// Evict LRU leaves until there is room to add nodes for one more insert.
    ///
    /// A single insert adds at most two nodes (a split plus a leaf), so this keeps that
    /// much headroom. `max_nodes` was previously stored and never read, leaving the
    /// tree unbounded.
    fn enforce_capacity(&mut self, pool: &mut PagedKvBlockPool) {
        if self.max_nodes == 0 {
            return;
        }
        while self.node_count() + 2 > self.max_nodes {
            if self.evict_lru(1, pool) == 0 {
                break; // everything left is pinned or internal — nothing to reclaim
            }
        }
    }

    /// Evict the oldest unpinned leaf nodes, returning their blocks to the pool.
    ///
    /// A node is evictable only when it has no children and none of its blocks is held
    /// by a caller. Pinning is read from the pool's own block reference counts: the
    /// tree holds exactly one reference per block, so `ref_count > 1` means someone
    /// called [`Self::pin_match`] and is still using it.
    ///
    /// This previously gated on a `RadixNode::ref_count` field that was initialised to
    /// zero and never incremented anywhere, so the guard was always true and a node an
    /// in-flight sequence depended on could be evicted out from under it.
    pub fn evict_lru(&mut self, count: usize, pool: &mut PagedKvBlockPool) -> usize {
        let mut evicted = 0;

        for _ in 0..count {
            let mut oldest_leaf: Option<(usize, Instant)> = None;

            for (idx, node_opt) in self.nodes.iter().enumerate() {
                if idx == self.root_id {
                    continue;
                }
                if let Some(node) = node_opt {
                    if node.children.is_empty() && !Self::node_is_pinned(node, pool) {
                        match oldest_leaf {
                            None => oldest_leaf = Some((idx, node.last_accessed)),
                            Some((_, oldest_time)) if node.last_accessed < oldest_time => {
                                oldest_leaf = Some((idx, node.last_accessed));
                            }
                            _ => {}
                        }
                    }
                }
            }

            if let Some((leaf_id, _)) = oldest_leaf {
                let leaf = self.nodes[leaf_id].take().unwrap();
                for &b in &leaf.blocks {
                    pool.release(b);
                }

                if let Some(parent_id) = leaf.parent {
                    if let Some(parent) = self.nodes[parent_id].as_mut() {
                        parent
                            .children
                            .retain(|_, &mut child_id| child_id != leaf_id);
                    }
                }

                self.free_node_ids.push(leaf_id);
                evicted += 1;
                self.stats.evicted_nodes += 1;
            } else {
                break;
            }
        }

        evicted
    }

    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::KvDtype;

    fn dummy_plan() -> crate::inference::kv_cache::LlamaKvCachePlan {
        crate::inference::kv_cache::LlamaKvCachePlan {
            max_sequence_length: 512,
            layer_count: 1,
            kv_head_count: 1,
            head_dim: 64,
            k_head_dim: 64,
            v_head_dim: 64,
            key_shape: vec![1, 1, 512, 64],
            value_shape: vec![1, 1, 512, 64],
        }
    }

    fn pool(capacity: usize) -> PagedKvBlockPool {
        PagedKvBlockPool::new(dummy_plan(), KvDtype::F32, capacity)
    }

    /// `n` blocks' worth of tokens starting from `base`, so sequences are always
    /// block-aligned and actually cacheable at this granularity.
    fn seq(base: u32, blocks: usize) -> Vec<u32> {
        (0..blocks * KV_BLOCK_TOKENS)
            .map(|i| base + i as u32)
            .collect()
    }

    fn alloc_n(pool: &mut PagedKvBlockPool, n: usize) -> Vec<PhysicalBlockId> {
        (0..n).map(|_| pool.allocate().unwrap()).collect()
    }

    /// Every reported hit must be fully backed by the blocks handed back with it.
    fn assert_consistent(blocks: &[PhysicalBlockId], tokens: usize) {
        assert_eq!(
            blocks.len() * KV_BLOCK_TOKENS,
            tokens,
            "reported {tokens} cached tokens but returned {} block(s)",
            blocks.len()
        );
    }

    #[test]
    fn insert_and_match_block_aligned_sequences() {
        let mut pool = pool(32);
        let mut cache = RadixPrefixCache::new(64);

        let tokens = seq(100, 4);
        let blocks = alloc_n(&mut pool, 4);
        cache.insert_sequence(&tokens, &blocks, &mut pool).unwrap();

        let (matched, len) = cache.match_longest_prefix(&tokens);
        assert_eq!(len, 4 * KV_BLOCK_TOKENS);
        assert_eq!(matched, blocks);
        assert_consistent(&matched, len);
    }

    /// Regression for the defect that made this structure unsafe to wire up.
    ///
    /// A split used to divide `tokens` at the exact divergence point but `blocks` at
    /// `div_ceil(KV_BLOCK_TOKENS)`, so the remainder node kept its tokens and lost all
    /// of its blocks. Re-querying the ORIGINAL sequence then reported a full-length hit
    /// while handing back KV for only part of it — the caller would skip prefill for
    /// positions that had no KV at all. The PR's own test never re-queried the first
    /// sequence after the split, which is the one query that exposes it.
    #[test]
    fn a_split_never_reports_more_tokens_than_its_blocks_cover() {
        let mut pool = pool(64);
        let mut cache = RadixPrefixCache::new(64);

        // Turn 1: four blocks of KV.
        let turn1 = seq(100, 4);
        let blocks1 = alloc_n(&mut pool, 4);
        cache.insert_sequence(&turn1, &blocks1, &mut pool).unwrap();

        // Turn 2 shares the first two blocks, then diverges. This forces a split.
        let mut turn2 = turn1[..2 * KV_BLOCK_TOKENS].to_vec();
        turn2.extend(seq(9000, 2));
        let blocks2 = alloc_n(&mut pool, 4);
        cache.insert_sequence(&turn2, &blocks2, &mut pool).unwrap();

        // Re-query turn 1 after the split.
        let (matched, len) = cache.match_longest_prefix(&turn1);
        assert_consistent(&matched, len);
        assert_eq!(
            len,
            4 * KV_BLOCK_TOKENS,
            "turn 1 is still fully cached across the split"
        );
        assert_eq!(matched, blocks1, "and by its original blocks");

        // The shared prefix is reachable on its own.
        let (shared, shared_len) = cache.match_longest_prefix(&turn2);
        assert_consistent(&shared, shared_len);
        assert!(shared_len >= 2 * KV_BLOCK_TOKENS);
    }

    /// Divergence part-way through a block can only share the whole blocks before it.
    #[test]
    fn a_mid_block_divergence_shares_only_the_completed_blocks() {
        let mut pool = pool(64);
        let mut cache = RadixPrefixCache::new(64);

        let turn1 = seq(100, 3);
        let blocks1 = alloc_n(&mut pool, 3);
        cache.insert_sequence(&turn1, &blocks1, &mut pool).unwrap();

        // Diverge 8 tokens into the third block: 2 whole blocks are shareable.
        let mut probe = turn1[..2 * KV_BLOCK_TOKENS + 8].to_vec();
        probe.push(77_777);
        let (matched, len) = cache.match_longest_prefix(&probe);
        assert_consistent(&matched, len);
        assert_eq!(len, 2 * KV_BLOCK_TOKENS);
    }

    /// A prompt shorter than one block has no reusable KV at this granularity, so it
    /// is not cached at all rather than cached inconsistently.
    #[test]
    fn sequences_below_one_block_are_not_cached() {
        let mut pool = pool(16);
        let mut cache = RadixPrefixCache::new(64);

        let short = vec![1, 2, 3, 4, 5];
        let b = alloc_n(&mut pool, 1);
        cache.insert_sequence(&short, &b, &mut pool).unwrap();

        assert_eq!(cache.node_count(), 1, "root only");
        let (matched, len) = cache.match_longest_prefix(&short);
        assert_eq!(len, 0);
        assert!(matched.is_empty());
    }

    /// `match_longest_prefix` is a read. It used to `retain` every matched block with
    /// no release path anywhere in the API, so refcounts rose with query volume and
    /// blocks could never return to the free list.
    #[test]
    fn matching_is_a_read_and_retains_nothing() {
        let mut pool = pool(32);
        let mut cache = RadixPrefixCache::new(64);

        let tokens = seq(100, 2);
        let blocks = alloc_n(&mut pool, 2);
        cache.insert_sequence(&tokens, &blocks, &mut pool).unwrap();

        let before = pool.block(blocks[0]).unwrap().ref_count;
        for _ in 0..25 {
            let _ = cache.match_longest_prefix(&tokens);
        }
        assert_eq!(
            pool.block(blocks[0]).unwrap().ref_count,
            before,
            "25 read-only queries must not move the reference count"
        );
    }

    /// Eviction must not pull KV out from under an in-flight sequence. The guard used
    /// to read a `RadixNode::ref_count` field that nothing ever incremented, so it was
    /// always true and any leaf was evictable.
    #[test]
    fn eviction_skips_pinned_nodes() {
        let mut pool = pool(32);
        let mut cache = RadixPrefixCache::new(64);

        let tokens = seq(100, 2);
        let blocks = alloc_n(&mut pool, 2);
        cache.insert_sequence(&tokens, &blocks, &mut pool).unwrap();
        for &b in &blocks {
            pool.release(b); // hand the tree sole ownership
        }

        let (matched, len) = cache.match_longest_prefix(&tokens);
        assert_eq!(len, 2 * KV_BLOCK_TOKENS);

        RadixPrefixCache::pin_match(&matched, &mut pool);
        assert_eq!(
            cache.evict_lru(4, &mut pool),
            0,
            "a pinned prefix must survive eviction"
        );
        let (_, still_there) = cache.match_longest_prefix(&tokens);
        assert_eq!(still_there, 2 * KV_BLOCK_TOKENS);

        RadixPrefixCache::unpin_match(&matched, &mut pool);
        assert_eq!(cache.evict_lru(4, &mut pool), 1, "unpinned, it can go");
        let (_, gone) = cache.match_longest_prefix(&tokens);
        assert_eq!(gone, 0);
    }

    /// Evicting a node returns its blocks to the pool exactly once.
    #[test]
    fn eviction_returns_blocks_to_the_pool() {
        let mut pool = pool(32);
        let mut cache = RadixPrefixCache::new(64);

        let tokens = seq(100, 2);
        let blocks = alloc_n(&mut pool, 2);
        cache.insert_sequence(&tokens, &blocks, &mut pool).unwrap();
        for &b in &blocks {
            pool.release(b);
        }
        let allocated = pool.allocated_count();

        assert_eq!(cache.evict_lru(1, &mut pool), 1);
        assert_eq!(
            pool.allocated_count(),
            allocated - 2,
            "both blocks returned to the free list"
        );
    }

    /// `max_nodes` was stored at construction and never read, so the tree grew without
    /// bound and eviction only ever happened if a caller asked for it explicitly.
    #[test]
    fn capacity_is_enforced_on_insert() {
        let mut pool = pool(256);
        let max_nodes = 6;
        let mut cache = RadixPrefixCache::new(max_nodes);

        for turn in 0..40u32 {
            let tokens = seq(turn * 1000, 1);
            let blocks = alloc_n(&mut pool, 1);
            cache.insert_sequence(&tokens, &blocks, &mut pool).unwrap();
            for &b in &blocks {
                pool.release(b);
            }
            assert!(
                cache.node_count() <= max_nodes,
                "node count {} exceeded max_nodes {max_nodes} after turn {turn}",
                cache.node_count()
            );
        }
    }

    /// A split that happens after descending through an existing node, so the split
    /// point is at a non-zero token offset.
    ///
    /// `split_at` is an offset *within the edge* being split, and has to be rebased
    /// onto the walk's `token_offset` before it indexes the caller's token/block
    /// arrays. Every case where the split lands on the very first edge has
    /// `token_offset == 0`, which makes the two spellings coincide and hides the
    /// difference — this exercises the case where they do not.
    #[test]
    fn a_split_below_the_root_rebases_onto_the_walk_offset() {
        let mut pool = pool(256);
        let mut cache = RadixPrefixCache::new(128);

        // Shared trunk (2 blocks), then branch A continues for 4 more.
        let trunk = seq(1, 2);
        let mut branch_a = trunk.clone();
        branch_a.extend(seq(500, 4));
        let blocks_a = alloc_n(&mut pool, 6);
        cache
            .insert_sequence(&branch_a, &blocks_a, &mut pool)
            .unwrap();

        // Branch B splits off a node that is itself reached by descending the trunk.
        let mut branch_b = trunk.clone();
        branch_b.extend(seq(500, 2)); // shares 2 more blocks with A
        branch_b.extend(seq(900, 3)); // then diverges on a block boundary
        let blocks_b = alloc_n(&mut pool, 7);
        cache
            .insert_sequence(&branch_b, &blocks_b, &mut pool)
            .unwrap();

        // Both branches must come back whole, and backed by their OWN blocks.
        let (matched_a, len_a) = cache.match_longest_prefix(&branch_a);
        assert_consistent(&matched_a, len_a);
        assert_eq!(len_a, 6 * KV_BLOCK_TOKENS, "branch A stays fully cached");
        assert_eq!(matched_a, blocks_a);

        let (matched_b, len_b) = cache.match_longest_prefix(&branch_b);
        assert_consistent(&matched_b, len_b);
        assert_eq!(len_b, 7 * KV_BLOCK_TOKENS, "branch B is cached end to end");
        // The shared prefix is A's blocks; only the divergent tail is B's own.
        assert_eq!(matched_b[..4], blocks_a[..4]);
        assert_eq!(matched_b[4..], blocks_b[4..]);
    }

    /// Walking a long shared trunk keeps the block cursor aligned with the token
    /// cursor. These were tracked independently before and could drift, which is also
    /// what made an out-of-range slice reachable.
    #[test]
    fn deep_chains_stay_block_aligned() {
        let mut pool = pool(256);
        let mut cache = RadixPrefixCache::new(128);

        let trunk = seq(1, 8);
        for extra in 0..4u32 {
            let mut branch = trunk.clone();
            branch.extend(seq(50_000 + extra * 1000, 2));
            let blocks = alloc_n(&mut pool, 10);
            cache.insert_sequence(&branch, &blocks, &mut pool).unwrap();
            for &b in &blocks {
                pool.release(b);
            }

            let (matched, len) = cache.match_longest_prefix(&branch);
            assert_consistent(&matched, len);
            assert_eq!(len, 10 * KV_BLOCK_TOKENS);
        }

        let (matched, len) = cache.match_longest_prefix(&trunk);
        assert_consistent(&matched, len);
        assert_eq!(len, 8 * KV_BLOCK_TOKENS);
    }
}
