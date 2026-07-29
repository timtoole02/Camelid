//! Lossless greedy speculative decoding.
//!
//! Decode is memory-bandwidth bound: every sequential token costs a full
//! weight read. Speculation drafts k candidate tokens cheaply, then verifies
//! them in ONE batched forward through the target model
//! (`forward_greedy_verify_chunk`), so a single weight read can yield several
//! accepted tokens. Every emitted token is the target model's own greedy
//! argmax given the accepted prefix, so accepted output is the same token
//! stream vanilla greedy decode produces; rejected drafts are dropped by KV
//! rollback and never observable.
//!
//! Support boundary: speculation is a default-off serving optimization. It
//! makes no support claim, moves no release-ledger row, and byte-parity for a
//! given lane is asserted only by evidence (tests and parity receipts), never
//! by resemblance.
//!
//! Two drafters:
//! - [`NGramDrafter`] (prompt lookup): proposes the continuation of the most
//!   recent earlier occurrence of the current suffix. Zero extra weights;
//!   wins on repetitive/structured text, proposes nothing on novel text.
//! - [`ModelDrafter`]: a small model (same tokenizer) greedily drafts k
//!   tokens; the target verifies them in one pass.

use crate::inference::{LlamaInferenceSession, LlamaSampler};
use crate::Result;
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
};

/// Default drafted tokens per round for the n-gram drafter. The n-gram lookup
/// itself is nearly free, but each extra draft widens the batched verify GEMM, so
/// over-drafting wastes work on partial-acceptance text (code, prose) without
/// helping. Measured on a 3B Q8_0 GPU resident decode (RTX 3060): a draft count of
/// 5 (verify batch k=6) sits at the sweet spot — within ~1% of the maximum
/// repetitive-text speedup (~2.55x) while giving the best result on moderately
/// repetitive code (~1.20x), where 7 drafts regress to ~1.09x. Bounded by
/// `cuda_resident::MAX_VERIFY_K - 1`.
pub const DEFAULT_NGRAM_DRAFT_TOKENS: usize = 5;
pub const DEFAULT_NGRAM_MIN_MATCH: usize = 3;
pub const DEFAULT_NGRAM_MAX_MATCH: usize = 4;

/// Default drafted tokens per round for the draft-model drafter. Each draft
/// token costs a sequential forward through the draft model, so the window
/// stays shorter.
pub const DEFAULT_MODEL_DRAFT_TOKENS: usize = 5;

/// Count the longest accepted prefix: drafted tokens that equal the target's
/// own greedy predictions position by position.
pub fn accepted_draft_prefix(drafts: &[u32], target_predictions: &[u32]) -> usize {
    drafts
        .iter()
        .zip(target_predictions.iter())
        .take_while(|(draft, prediction)| draft == prediction)
        .count()
}

pub enum SpeculativeDrafter {
    NGram(NGramDrafter),
    Model(Box<ModelDrafter>),
}

impl SpeculativeDrafter {
    /// Propose up to `max_tokens` draft tokens to follow `history` (the full
    /// token sequence so far: prompt plus generated, including the trailing
    /// token the target has not consumed yet). May return fewer or none.
    pub fn draft(&mut self, history: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        match self {
            Self::NGram(drafter) => Ok(drafter.draft(history, max_tokens)),
            Self::Model(drafter) => drafter.draft(history, max_tokens),
        }
    }

    /// Draft-decode profiling: (resident GPU forward µs, resident steps, CPU-fallback steps).
    /// Zero for the n-gram drafter (no model forward). Resets on read.
    pub fn take_forward_stats(&mut self) -> (u128, u64, u64) {
        match self {
            Self::NGram(_) => (0, 0, 0),
            Self::Model(drafter) => drafter.take_forward_stats(),
        }
    }
}

/// Prompt-lookup drafting: find the longest n-gram suffix of `history`
/// (between `min_ngram` and `max_ngram`) that occurred earlier, preferring
/// the most recent occurrence, and propose the tokens that followed it.
#[derive(Debug)]
pub struct NGramDrafter {
    pub max_ngram: usize,
    pub min_ngram: usize,
    index: RefCell<BoundedNGramIndex>,
}

impl Clone for NGramDrafter {
    fn clone(&self) -> Self {
        // A clone is a new drafting stream. Copying a long prompt index is
        // expensive and can also couple two unrelated histories; preserve the
        // policy/capacity and let the clone index its own first history.
        Self::new_with_index_capacity(
            self.min_ngram,
            self.max_ngram,
            self.index.borrow().max_entries,
        )
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NGramIndexStats {
    pub indexed_tokens: usize,
    pub records: usize,
    pub rebuilds: u64,
    pub appended_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NGramKey {
    len: usize,
    hash: u64,
}

#[derive(Debug, Clone)]
struct BoundedNGramIndex {
    history: Vec<u32>,
    occurrences: HashMap<NGramKey, VecDeque<usize>>,
    insertion_order: VecDeque<(NGramKey, usize)>,
    max_entries: usize,
    min_ngram: usize,
    max_ngram: usize,
    rebuilds: u64,
    appended_tokens: u64,
}

impl Default for NGramDrafter {
    fn default() -> Self {
        // Two-token patterns (e.g. ", " pairs) recur with unrelated
        // continuations and mostly waste verify rows; three-token matches
        // measure far higher acceptance.
        Self::new(DEFAULT_NGRAM_MIN_MATCH, DEFAULT_NGRAM_MAX_MATCH)
    }
}

impl NGramDrafter {
    pub fn new(min_ngram: usize, max_ngram: usize) -> Self {
        Self::new_with_index_capacity(
            min_ngram,
            max_ngram,
            crate::runtime_config::ngram_index_max_entries(),
        )
    }

    pub fn new_with_index_capacity(min_ngram: usize, max_ngram: usize, max_entries: usize) -> Self {
        let min_ngram = min_ngram.max(1);
        let max_ngram = max_ngram.max(min_ngram);
        Self {
            min_ngram,
            max_ngram,
            index: RefCell::new(BoundedNGramIndex::new(min_ngram, max_ngram, max_entries)),
        }
    }

    pub fn draft(&self, history: &[u32], max_tokens: usize) -> Vec<u32> {
        if max_tokens == 0 || self.min_ngram == 0 || history.len() <= self.min_ngram {
            return Vec::new();
        }
        let mut index = self.index.borrow_mut();
        index.sync(history);
        index.draft(max_tokens)
    }

    pub fn index_stats(&self) -> NGramIndexStats {
        self.index.borrow().stats()
    }
}

impl BoundedNGramIndex {
    fn new(min_ngram: usize, max_ngram: usize, max_entries: usize) -> Self {
        Self {
            history: Vec::new(),
            occurrences: HashMap::new(),
            insertion_order: VecDeque::new(),
            max_entries: max_entries.max(1),
            min_ngram,
            max_ngram,
            rebuilds: 0,
            appended_tokens: 0,
        }
    }

    fn sync(&mut self, history: &[u32]) {
        let append_only = self.history.len() <= history.len() && history.starts_with(&self.history);
        if !append_only {
            self.history.clear();
            self.occurrences.clear();
            self.insertion_order.clear();
            self.rebuilds = self.rebuilds.saturating_add(1);
        }
        let old_len = self.history.len();
        for &token in &history[old_len..] {
            self.history.push(token);
            self.index_new_tail();
        }
        self.appended_tokens = self
            .appended_tokens
            .saturating_add((history.len() - old_len) as u64);
    }

    fn index_new_tail(&mut self) {
        let end = self.history.len();
        for len in self.min_ngram..=self.max_ngram.min(end) {
            let start = end - len;
            let key = NGramKey {
                len,
                hash: hash_ngram(&self.history[start..end]),
            };
            self.occurrences.entry(key).or_default().push_back(start);
            self.insertion_order.push_back((key, start));
        }
        while self.insertion_order.len() > self.max_entries {
            let (key, start) = self
                .insertion_order
                .pop_front()
                .expect("length checked above");
            let remove_key = if let Some(starts) = self.occurrences.get_mut(&key) {
                if starts.front() == Some(&start) {
                    starts.pop_front();
                } else if let Some(index) = starts.iter().position(|candidate| *candidate == start)
                {
                    starts.remove(index);
                }
                starts.is_empty()
            } else {
                false
            };
            if remove_key {
                self.occurrences.remove(&key);
            }
        }
    }

    fn draft(&self, max_tokens: usize) -> Vec<u32> {
        let len = self.history.len();
        let max_n = self.max_ngram.min(len.saturating_sub(1));
        for n in (self.min_ngram..=max_n).rev() {
            let suffix_start = len - n;
            let pattern = &self.history[suffix_start..];
            let key = NGramKey {
                len: n,
                hash: hash_ngram(pattern),
            };
            let Some(starts) = self.occurrences.get(&key) else {
                continue;
            };
            for &start in starts.iter().rev() {
                // Exclude the suffix itself and token-verify every hash hit.
                // A collision can cost lookup work, never change a draft.
                if start >= suffix_start
                    || &self.history[start..start + n] != pattern
                    || start + n >= len
                {
                    continue;
                }
                let continuation_start = start + n;
                let continuation_end = (continuation_start + max_tokens).min(len);
                return self.history[continuation_start..continuation_end].to_vec();
            }
        }
        Vec::new()
    }

    fn stats(&self) -> NGramIndexStats {
        NGramIndexStats {
            indexed_tokens: self.history.len(),
            records: self.insertion_order.len(),
            rebuilds: self.rebuilds,
            appended_tokens: self.appended_tokens,
        }
    }
}

fn hash_ngram(tokens: &[u32]) -> u64 {
    // Stable FNV-1a over token bytes. Hashes are lookup accelerators only:
    // every candidate is compared token-for-token before it can draft.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Draft-model drafting: a smaller model with the SAME token mapping runs
/// greedy decode ahead of the target. The draft session mirrors the accepted
/// sequence by re-ingesting tokens from `history` (`committed` counts the
/// history tokens whose KV entries are valid); each round's speculative tail
/// is rolled back before the next round, so rejected drafts never contaminate
/// the draft context.
pub struct ModelDrafter {
    session: LlamaInferenceSession,
    committed: usize,
    /// Drafted tokens fed into the session's KV beyond `committed` last
    /// round. The prefix of these that the target accepted is now real
    /// history, so its KV entries can be kept instead of re-ingested.
    speculative_fed: Vec<u32>,
    /// Profiling: summed GPU forward microseconds reported by the resident decode for draft
    /// steps, the count of resident steps, and the count that fell back to the CPU path. Lets a
    /// caller compare the GPU forward time against the wall-clock draft time to localize overhead.
    resident_forward_us: u128,
    resident_steps: u64,
    cpu_fallback_steps: u64,
}

impl ModelDrafter {
    pub fn new(mut session: LlamaInferenceSession) -> Self {
        // Route the draft session's GPU resident engine to the dedicated drafter
        // cache so it coexists with the target's engine. Resident decode stays
        // enabled (the draft model runs fast on the GPU); rollback of rejected
        // drafts uses `rollback_resident_to_position`, which resets the engine's
        // `filled` so the GPU KV (still valid up to the accepted prefix) is trusted
        // rather than reseeded. If the draft engine doesn't fit in VRAM it falls
        // back to the CPU path per token automatically.
        session.set_is_drafter(true);
        // Register the draft's resident VRAM footprint so a target engine built AFTER this
        // (e.g. when the drafter is configured before the target's first decode) leaves room for
        // the draft to stay GPU-resident too. Only honored on a GPU where the target still fits
        // fully resident after the reserve; otherwise the draft falls back to CPU. No-op on
        // non-CUDA builds. (Does not evict an already-built target — see set_spec_coexist_reserve.)
        crate::inference::set_spec_coexist_reserve(session.spec_coexist_reserve_estimate());
        Self {
            session,
            committed: 0,
            speculative_fed: Vec::new(),
            resident_forward_us: 0,
            resident_steps: 0,
            cpu_fallback_steps: 0,
        }
    }

    /// Take and reset the draft-decode profiling counters: (summed resident GPU forward µs,
    /// resident step count, CPU-fallback step count).
    pub fn take_forward_stats(&mut self) -> (u128, u64, u64) {
        let stats = (
            self.resident_forward_us,
            self.resident_steps,
            self.cpu_fallback_steps,
        );
        self.resident_forward_us = 0;
        self.resident_steps = 0;
        self.cpu_fallback_steps = 0;
        stats
    }

    pub fn draft(&mut self, history: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        if max_tokens == 0 || history.is_empty() {
            return Ok(Vec::new());
        }
        // Last round's speculative KV entries start at `committed`. The
        // prefix that matches history (accepted drafts) is kept; only the
        // rejected tail is rolled back and never re-fed.
        let reuse = history[self.committed..]
            .iter()
            .zip(self.speculative_fed.iter())
            .take_while(|(token, fed)| token == fed)
            .count();
        self.session
            .rollback_resident_to_position(self.committed + reuse)?;
        self.committed += reuse;
        self.speculative_fed.clear();
        let pending = &history[self.committed..];
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        // Cap so the pending chunk plus the drafted tail fits the draft
        // model's context window.
        let room = self
            .session
            .remaining_context()
            .saturating_sub(pending.len());
        let max_tokens = max_tokens.min(room.saturating_add(1));
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        // Re-ingest the pending (known) tokens, then the prediction after the LAST one is the
        // first draft. The whole chunk rides the fast resident GPU-argmax lane (the draft only
        // needs the argmax, so the full-logits copy + CPU sample the diagnostics path does is pure
        // per-round overhead — the dominant cost once the draft model is GPU-resident). The
        // diagnostics path is the fallback only when the resident engine isn't ready (not yet
        // seeded), in which case nothing has been fed so re-feeding the whole chunk is consistent.
        // Lossless either way — the target verify is authoritative, so the draft's greedy choice
        // only affects accept rate, never the emitted tokens.
        // Feed the pending (known) tokens one at a time on the fast resident GPU-argmax lane; the
        // prediction after the LAST is the first draft. Token-by-token keeps the draft KV exactly
        // in sync (the batched-prefill diagnostics path desyncs the drafter's resident engine and
        // tanks accept). The diagnostics path is the fallback only when the resident engine isn't
        // ready, in which case nothing has been fed yet so re-feeding the whole chunk is consistent.
        let (&head, rest) = pending
            .split_first()
            .expect("pending is non-empty (checked above)");
        let first = match self.session.generate_next_token_greedy_resident(head)? {
            Some((mut pred, us)) => {
                self.resident_forward_us += us;
                self.resident_steps += 1;
                for &tok in rest {
                    pred = match self.session.generate_next_token_greedy_resident(tok)? {
                        Some((id, us)) => {
                            self.resident_forward_us += us;
                            self.resident_steps += 1;
                            id
                        }
                        None => {
                            self.cpu_fallback_steps += 1;
                            self.session
                                .generate_next_token_with_history_diagnostics(
                                    &[tok],
                                    LlamaSampler::Greedy,
                                    history,
                                    false,
                                    None,
                                )?
                                .next_token_id
                        }
                    };
                }
                pred
            }
            None => {
                self.cpu_fallback_steps += 1;
                self.session
                    .generate_next_token_with_history_diagnostics(
                        pending,
                        LlamaSampler::Greedy,
                        history,
                        false,
                        None,
                    )?
                    .next_token_id
            }
        };
        self.committed = history.len();
        let mut drafts = Vec::with_capacity(max_tokens);
        drafts.push(first);
        while drafts.len() < max_tokens {
            let last = *drafts.last().expect("drafts is non-empty");
            // Sequential draft steps on the fast resident GPU-argmax lane (no full-logits copy).
            let next = match self.session.generate_next_token_greedy_resident(last)? {
                Some((id, us)) => {
                    self.resident_forward_us += us;
                    self.resident_steps += 1;
                    id
                }
                None => {
                    self.cpu_fallback_steps += 1;
                    self.session
                        .generate_next_token_with_history_diagnostics(
                            &[last],
                            LlamaSampler::Greedy,
                            history,
                            false,
                            None,
                        )?
                        .next_token_id
                }
            };
            drafts.push(next);
        }
        // KV now holds `committed` history tokens plus the fed drafts (all
        // but the last drafted token); the next round keeps whatever prefix
        // the target accepts and rolls back the rest.
        self.speculative_fed = drafts[..drafts.len().saturating_sub(1)].to_vec();
        Ok(drafts)
    }
}

/// STAMPEDE Phase 5 (P5.2) — the acceptance-gated RUN-LENGTH latch, extracted
/// verbatim from the bench-speculative tree lane so the GPU-verified and
/// CPU-verified rounds (and, staged, the serve loop) drive ONE policy instead
/// of divergent copies.
///
/// Policy (measured on this box, 3B Q8, see the tree-lane receipts): while
/// speculating, draw the FULL budget every round; only a RUN of `exit_run`
/// consecutive rounds each accepting fewer than `productive_drafts` drafts
/// latches speculation OFF (run-length, not EWMA — real-text acceptance is
/// bursty). While OFF, skip speculation entirely (~1.0× floor); every
/// `low_reprobe` skips, spend ONE full-budget probe and re-latch ON when it
/// accepts ≥ `enter_drafts`. Warm-up rounds always speculate so a stream's
/// true acceptance is observed before the latch may turn off. Anchor-only
/// misses and engine-readiness misses must NOT be reported (they are not
/// acceptance measurements).
#[derive(Debug, Clone)]
pub struct SpecLatch {
    /// A round accepting >= this many DRAFTS (the +1 bonus excluded) is "productive".
    pub productive_drafts: u32,
    /// Consecutive non-productive VERIFIED rounds before latching OFF.
    /// 4, not 2: repetitive text strings together 2-3 sub-productive rounds
    /// mid-list; exiting early erodes the win (measured: EXIT_RUN=2 cut
    /// repetitive below 1.2×).
    pub exit_run: u32,
    /// A re-probe accepting >= this many drafts re-latches ON.
    pub enter_drafts: u32,
    /// Verified rounds before the latch may turn off.
    pub warmup_rounds: u64,
    /// Consecutive skips between full-budget re-probes (rare on purpose: a
    /// novel stream pays ~1 wasted verify per this many tokens).
    pub low_reprobe: u32,
    rounds_done: u64,
    consecutive_skips: u32,
    nonproductive_run: u32,
    speculating: bool,
}

impl Default for SpecLatch {
    fn default() -> Self {
        Self {
            productive_drafts: 2,
            exit_run: 4,
            enter_drafts: 2,
            warmup_rounds: 1,
            low_reprobe: 64,
            rounds_done: 0,
            consecutive_skips: 0,
            nonproductive_run: 0,
            // Start latched ON so warm-up measures true acceptance.
            speculating: true,
        }
    }
}

impl SpecLatch {
    /// Should this round draft at the full budget? `false` = skip speculation
    /// (plain decode step); callers must then report the skip via
    /// [`SpecLatch::note_skip`].
    pub fn should_speculate(&self) -> bool {
        self.rounds_done < self.warmup_rounds
            || self.speculating
            || self.consecutive_skips >= self.low_reprobe
    }

    /// Record a skipped (non-drafted) round.
    pub fn note_skip(&mut self) {
        self.consecutive_skips = self.consecutive_skips.saturating_add(1);
    }

    /// Record a VERIFIED round's accepted-draft count (the +1 bonus token
    /// excluded). Never call for anchor-only or engine-miss rounds.
    pub fn note_verified(&mut self, accepted_drafts: u32) {
        self.consecutive_skips = 0;
        if accepted_drafts >= self.productive_drafts {
            self.nonproductive_run = 0;
            if !self.speculating && accepted_drafts >= self.enter_drafts {
                self.speculating = true;
            }
        } else {
            self.nonproductive_run = self.nonproductive_run.saturating_add(1);
            if self.speculating && self.nonproductive_run >= self.exit_run {
                self.speculating = false;
                self.nonproductive_run = 0;
            }
        }
        self.rounds_done = self.rounds_done.saturating_add(1);
    }

    pub fn speculating(&self) -> bool {
        self.speculating
    }

    pub fn rounds_done(&self) -> u64 {
        self.rounds_done
    }

    pub fn consecutive_skips(&self) -> u32 {
        self.consecutive_skips
    }

    pub fn nonproductive_run(&self) -> u32 {
        self.nonproductive_run
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_drafts_continuation_of_most_recent_match() {
        let drafter = NGramDrafter::default();
        // ... 1 2 3 4 | 9 9 | 1 2 3 4 | 7 8 | ... suffix [7, 8] has no
        // earlier match; suffix ending [3, 4] repeats.
        let history = vec![1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3, 4];
        // Suffix [1, 2, 3, 4] (n=4) matches at start; continuation is [5, 6, 9].
        assert_eq!(drafter.draft(&history, 3), vec![5, 6, 9]);
    }

    #[test]
    fn ngram_prefers_longer_patterns_and_recent_matches() {
        let drafter = NGramDrafter::default();
        // [3, 4] occurs twice earlier with different continuations; the most
        // recent occurrence (followed by 8) wins.
        let history = vec![3, 4, 7, 0, 3, 4, 8, 0, 3, 4];
        assert_eq!(drafter.draft(&history, 2), vec![8, 0]);
    }

    #[test]
    fn ngram_returns_empty_when_no_repeat_exists() {
        let drafter = NGramDrafter::default();
        assert!(drafter.draft(&[1, 2, 3, 4, 5], 4).is_empty());
        assert!(drafter.draft(&[1, 2], 4).is_empty());
        assert!(drafter.draft(&[], 4).is_empty());
    }

    #[test]
    fn ngram_caps_at_requested_tokens() {
        let drafter = NGramDrafter::new(2, 3);
        let history = vec![1, 2, 9, 8, 7, 6, 1, 2];
        assert_eq!(drafter.draft(&history, 2), vec![9, 8]);
        assert_eq!(drafter.draft(&history, 10), vec![9, 8, 7, 6, 1, 2]);
    }

    #[test]
    fn ngram_constructor_normalizes_bounds() {
        let drafter = NGramDrafter::new(2, 5);
        assert_eq!(drafter.min_ngram, 2);
        assert_eq!(drafter.max_ngram, 5);
        assert_eq!(drafter.draft(&[1, 2, 9, 8, 7, 6, 1, 2], 3), vec![9, 8, 7]);

        let zero = NGramDrafter::new(0, 0);
        assert_eq!(zero.min_ngram, 1);
        assert_eq!(zero.max_ngram, 1);

        let inverted = NGramDrafter::new(5, 2);
        assert_eq!(inverted.min_ngram, 5);
        assert_eq!(inverted.max_ngram, 5);
    }

    fn reference_ngram_draft(
        history: &[u32],
        min_ngram: usize,
        max_ngram: usize,
        max_tokens: usize,
    ) -> Vec<u32> {
        if max_tokens == 0 || history.len() <= min_ngram {
            return Vec::new();
        }
        let len = history.len();
        let max_n = max_ngram.min(len.saturating_sub(1));
        for n in (min_ngram..=max_n).rev() {
            let pattern = &history[len - n..];
            for start in (0..len - n).rev() {
                if &history[start..start + n] == pattern {
                    let continuation_start = start + n;
                    let continuation_end = (continuation_start + max_tokens).min(len);
                    if continuation_start < continuation_end {
                        return history[continuation_start..continuation_end].to_vec();
                    }
                    break;
                }
            }
        }
        Vec::new()
    }

    #[test]
    fn indexed_ngram_matches_reference_scanner() {
        let drafter = NGramDrafter::new_with_index_capacity(2, 6, 1_000_000);
        let mut history = Vec::new();
        let mut state = 0x9e37_79b9u32;
        for step in 0..1200usize {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Small alphabet creates matches; occasional copied runs exercise
            // longest-match and most-recent tie-breaking.
            let token = if step > 32 && step % 17 < 5 {
                history[step - 17]
            } else {
                state % 23
            };
            history.push(token);
            for max_tokens in [1usize, 3, 7] {
                assert_eq!(
                    drafter.draft(&history, max_tokens),
                    reference_ngram_draft(&history, 2, 6, max_tokens),
                    "indexed/reference mismatch at history length {}",
                    history.len()
                );
            }
        }
        let stats = drafter.index_stats();
        assert_eq!(stats.indexed_tokens, history.len());
        assert_eq!(
            stats.rebuilds, 0,
            "append-only history must stay incremental"
        );
    }

    #[test]
    fn ngram_index_is_bounded_and_survives_rollback() {
        let drafter = NGramDrafter::new_with_index_capacity(2, 5, 48);
        let mut history: Vec<u32> = (0..400).map(|i| (i % 19) as u32).collect();
        let _ = drafter.draft(&history, 5);
        assert!(drafter.index_stats().records <= 48);

        history.truncate(220);
        let indexed = drafter.draft(&history, 5);
        let reference = reference_ngram_draft(&history, 2, 5, 5);
        // The bounded index may intentionally miss an evicted old pattern,
        // but any proposal it does return must retain exact scan semantics.
        assert!(indexed.is_empty() || indexed == reference);
        let stats = drafter.index_stats();
        assert_eq!(stats.indexed_tokens, history.len());
        assert!(stats.records <= 48);
        assert_eq!(stats.rebuilds, 1);

        history.extend([7, 8, 9, 7, 8, 9]);
        let _ = drafter.draft(&history, 3);
        assert_eq!(
            drafter.index_stats().rebuilds,
            1,
            "append after rollback rebuild must be incremental"
        );
    }

    #[test]
    #[ignore = "microbenchmark; run explicitly with --release --nocapture"]
    fn indexed_ngram_microbench_against_reference_scan() {
        use std::{hint::black_box, time::Instant};

        const BASE_TOKENS: usize = 16_384;
        const ROUNDS: usize = 512;
        let appended: Vec<u32> = (BASE_TOKENS..BASE_TOKENS + ROUNDS)
            .map(|token| token as u32)
            .collect();

        let drafter = NGramDrafter::new_with_index_capacity(3, 4, 100_000);
        let mut indexed_history: Vec<u32> = (0..BASE_TOKENS as u32).collect();
        assert!(drafter.draft(&indexed_history, 5).is_empty());
        let indexed_start = Instant::now();
        for token in &appended {
            indexed_history.push(*token);
            black_box(drafter.draft(black_box(&indexed_history), 5));
        }
        let indexed_elapsed = indexed_start.elapsed();

        let mut reference_history: Vec<u32> = (0..BASE_TOKENS as u32).collect();
        let reference_start = Instant::now();
        for token in &appended {
            reference_history.push(*token);
            black_box(reference_ngram_draft(
                black_box(&reference_history),
                3,
                4,
                5,
            ));
        }
        let reference_elapsed = reference_start.elapsed();
        println!(
            "ngram lookup: indexed={indexed_elapsed:?}, scan={reference_elapsed:?}, speedup={:.2}x",
            reference_elapsed.as_secs_f64() / indexed_elapsed.as_secs_f64()
        );
    }

    #[test]
    fn accepted_prefix_counts_matches_until_first_divergence() {
        assert_eq!(accepted_draft_prefix(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(accepted_draft_prefix(&[1, 2, 3], &[1, 9, 3, 4]), 1);
        assert_eq!(accepted_draft_prefix(&[1, 2, 3], &[9, 9, 9, 9]), 0);
        assert_eq!(accepted_draft_prefix(&[], &[5]), 0);
    }

    #[test]
    fn spec_latch_warmup_always_speculates() {
        let latch = SpecLatch::default();
        assert!(latch.should_speculate());
    }

    #[test]
    fn spec_latch_exits_after_run_of_nonproductive_rounds() {
        let mut latch = SpecLatch::default();
        // Warm-up round (counts as verified).
        latch.note_verified(0);
        // Three more non-productive rounds reach exit_run = 4.
        for _ in 0..2 {
            latch.note_verified(1);
            assert!(latch.should_speculate(), "run not yet complete");
        }
        latch.note_verified(0);
        assert!(
            !latch.speculating(),
            "4 consecutive non-productive rounds must latch off"
        );
        assert!(!latch.should_speculate());
    }

    #[test]
    fn spec_latch_productive_round_resets_the_run() {
        let mut latch = SpecLatch::default();
        latch.note_verified(0);
        latch.note_verified(1);
        latch.note_verified(0);
        // Productive round resets before the run reaches exit_run.
        latch.note_verified(3);
        latch.note_verified(0);
        latch.note_verified(0);
        latch.note_verified(1);
        assert!(
            latch.speculating(),
            "run must restart after a productive round"
        );
    }

    #[test]
    fn spec_latch_reprobe_after_low_reprobe_skips_and_reenters() {
        let mut latch = SpecLatch::default();
        for _ in 0..4 {
            latch.note_verified(0);
        }
        assert!(!latch.should_speculate());
        for _ in 0..63 {
            latch.note_skip();
            assert!(!latch.should_speculate());
        }
        latch.note_skip();
        assert!(latch.should_speculate(), "64th skip earns a re-probe");
        // Probe lands >= enter_drafts: re-latch ON.
        latch.note_verified(2);
        assert!(latch.speculating());
        assert!(latch.should_speculate());
    }

    #[test]
    fn spec_latch_failed_reprobe_stays_off() {
        let mut latch = SpecLatch::default();
        for _ in 0..4 {
            latch.note_verified(0);
        }
        for _ in 0..64 {
            latch.note_skip();
        }
        assert!(latch.should_speculate());
        latch.note_verified(0);
        assert!(!latch.speculating());
        assert!(!latch.should_speculate(), "failed probe resumes skipping");
    }
}
