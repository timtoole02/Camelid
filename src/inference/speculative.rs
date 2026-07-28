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
#[derive(Debug, Clone)]
pub struct NGramDrafter {
    pub max_ngram: usize,
    pub min_ngram: usize,
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
        Self {
            min_ngram: min_ngram.max(1),
            max_ngram: max_ngram.max(min_ngram.max(1)),
        }
    }

    pub fn draft(&self, history: &[u32], max_tokens: usize) -> Vec<u32> {
        if max_tokens == 0 || self.min_ngram == 0 || history.len() <= self.min_ngram {
            return Vec::new();
        }
        let len = history.len();
        let max_n = self.max_ngram.min(len.saturating_sub(1));
        for n in (self.min_ngram..=max_n).rev() {
            let pattern = &history[len - n..];
            // Most recent earlier occurrence; the window at len-n is the
            // suffix itself and is excluded.
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
        let drafter = NGramDrafter {
            max_ngram: 3,
            min_ngram: 2,
        };
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
