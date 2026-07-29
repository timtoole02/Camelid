//! Quality-tier instrument (Lane B): in-Rust perplexity scoring.
//!
//! So lossy formats (Q4_K, Q5_K, …) can ship HONESTLY, we need a numeric
//! fidelity measurement that does not depend on a GPU and whose methodology is
//! validated against the established comparator (`llama-perplexity`). This
//! module provides:
//!
//! - [`negative_log_prob`] / [`Perplexity`]: streaming negative-log-likelihood
//!   accumulation in `f64` over a model's per-position logits, exponentiated to
//!   a perplexity. Pure math, unit-testable with synthetic logits (no model).
//! - [`PerplexityConvention`]: documents and pins the scoring convention
//!   (stride, BOS, which positions are scored) so it mirrors `llama-perplexity`.
//!
//! NOT provided: a format-routing policy. An earlier version of this doc listed a
//! `routing` submodule ("prefer Q5_K for small quant-sensitive models when the
//! Q4_K perplexity delta is too large") and the module declared `pub mod routing;`
//! for a `routing.rs` that was never written and never tracked. Combined with the
//! module never being declared in `lib.rs`, that made the declaration invisible
//! rather than a build error. Both claims are now removed; if a routing policy is
//! wanted, it should be added with its own evidence rather than re-declared here.
//!
//! SCOPE: this instrument computes perplexity from logits you hand it. It has no
//! model-driving loop of its own yet, so it does not by itself replace the pinned
//! external `llama-perplexity` comparator — see `basalt_eval_protocol.md`.

/// The perplexity scoring convention, pinned to mirror `llama-perplexity`.
///
/// `llama-perplexity` tokenizes the corpus once (prepending BOS), then slides a
/// window of `n_ctx` tokens with stride `n_ctx/2` by default. Within each
/// window it scores the SECOND half of positions (the first half is context
/// only, except the very first window which scores from position 1), summing
/// `-log p(token_i | token_{<i})` and dividing by the count of scored tokens;
/// PPL = exp(mean NLL). We pin the same convention so a Camelid PPL is
/// comparable to a `llama-perplexity` number byte-for-byte in methodology.
///
/// The token at position 0 is never scored (it has no preceding context); BOS
/// is prepended exactly once at the corpus start, not per window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerplexityConvention {
    /// Context window length (tokens per forward window). `0` means "score the
    /// whole sequence in one window" (used by the simple full-sequence path).
    pub n_ctx: usize,
    /// Sliding stride between windows. `llama-perplexity` defaults to `n_ctx/2`.
    pub stride: usize,
    /// Prepend a single BOS token at the corpus start.
    pub add_bos: bool,
}

impl PerplexityConvention {
    /// The simple single-window convention: score every position from 1 to the
    /// end of the (BOS-prefixed) sequence. Equivalent to `llama-perplexity`
    /// over a corpus that fits in one context window.
    pub fn full_sequence(add_bos: bool) -> Self {
        Self {
            n_ctx: 0,
            stride: 0,
            add_bos,
        }
    }

    /// The sliding-window convention mirroring `llama-perplexity --ctx-size n`.
    pub fn sliding(n_ctx: usize, add_bos: bool) -> Self {
        Self {
            n_ctx,
            stride: (n_ctx / 2).max(1),
            add_bos,
        }
    }
}

/// `-log p(target | logits)` under a numerically-stable f64 log-softmax.
///
/// Computed as `log_sum_exp(logits) - logits[target]`, all in f64 with the
/// max-subtraction trick so it never overflows. `logits` are the raw
/// pre-softmax scores for the full vocabulary; `target` indexes the actual next
/// token.
pub fn negative_log_prob(logits: &[f32], target: usize) -> f64 {
    debug_assert!(target < logits.len(), "target token out of vocab range");
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum_exp = 0.0f64;
    for &l in logits {
        sum_exp += ((l as f64) - max).exp();
    }
    let log_sum_exp = max + sum_exp.ln();
    let target_logit = logits[target] as f64;
    log_sum_exp - target_logit // = -log_softmax(logits)[target]
}

/// Streaming perplexity accumulator: folds per-position negative log-likelihood
/// in f64 and exponentiates the mean. Order-independent (summation is the only
/// reduction), so it matches whether positions are scored in one window or many.
#[derive(Debug, Clone, Default)]
pub struct Perplexity {
    sum_nll: f64,
    scored: usize,
}

impl Perplexity {
    pub fn new() -> Self {
        Self::default()
    }

    /// Score one position: add `-log p(target | logits)`.
    pub fn observe(&mut self, logits: &[f32], target: usize) {
        self.sum_nll += negative_log_prob(logits, target);
        self.scored += 1;
    }

    /// Number of scored positions.
    pub fn scored(&self) -> usize {
        self.scored
    }

    /// Mean negative log-likelihood (natural log) over scored positions.
    pub fn mean_nll(&self) -> f64 {
        if self.scored == 0 {
            f64::NAN
        } else {
            self.sum_nll / self.scored as f64
        }
    }

    /// Perplexity = exp(mean NLL).
    pub fn perplexity(&self) -> f64 {
        self.mean_nll().exp()
    }
}

/// A small JSON-serializable perplexity result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PerplexityReport {
    pub model: String,
    pub format: String,
    /// Scored token positions.
    pub scored_tokens: usize,
    /// Mean negative log-likelihood (nats).
    pub mean_nll: f64,
    /// Perplexity = exp(mean_nll).
    pub perplexity: f64,
    /// `add_bos` setting used.
    pub add_bos: bool,
    /// Context window (0 = full sequence).
    pub n_ctx: usize,
}

/// Compare a Camelid PPL against a `llama-perplexity` PPL and decide whether the
/// METHODOLOGY agrees within tolerance. On the SAME Q8 model the two engines run
/// the same numbers, so their PPL must match tightly — this validates the
/// in-Rust scoring before any Q4_K delta computed by it is trusted.
pub fn ppl_methodology_agrees(camelid_ppl: f64, llama_ppl: f64, rel_tol: f64) -> bool {
    if !(camelid_ppl.is_finite() && llama_ppl.is_finite()) || llama_ppl <= 0.0 {
        return false;
    }
    ((camelid_ppl - llama_ppl).abs() / llama_ppl) <= rel_tol
}

#[cfg(test)]
mod tests {
    use super::*;

    /// log-softmax of a uniform logit vector: every token has log-prob -ln(V),
    /// so NLL = ln(V) and PPL = V.
    #[test]
    fn uniform_logits_give_vocab_size_perplexity() {
        let vocab = 8;
        let logits = vec![0.0f32; vocab];
        let mut ppl = Perplexity::new();
        for target in 0..vocab {
            ppl.observe(&logits, target);
        }
        let expected_nll = (vocab as f64).ln();
        assert!((ppl.mean_nll() - expected_nll).abs() < 1e-9);
        assert!((ppl.perplexity() - vocab as f64).abs() < 1e-6);
    }

    /// A one-hot-ish logit (huge value on the target) gives near-zero NLL ⇒ PPL ≈ 1.
    #[test]
    fn confident_correct_prediction_perplexity_near_one() {
        let mut logits = vec![0.0f32; 100];
        logits[42] = 50.0; // overwhelmingly confident
        let nll = negative_log_prob(&logits, 42);
        assert!(nll < 1e-6, "confident-correct NLL should be ~0, got {nll}");
    }

    /// negative_log_prob is numerically stable for large logits (no overflow).
    #[test]
    fn logprob_stable_for_large_logits() {
        let logits = vec![1000.0f32, 1000.0, 1000.0, 1000.0];
        let nll = negative_log_prob(&logits, 0);
        // Uniform over 4 ⇒ -log(1/4) = ln(4).
        assert!((nll - 4.0f64.ln()).abs() < 1e-9, "got {nll}");
    }

    /// Hand-computed two-token case: logits [ln 1, ln 3] = [0, 1.0986...].
    /// softmax = [1/4, 3/4]; NLL(target=1) = -ln(3/4).
    #[test]
    fn matches_hand_computed_softmax() {
        let logits = [0.0f32, 3.0f32.ln()];
        let nll = negative_log_prob(&logits, 1);
        // exp(0) : exp(ln 3) = 1 : 3, so p(target=1) = 3/4 and NLL = -ln(0.75).
        //
        // Tolerance is 1e-6, not 1e-9: the logits are f32, and `3.0f32.ln()`
        // rounds to 1.09861230850219727 against a true ln(3) of
        // 1.09861228866810970. That ~2e-8 input error propagates to ~5e-9 in the
        // result, so a 1e-9 bound is unsatisfiable by construction no matter how
        // exact the f64 arithmetic is. (This test had never been executed — the
        // module was never declared in lib.rs — so the impossible bound went
        // unnoticed. The function itself is correct.)
        assert!((nll - (-(0.75f64).ln())).abs() < 1e-6, "got {nll}");
    }

    /// Order independence: scoring the same positions in a different order
    /// yields the same perplexity (summation reduction is associative in f64
    /// to within rounding, and we assert exact-equality here since it's the
    /// same set of adds).
    #[test]
    fn perplexity_is_scoring_order_independent() {
        let cases: &[(Vec<f32>, usize)] = &[
            (vec![0.1, 0.2, 0.3, 0.4], 2),
            (vec![1.0, -1.0, 0.5, 0.0], 0),
            (vec![3.0, 3.0, 0.0, 0.0], 3),
        ];
        let mut a = Perplexity::new();
        for (l, t) in cases {
            a.observe(l, *t);
        }
        let mut b = Perplexity::new();
        for (l, t) in cases.iter().rev() {
            b.observe(l, *t);
        }
        assert_eq!(a.scored(), b.scored());
        // Sum of the same f64 terms in reverse order: equal here (same values).
        assert!((a.perplexity() - b.perplexity()).abs() < 1e-12);
    }

    #[test]
    fn methodology_agreement_tolerance() {
        // Same-model Q8 PPLs should agree tightly.
        assert!(ppl_methodology_agrees(7.123, 7.120, 0.01));
        // A gross disagreement is rejected.
        assert!(!ppl_methodology_agrees(7.1, 9.0, 0.01));
        // Non-finite / nonpositive baselines are rejected.
        assert!(!ppl_methodology_agrees(f64::NAN, 7.0, 0.01));
        assert!(!ppl_methodology_agrees(7.0, 0.0, 0.01));
    }

    /// Fixture-based end-to-end of the accumulator over a small synthetic
    /// "model output": a sequence of per-position logit rows + the actual next
    /// tokens, with the expected PPL computed independently here.
    #[test]
    fn fixture_sequence_perplexity() {
        // 3 scored positions, vocab 4.
        let rows: &[(Vec<f32>, usize)] = &[
            (vec![2.0, 1.0, 0.0, 0.0], 0),
            (vec![0.0, 0.0, 4.0, 0.0], 2),
            (vec![1.0, 1.0, 1.0, 1.0], 3),
        ];
        let mut ppl = Perplexity::new();
        let mut expected_sum = 0.0f64;
        for (l, t) in rows {
            ppl.observe(l, *t);
            // independent reference
            let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = max
                + l.iter()
                    .map(|&x| ((x as f64) - max).exp())
                    .sum::<f64>()
                    .ln();
            expected_sum += lse - l[*t] as f64;
        }
        assert_eq!(ppl.scored(), 3);
        let expected_ppl = (expected_sum / 3.0).exp();
        assert!((ppl.perplexity() - expected_ppl).abs() < 1e-9);
    }
}
