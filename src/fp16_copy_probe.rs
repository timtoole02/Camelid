// CLI-only causal source-copy experiment. The proposer receives source-derived
// token candidates and committed output only. Reference generation and oracle
// comparisons belong to the caller and cannot enter this interface.

#[derive(Clone, Debug)]
struct Fp16CopyBank {
    candidates: Vec<Vec<u32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fp16CopyAlignment {
    Prefix,
    Suffix,
    Unmatched,
}

impl Fp16CopyAlignment {
    fn label(self) -> &'static str {
        match self {
            Self::Prefix => "source_prefix",
            Self::Suffix => "committed_suffix",
            Self::Unmatched => "no_unique_continuation",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Fp16CopyDraft {
    tokens: Vec<u32>,
    alignment: Fp16CopyAlignment,
    matched_tokens: usize,
    matching_continuations: usize,
}

impl Fp16CopyBank {
    fn from_candidates(candidates: impl IntoIterator<Item = Vec<u32>>) -> Self {
        let mut unique = Vec::new();
        for candidate in candidates {
            if !candidate.is_empty() && !unique.contains(&candidate) {
                unique.push(candidate);
            }
        }
        Self { candidates: unique }
    }

    /// A source-only bank. No prompt target, reference, observed future, or
    /// transformed token stream can be supplied through another parameter.
    fn from_source(tokenizer: &Tokenizer, source: &str, language: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !language.contains(['\n', '\r', '`']),
            "invalid code-fence language"
        );
        let separator = if source.ends_with('\n') { "" } else { "\n" };
        let fenced = format!("```{language}\n{source}{separator}```");
        // Two variants reproduce the fixed raw/fenced policy used by the
        // earlier source-only replay. Four remains the default. This choice
        // is independent of target output; every offered token is verified.
        let variants = match std::env::var("CAMELID_BENCH_FP16_COPY_VARIANTS") {
            Ok(value) if value == "2" => vec![source.to_owned(), fenced],
            Ok(value) if value != "4" => anyhow::bail!("copy variants must be2 or4"),
            Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!("invalid copy variants"),
            _ => vec![source.to_owned(), format!("{source}\n"), fenced.clone(), format!("{fenced}\n")],
        };
        let candidates = variants
            .iter()
            .map(|text| tokenizer.encode(text, false, false))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_candidates(candidates))
    }

    /// Propose at most `budget` tokens using only committed output. The full
    /// generated prefix may align from its first known token. If a model emits
    /// a preamble, later 4..32-token suffix matches can recover alignment.
    /// Ambiguous continuations are restricted to their common prefix; an empty
    /// continuation participates in the ambiguity check instead of guessing
    /// past the end of a source variant.
    fn propose(&self, committed: &[u32], budget: usize) -> Fp16CopyDraft {
        let empty = || Fp16CopyDraft {
            tokens: Vec::new(),
            alignment: Fp16CopyAlignment::Unmatched,
            matched_tokens: 0,
            matching_continuations: 0,
        };
        if committed.is_empty() || budget == 0 {
            return empty();
        }
        let prefixes: Vec<&[u32]> = self
            .candidates
            .iter()
            .filter(|candidate| candidate.starts_with(committed))
            .map(|candidate| &candidate[committed.len()..])
            .collect();
        if !prefixes.is_empty() {
            return Fp16CopyDraft {
                tokens: fp16_copy_common_prefix(&prefixes, budget),
                alignment: Fp16CopyAlignment::Prefix,
                matched_tokens: committed.len(),
                matching_continuations: prefixes.len(),
            };
        }
        for matched in (4..=committed.len().min(32)).rev() {
            let suffix = &committed[committed.len() - matched..];
            let mut continuations = Vec::new();
            for candidate in &self.candidates {
                for (start, window) in candidate.windows(matched).enumerate() {
                    if window == suffix {
                        continuations.push(&candidate[start + matched..]);
                    }
                }
            }
            if !continuations.is_empty() {
                // A shorter suffix has a superset of these matches and cannot
                // resolve an ambiguity found at this longest matching suffix.
                return Fp16CopyDraft {
                    tokens: fp16_copy_common_prefix(&continuations, budget),
                    alignment: Fp16CopyAlignment::Suffix,
                    matched_tokens: matched,
                    matching_continuations: continuations.len(),
                };
            }
        }
        empty()
    }
}

fn fp16_copy_common_prefix(continuations: &[&[u32]], budget: usize) -> Vec<u32> {
    let Some(first) = continuations.first() else {
        return Vec::new();
    };
    let mut count = first.len().min(budget);
    for continuation in &continuations[1..] {
        count = first[..count]
            .iter()
            .zip(*continuation)
            .take_while(|(a, b)| a == b)
            .count();
    }
    first[..count].to_vec()
}

struct Fp16CopyAcceptance {
    emitted: Vec<u32>,
    matching_drafts: usize,
    committed_drafts: usize,
    reached_eos: bool,
}

/// With inputs [anchor,drafts...] the target prediction at row j follows input
/// j. Keep matching drafts then the first authoritative correction/bonus.
/// `emitted.len()` also gives the number of input KV rows to retain: the final
/// emitted token remains the uncached anchor for the next round. This stays
/// true when EOS or the output budget truncates an otherwise accepted window.
fn fp16_copy_accept(
    drafts: &[u32],
    predictions: &[u32],
    remaining: usize,
    is_eos: impl Fn(u32) -> bool,
) -> Option<Fp16CopyAcceptance> {
    if remaining == 0 || predictions.len() != drafts.len() + 1 {
        return None;
    }
    let matching = drafts
        .iter()
        .zip(predictions)
        .take_while(|(draft, prediction)| draft == prediction)
        .count();
    let mut emitted = drafts[..matching].to_vec();
    emitted.push(predictions[matching]);
    emitted.truncate(remaining);
    if let Some(index) = emitted.iter().position(|&token| is_eos(token)) {
        emitted.truncate(index + 1);
    }
    let reached_eos = emitted.last().is_some_and(|&token| is_eos(token));
    Some(Fp16CopyAcceptance {
        committed_drafts: matching.min(emitted.len()),
        emitted,
        matching_drafts: matching,
        reached_eos,
    })
}

#[derive(Debug, serde::Serialize)]
struct Fp16CopyRoundStats {
    input_rows: usize,
    proposed_drafts: usize,
    matching_drafts: usize,
    committed_drafts: usize,
    emitted_tokens: usize,
    rolled_back_rows: usize,
    alignment: &'static str,
    matched_tokens: usize,
    matching_continuations: usize,
    proposal_ms: f64,
    verify_ms: f64,
}

#[derive(Debug, serde::Serialize)]
struct Fp16CopyProbeStats {
    arithmetic: &'static str,
    proposer: &'static str,
    reference_available_to_proposer: bool,
    proposal_source_sha256: String,
    candidate_token_counts: Vec<usize>,
    max_rows: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    emitted_tokens_excluding_initial_anchor: usize,
    proposal_setup_ms: f64,
    prefill_ms: f64,
    generation_ms: f64,
    emitted_tokens_per_second: f64,
    proposal_ms: f64,
    verify_ms: f64,
    other_generation_ms: f64,
    verified_rows: usize,
    proposed_drafts: usize,
    matching_drafts: usize,
    committed_drafts: usize,
    rolled_back_rows: usize,
    projection_dispatches: u64,
    logical_projection_count: u64,
    stop_reason: &'static str,
    rounds: Vec<Fp16CopyRoundStats>,
}

#[derive(Debug, serde::Serialize)]
struct Fp16CopyProbeResult {
    generated_ids: Vec<u32>,
    stats: Fp16CopyProbeStats,
}

/// Actual causal generation, not a perfect-draft timing probe. Proposal
/// tokenization and prompt prefill are reported separately. Generation timing
/// includes source lookup, input assembly, GPU verification, acceptance, and KV
/// rollback. The already returned prompt prediction is excluded from its rate.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_fp16_copy_probe(
    session: &LlamaInferenceSession,
    probe: &mut camelid::metal::ResidentFp16Probe,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    proposal_text: &str,
    language: &str,
    max_rows: usize,
    max_tokens: usize,
) -> anyhow::Result<Fp16CopyProbeResult> {
    anyhow::ensure!(
        (1..=64).contains(&max_rows),
        "causal FP16 width must be1..64"
    );
    anyhow::ensure!(
        !prompt.is_empty() && max_tokens > 0,
        "nonempty prompt/output budget required"
    );
    anyhow::ensure!(
        prompt
            .len()
            .checked_add(max_tokens)
            .is_some_and(|n| n <= probe.max_positions()),
        "causal FP16 context budget exceeds probe capacity"
    );
    let setup_started = std::time::Instant::now();
    let bank = Fp16CopyBank::from_source(tokenizer, proposal_text, language)?;
    let proposal_setup_ms = setup_started.elapsed().as_secs_f64() * 1000.0;
    let prefill_started = std::time::Instant::now();
    probe.reset();
    let mut first = None;
    for chunk in prompt.chunks(64) {
        let predictions = session
            .forward_fp16_probe_tokens(probe, chunk)?
            .ok_or_else(|| anyhow::anyhow!("causal FP16 prompt left diagnostic route"))?;
        anyhow::ensure!(
            predictions.len() == chunk.len(),
            "causal FP16 prompt shape mismatch"
        );
        first = predictions.last().copied();
    }
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let mut generated =
        vec![first.ok_or_else(|| anyhow::anyhow!("causal FP16 prompt has no prediction"))?];
    let mut reached_eos = tokenizer.special.eog.contains(&generated[0]);
    let before = probe.projection_count();
    let matrix_before = probe.actual_matrix_dispatch_count();
    let mut rounds = Vec::new();
    let generation_started = std::time::Instant::now();
    while generated.len() < max_tokens && !reached_eos {
        let old_filled = probe.filled();
        let remaining = max_tokens - generated.len();
        let row_budget = max_rows
            .min(remaining)
            .min(probe.max_positions() - old_filled);
        anyhow::ensure!(row_budget > 0, "causal FP16 context exhausted");
        let proposal_started = std::time::Instant::now();
        let draft = bank.propose(&generated, row_budget - 1);
        let mut inputs = Vec::with_capacity(1 + draft.tokens.len());
        inputs.push(*generated.last().expect("initial anchor present"));
        inputs.extend_from_slice(&draft.tokens);
        let proposal_ms = proposal_started.elapsed().as_secs_f64() * 1000.0;
        let verify_started = std::time::Instant::now();
        let predictions = session
            .forward_fp16_probe_tokens(probe, &inputs)?
            .ok_or_else(|| anyhow::anyhow!("causal FP16 verification left diagnostic route"))?;
        let verify_ms = verify_started.elapsed().as_secs_f64() * 1000.0;
        anyhow::ensure!(
            predictions.len() == inputs.len(),
            "causal FP16 prediction shape mismatch"
        );
        let acceptance = fp16_copy_accept(&draft.tokens, &predictions, remaining, |token| {
            tokenizer.special.eog.contains(&token)
        })
        .ok_or_else(|| anyhow::anyhow!("invalid causal FP16 acceptance window"))?;
        let kept = acceptance.emitted.len();
        anyhow::ensure!(
            probe.truncate(old_filled + kept),
            "causal FP16 rollback attempted to advance KV"
        );
        reached_eos = acceptance.reached_eos;
        generated.extend_from_slice(&acceptance.emitted);
        rounds.push(Fp16CopyRoundStats {
            input_rows: inputs.len(),
            proposed_drafts: draft.tokens.len(),
            matching_drafts: acceptance.matching_drafts,
            committed_drafts: acceptance.committed_drafts,
            emitted_tokens: kept,
            rolled_back_rows: inputs.len() - kept,
            alignment: draft.alignment.label(),
            matched_tokens: draft.matched_tokens,
            matching_continuations: draft.matching_continuations,
            proposal_ms,
            verify_ms,
        });
    }
    let generation_ms = generation_started.elapsed().as_secs_f64() * 1000.0;
    let emitted = generated.len() - 1;
    let proposal_ms = rounds.iter().map(|r| r.proposal_ms).sum::<f64>();
    let verify_ms = rounds.iter().map(|r| r.verify_ms).sum::<f64>();
    let projections = probe.projection_count() - before;
    let matrices = probe.actual_matrix_dispatch_count() - matrix_before;
    anyhow::ensure!(
        projections == 197 * rounds.len() as u64,
        "causal FP16 route count mismatch"
    );
    anyhow::ensure!(matrices == (if probe.gate_up_fused() {169} else {197}) * rounds.len() as u64,
        "causal FP16 matrix dispatch count mismatch");
    let stats=Fp16CopyProbeStats {
        arithmetic:"same isolated canonical-half/native-MMA target as FP16 serial reference",
        proposer:"source-only raw/fenced bank; committed-prefix/common-continuation; unique4..32-token suffix",
        reference_available_to_proposer:false,
        proposal_source_sha256:camelid::receipt::sha256_hex(proposal_text.as_bytes()),
        candidate_token_counts:bank.candidates.iter().map(Vec::len).collect(),
        max_rows,prompt_tokens:prompt.len(),generated_tokens:generated.len(),emitted_tokens_excluding_initial_anchor:emitted,
        proposal_setup_ms,prefill_ms,generation_ms,
        emitted_tokens_per_second:if generation_ms>0.0 {emitted as f64*1000.0/generation_ms} else {0.0},
        proposal_ms,verify_ms,other_generation_ms:(generation_ms-proposal_ms-verify_ms).max(0.0),
        verified_rows:rounds.iter().map(|r|r.input_rows).sum(),
        proposed_drafts:rounds.iter().map(|r|r.proposed_drafts).sum(),
        matching_drafts:rounds.iter().map(|r|r.matching_drafts).sum(),
        committed_drafts:rounds.iter().map(|r|r.committed_drafts).sum(),
        rolled_back_rows:rounds.iter().map(|r|r.rolled_back_rows).sum(),projection_dispatches:matrices,
        logical_projection_count:projections,
        stop_reason:if reached_eos {"eos"} else {"max_tokens"},rounds,
    };
    Ok(Fp16CopyProbeResult {
        generated_ids: generated,
        stats,
    })
}

#[cfg(test)]
mod fp16_copy_tests {
    use super::*;

    #[test]
    fn proposer_uses_only_committed_tokens_and_source() {
        let bank =
            Fp16CopyBank::from_candidates([vec![10, 20, 30, 40, 50], vec![10, 20, 30, 40, 50, 60]]);
        let future_a = [10, 20, 30, 999];
        let future_b = [10, 77, 88, 99];
        assert_eq!(
            bank.propose(&future_a[..1], 63),
            bank.propose(&future_b[..1], 63)
        );
        assert_eq!(
            bank.propose(&future_a[..1], 63).tokens,
            vec![20, 30, 40, 50]
        );
        assert!(bank.propose(&[], 63).tokens.is_empty());
        assert!(bank.propose(&[77], 63).tokens.is_empty());
        assert_eq!(bank.propose(&[10], 2).tokens, vec![20, 30]);
    }

    #[test]
    fn ambiguous_suffixes_only_emit_common_continuation() {
        let bank = Fp16CopyBank::from_candidates([vec![1, 2, 3, 4, 5, 7], vec![1, 2, 3, 4, 5, 8]]);
        let draft = bank.propose(&[99, 1, 2, 3, 4], 63);
        assert_eq!(draft.alignment, Fp16CopyAlignment::Suffix);
        assert_eq!(draft.tokens, vec![5]);
        assert!(bank.propose(&[99, 1, 2, 3, 4, 5], 63).tokens.is_empty());
        let ended = Fp16CopyBank::from_candidates([vec![1, 2, 3, 4], vec![1, 2, 3, 4, 5]]);
        assert!(ended.propose(&[99, 1, 2, 3, 4], 63).tokens.is_empty());
    }

    #[test]
    fn longest_suffix_recovers_after_preamble_and_deduplicates_wrappers() {
        let source = vec![7, 1, 2, 3, 4, 5, 8, 1, 2, 3, 4, 6];
        let bank = Fp16CopyBank::from_candidates([source.clone(), source]);
        let draft = bank.propose(&[99, 7, 1, 2, 3, 4], 3);
        assert_eq!(draft.matched_tokens, 5);
        assert_eq!(draft.tokens, vec![5, 8, 1]);
        assert_eq!(draft.matching_continuations, 1);
        assert!(bank.propose(&[99, 1, 2, 3, 4], 63).tokens.is_empty());
    }

    #[test]
    fn acceptance_keeps_anchor_and_matching_inputs_then_authoritative_token() {
        let accepted = fp16_copy_accept(&[20, 30, 40], &[20, 99, 77, 88], 10, |_| false).unwrap();
        assert_eq!(accepted.emitted, vec![20, 99]);
        assert_eq!(accepted.matching_drafts, 1);
        assert_eq!(accepted.emitted.len(), 2); // retain old anchor and accepted20, not correction99
        let full = fp16_copy_accept(&[20, 30], &[20, 30, 40], 10, |_| false).unwrap();
        assert_eq!(full.emitted, vec![20, 30, 40]);
        let eos = fp16_copy_accept(&[20, 30, 40], &[20, 30, 40, 50], 10, |t| t == 30).unwrap();
        assert_eq!(eos.emitted, vec![20, 30]);
        assert!(eos.reached_eos);
        assert_eq!(eos.emitted.len(), 2); // EOS remains the uncached terminal anchor
        let budget = fp16_copy_accept(&[20, 30, 40], &[20, 30, 40, 50], 1, |_| false).unwrap();
        assert_eq!(budget.emitted, vec![20]);
        assert!(fp16_copy_accept(&[], &[10], 0, |_| false).is_none());
    }
}
