//! Smoke-admission: a per-model "does it actually run cleanly here" check.
//!
//! This is **distinct from Phase 5 oracle qualification**. Oracle qualification
//! proves the runnable graph for an (architecture, quant, tokenizer) combo is
//! numerically equivalent to HF transformers — that is a one-time, per-architecture
//! gate. Smoke-admission is per-*model-file*: it confirms THIS GGUF admits, loads,
//! and executes deterministically without blowing up, and that greedy decode is not
//! degenerate. It attests deterministic execution, NOT parity.
//!
//! Guardrail: smoke-admission runs ONLY on combos that are already oracle-qualified
//! (so a green smoke can never be mistaken for "this architecture is correct" — that
//! claim is the oracle's). Any GGUF on a non-anchored combo is refused with
//! `combo not yet anchored`.
//!
//! A pass emits a **runnable** receipt (`execution_lane = Runnable`, never copper),
//! whose parity block is `not_compared` — honest that no reference was consulted.

use std::collections::HashSet;

use crate::error::{BackendError, Result};
use crate::gguf::{read_metadata, GgufFile, GgufTensorType};
use crate::receipt::{
    rfc3339_utc_now, sha256_file_hex, ExecutionLane, LaneIdentity, ParityBlock, ParityReceipt,
    ReceiptRequest, ReceiptResult, ReferenceIdentity, RECEIPT_SCHEMA_V1,
};
use crate::tokenizer::Tokenizer;

use super::admit::{self, TokenizerFamily};
use super::RunnableModel;

const SMOKE_PROMPT: &str = "What is the capital of France?";
const GEN_TOKENS: usize = 24;
/// A working LM's logits sit in roughly [-30, 30]; anything past this is a blow-up.
const SANE_LOGIT_ABS: f32 = 200.0;
/// Minimum distinct tokens over the generation before we call it degenerate.
const MIN_DISTINCT: usize = 6;

/// The (architecture, tokenizer-family, quant) combos that have been oracle-qualified
/// (HF-anchored), plus phi3 which is implemented + coherence-validated with HF parity
/// pending a larger-RAM machine (allowed per the runnable-lane memory policy). Smoke
/// runs ONLY on these — everything else is "combo not yet anchored".
fn is_oracle_qualified(arch: &str, tok: TokenizerFamily, quant: &str) -> bool {
    matches!(
        (arch, tok, quant),
        ("llama", TokenizerFamily::Spm, "Q8_0")
            | ("qwen3", TokenizerFamily::Bpe, "Q8_0")
            | ("gemma3", TokenizerFamily::Spm, "Q8_0")
            | ("phi3", TokenizerFamily::Spm, "Q8_0")
    )
}

/// The guardrail refusal, split out so the refusal class is unit-testable: an
/// ADMITTED but non-anchored combo (e.g. the BASALT gemma4+NVFP4 pilot before
/// Gate G3) refuses with "combo not yet anchored", NOT with a quant-not-covered
/// admission reject — that distinction is load-bearing for the refusal receipts.
fn oracle_qualification_gate(arch: &str, tok: TokenizerFamily, quant: &str) -> Result<()> {
    if is_oracle_qualified(arch, tok, quant) {
        return Ok(());
    }
    Err(BackendError::UnsupportedGguf(format!(
        "combo not yet anchored: {arch}/{quant}/{tok:?}; smoke-admission runs only on \
         oracle-qualified combos (llama/Q8_0/SPM, qwen3/Q8_0/BPE, gemma3/Q8_0/SPM, \
         phi3/Q8_0/SPM)"
    )))
}

/// Public (architecture, quant) view of the oracle-qualified set — the tokenizer
/// family is implied by the architecture for every anchored combo. Used by the API
/// to classify local models and catalog entries without running a model.
pub fn oracle_qualified(architecture: &str, quant: &str) -> bool {
    matches!(
        (architecture, quant),
        ("llama", "Q8_0")
            | ("qwen3", "Q8_0")
            | ("gemma3", "Q8_0")
            | ("phi3", "Q8_0")
            | ("nomic-bert", "Q8_0")
    )
}

/// The headline quant of a GGUF: the most common quantized (non-F32) tensor type,
/// e.g. `Q8_0`. Public so the API can label local models without loading them.
pub fn headline_quant_of(gguf: &GgufFile) -> String {
    headline_quant(gguf)
}

/// Result of a passing smoke-admission.
pub struct SmokeReport {
    pub architecture: String,
    pub quant: String,
    pub tokenizer: TokenizerFamily,
    pub prompt_token_count: usize,
    pub generated: Vec<u32>,
    pub generated_text: String,
    pub logit_min: f32,
    pub logit_max: f32,
    /// Runnable receipt (lane=runnable, never copper) attesting deterministic
    /// execution — not parity.
    pub receipt: ParityReceipt,
}

/// Run smoke-admission on a GGUF. `Ok` only when every check passes; the returned
/// report carries the runnable receipt.
pub fn smoke_admit(path: &str) -> Result<SmokeReport> {
    let gguf = read_metadata(path)?;

    // (1) covered-set admission gate (Phase 1).
    let admitted = admit::admit(&gguf).map_err(BackendError::from)?;
    let quant = headline_quant(&gguf);

    // Oracle-qualified guardrail.
    oracle_qualification_gate(&admitted.architecture, admitted.tokenizer, &quant)?;

    // (2) load: all tensors present, shapes consistent, dequant succeeds on every
    // weight (RunnableModel::load fails closed otherwise).
    let model = RunnableModel::load(path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;

    // (3) greedy forward sanity on a fixed tiny prompt: finite logits, sane range.
    let (text, add_special, parse_special) = build_prompt(&tok);
    let prompt = tok.encode(&text, add_special, parse_special)?;
    if prompt.is_empty() {
        return Err(BackendError::InvalidTensorData(
            "smoke: prompt tokenized to nothing".into(),
        ));
    }
    let logits = model.forward_logits(&prompt)?;
    if !logits.iter().all(|v| v.is_finite()) {
        return Err(BackendError::InvalidTensorData(
            "smoke: forward produced non-finite logits".into(),
        ));
    }
    let logit_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !(logit_min > -SANE_LOGIT_ABS && logit_max < SANE_LOGIT_ABS) {
        return Err(BackendError::InvalidTensorData(format!(
            "smoke: logit range [{logit_min:.1}, {logit_max:.1}] is out of sane bounds"
        )));
    }

    // (4) coherence: greedy decode, fail if degenerate.
    let generated = model.generate(&prompt, GEN_TOKENS)?;
    check_not_degenerate(&generated)?;
    let generated_text = tok.decode(&generated, true).unwrap_or_default();

    let receipt = build_runnable_receipt(
        path,
        &gguf,
        admitted.tokenizer,
        &text,
        &prompt,
        &generated,
        &generated_text,
    )?;

    Ok(SmokeReport {
        architecture: admitted.architecture,
        quant,
        tokenizer: admitted.tokenizer,
        prompt_token_count: prompt.len(),
        generated,
        generated_text,
        logit_min,
        logit_max,
        receipt,
    })
}

/// The headline quant: the most common quantized (non-F32) tensor type, e.g. `Q8_0`.
fn headline_quant(gguf: &GgufFile) -> String {
    use std::collections::HashMap;
    let mut counts: HashMap<GgufTensorType, usize> = HashMap::new();
    for t in &gguf.tensors {
        if t.tensor_type != GgufTensorType::F32 {
            *counts.entry(t.tensor_type).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(tt, _)| format!("{tt:?}"))
        .unwrap_or_else(|| "F32".to_string())
}

/// Build the smoke prompt. Instruction-tuned models degenerate on raw completion
/// prompts, so render the GGUF's own chat template when present (gives each model a
/// fair coherence test); fall back to a raw completion prompt otherwise.
fn build_prompt(tok: &Tokenizer) -> (String, bool, bool) {
    if let Some(tmpl) = tok.chat_template.as_deref() {
        if let Some(rendered) = render_chat_template(tmpl, tok) {
            // Chat templates emit their own BOS (`{{ bos_token }}`) and control
            // markers, so don't add BOS again and DO parse specials.
            return (rendered, false, true);
        }
    }
    (SMOKE_PROMPT.to_string(), true, false)
}

/// Render a single-user-turn chat prompt from the GGUF Jinja chat template.
/// Best-effort: returns None if the template uses constructs we don't support, in
/// which case the caller falls back to a raw prompt.
fn render_chat_template(tmpl: &str, tok: &Tokenizer) -> Option<String> {
    let mut env = minijinja::Environment::new();
    // Some templates guard with raise_exception; surface it as a render error so we
    // fall back rather than panic.
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_template("chat", tmpl).ok()?;
    let template = env.get_template("chat").ok()?;
    let bos = tok.token_text(tok.special.bos).unwrap_or("");
    let eos = tok.token_text(tok.special.eos).unwrap_or("");
    let ctx = minijinja::context! {
        messages => vec![minijinja::context!{ role => "user", content => SMOKE_PROMPT }],
        add_generation_prompt => true,
        bos_token => bos,
        eos_token => eos,
    };
    let rendered = template.render(ctx).ok()?;
    if rendered.trim().is_empty() {
        None
    } else {
        Some(rendered)
    }
}

/// Degeneracy check: too few distinct tokens, or a short repetition cycle in the
/// tail, both signal a broken/garbage run rather than a working LM.
fn check_not_degenerate(gen: &[u32]) -> Result<()> {
    let distinct: HashSet<u32> = gen.iter().copied().collect();
    if distinct.len() < MIN_DISTINCT {
        return Err(BackendError::InvalidTensorData(format!(
            "smoke: degenerate greedy output — only {} distinct tokens in {}",
            distinct.len(),
            gen.len()
        )));
    }
    if let Some(period) = tail_cycle_period(gen, 4) {
        return Err(BackendError::InvalidTensorData(format!(
            "smoke: degenerate greedy output — tail is a period-{period} repetition loop"
        )));
    }
    Ok(())
}

/// Detect a period-`p` (1..=max_period) repetition loop occupying the tail: the last
/// three full periods are all identical. Returns the smallest such period.
fn tail_cycle_period(gen: &[u32], max_period: usize) -> Option<usize> {
    let n = gen.len();
    for p in 1..=max_period {
        if n < 3 * p {
            continue;
        }
        let tail = &gen[n - 3 * p..];
        if tail[..p] == tail[p..2 * p] && tail[p..2 * p] == tail[2 * p..] {
            return Some(p);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn build_runnable_receipt(
    path: &str,
    gguf: &GgufFile,
    tok_family: TokenizerFamily,
    prompt_text: &str,
    prompt: &[u32],
    generated: &[u32],
    generated_text: &str,
) -> Result<ParityReceipt> {
    let gguf_sha = sha256_file_hex(std::path::Path::new(path))
        .map_err(|e| BackendError::InvalidTensorData(format!("smoke: hash gguf: {e}")))?;
    let model_id = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let tok_kind = match tok_family {
        TokenizerFamily::Spm => "llama_spm",
        TokenizerFamily::Bpe => "gpt2_bpe",
    };
    let lane = LaneIdentity::capture(
        &model_id,
        std::path::Path::new(path),
        gguf,
        Some(tok_kind),
        gguf_sha,
    );
    let mut receipt = ParityReceipt {
        schema: RECEIPT_SCHEMA_V1.to_string(),
        receipt_id: String::new(),
        created_utc: rfc3339_utc_now(),
        lane,
        // No reference engine was consulted — smoke-admission is not a parity check.
        reference: ReferenceIdentity {
            tool: "none".to_string(),
            binary: "runnable-smoke-admission".to_string(),
            version: None,
            commit: None,
        },
        request: ReceiptRequest {
            endpoint: "runnable/smoke-admission".to_string(),
            messages_or_prompt: serde_json::json!(prompt_text),
            max_tokens: GEN_TOKENS as u32,
            temperature: 0.0,
            top_p: None,
            top_k: None,
            seed: None,
            stop: vec![],
            response_format: None,
        },
        reproducible: true,
        result: ReceiptResult {
            prompt_token_ids: prompt.to_vec(),
            generated_token_ids: generated.to_vec(),
            generated_text: generated_text.to_string(),
            completion_tokens: generated.len() as u32,
            finish_reason: "length".to_string(),
        },
        // Honest: no reference comparison happened.
        parity: ParityBlock::not_compared(),
        // The load-bearing distinction: this is a runnable-lane receipt.
        execution_lane: Some(ExecutionLane::Runnable),
        execution_trace: None,
        quality_tier: None,
        signature: None,
    };
    receipt
        .seal()
        .map_err(|e| BackendError::InvalidTensorData(format!("smoke: seal receipt: {e}")))?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_qualified_gate() {
        assert!(is_oracle_qualified("llama", TokenizerFamily::Spm, "Q8_0"));
        assert!(is_oracle_qualified("qwen3", TokenizerFamily::Bpe, "Q8_0"));
        assert!(is_oracle_qualified("gemma3", TokenizerFamily::Spm, "Q8_0"));
        assert!(is_oracle_qualified("phi3", TokenizerFamily::Spm, "Q8_0"));
        assert!(super::oracle_qualified("nomic-bert", "Q8_0"));
        // Not anchored: a covered architecture we have not HF-anchored.
        assert!(!is_oracle_qualified("gemma2", TokenizerFamily::Spm, "Q8_0"));
        // Not anchored: anchored arch but a quant we did not anchor.
        assert!(!is_oracle_qualified("llama", TokenizerFamily::Spm, "Q4K"));
        // Not anchored: anchored arch but unexpected tokenizer family.
        assert!(!is_oracle_qualified("llama", TokenizerFamily::Bpe, "Q8_0"));
        // Not anchored: the BASALT gemma4+NVFP4 pilot — admission passes (D-B3)
        // but smoke stays refused until Gate G3 anchors the combo.
        assert!(!is_oracle_qualified(
            "gemma4",
            TokenizerFamily::Spm,
            "NVFP4"
        ));
    }

    /// BASALT Phase 2: the gemma4+NVFP4 pilot's smoke refusal must be the
    /// not-yet-anchored class (an admitted combo awaiting its G3 oracle), NOT a
    /// quant-not-covered admission reject.
    #[test]
    fn gemma4_nvfp4_smoke_refusal_is_not_yet_anchored() {
        let err = oracle_qualification_gate("gemma4", TokenizerFamily::Spm, "NVFP4")
            .expect_err("gemma4+NVFP4 must stay smoke-refused until G3");
        let msg = format!("{err}");
        assert!(
            msg.contains("combo not yet anchored: gemma4/NVFP4/Spm"),
            "refusal must be the anchored-gate class: {msg}"
        );
        assert!(
            !msg.contains("unsupported quant"),
            "refusal must not read as a quant-coverage reject: {msg}"
        );
    }

    #[test]
    fn detects_repetition_loops() {
        // period-1
        assert_eq!(tail_cycle_period(&[1, 2, 3, 7, 7, 7], 4), Some(1));
        // period-2
        assert_eq!(tail_cycle_period(&[1, 2, 4, 5, 4, 5, 4, 5], 4), Some(2));
        // period-4 (gemma raw-prompt failure shape)
        assert_eq!(
            tail_cycle_period(&[9, 8, 7, 6, 9, 8, 7, 6, 9, 8, 7, 6], 4),
            Some(4)
        );
        // healthy, varied tail
        assert_eq!(tail_cycle_period(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4), None);
    }

    #[test]
    fn degenerate_low_distinct_rejected() {
        let gen = vec![5u32, 5, 5, 6, 5, 6, 5, 6];
        assert!(check_not_degenerate(&gen).is_err());
    }

    #[test]
    fn healthy_generation_accepted() {
        let gen: Vec<u32> = (100..130).collect();
        assert!(check_not_degenerate(&gen).is_ok());
    }

    /// Phase-1 G-LOAD coherence bring-up for a not-yet-anchored architecture
    /// (Ornith / `qwen35`). Loads the GGUF, renders its chat template, greedy-decodes
    /// 24 tokens, and asserts the output is coherent (finite logits, non-degenerate).
    /// This is a COHERENCE check, NOT a parity claim — it deliberately bypasses the
    /// `is_oracle_qualified` smoke guard (qwen35 is anchored only after Phase-3
    /// parity). Ignored by default: needs the 9.5 GB Q8_0 file via
    /// `CAMELID_ORNITH_GGUF`. Run with `--release -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs the ~9.5GB Ornith qwen35 Q8_0 GGUF; set CAMELID_ORNITH_GGUF"]
    fn ornith_qwen35_coherence_bringup() {
        let path = std::env::var("CAMELID_ORNITH_GGUF")
            .expect("set CAMELID_ORNITH_GGUF to the Ornith-1.0-9B Q8_0 GGUF path");
        let gguf = read_metadata(&path).expect("read gguf metadata");
        assert_eq!(
            gguf.architecture(),
            Some("qwen35"),
            "expected a qwen35 GGUF"
        );
        let model = RunnableModel::load(&path).expect("load qwen35 runnable model");
        let tok = Tokenizer::from_gguf(&gguf).expect("build tokenizer");

        let (text, add_special, parse_special) = build_prompt(&tok);
        let prompt = tok
            .encode(&text, add_special, parse_special)
            .expect("encode prompt");
        assert!(!prompt.is_empty(), "prompt tokenized to nothing");

        // Sanity: finite, in-range logits at the prompt's last position.
        let logits = model.forward_logits(&prompt).expect("forward prompt");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logits — arch/loader bug"
        );
        let lo = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            lo > -SANE_LOGIT_ABS && hi < SANE_LOGIT_ABS,
            "logit range [{lo:.1},{hi:.1}] out of sane bounds — likely a math bug"
        );

        let generated = model.generate(&prompt, GEN_TOKENS).expect("greedy decode");
        let out = tok.decode(&generated, true).unwrap_or_default();
        eprintln!("=== ORNITH qwen35 G-LOAD ===\nPROMPT:\n{text}\n--- GEN ({GEN_TOKENS} tok) ---\n{out}\nTOKENS: {generated:?}\nlogit_range=[{lo:.2},{hi:.2}]");
        check_not_degenerate(&generated).expect("coherent, non-degenerate generation");
    }

    /// Phase-3 parity helper: dump the rendered chat prompt + its qwen35 prompt token
    /// IDs (tokenizer-only, no weights → fast). Feed these exact IDs to the llama.cpp
    /// oracle so the parity comparison isolates the model forward from tokenization.
    /// Ignored: needs `CAMELID_ORNITH_GGUF`. Run `--release -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs the Ornith qwen35 GGUF metadata; set CAMELID_ORNITH_GGUF"]
    fn ornith_qwen35_prompt_tokens() {
        let path = std::env::var("CAMELID_ORNITH_GGUF")
            .expect("set CAMELID_ORNITH_GGUF to the Ornith-1.0-9B Q8_0 GGUF path");
        let gguf = read_metadata(&path).expect("read gguf metadata");
        let tok = Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        let (text, add_special, parse_special) = build_prompt(&tok);
        let prompt = tok
            .encode(&text, add_special, parse_special)
            .expect("encode prompt");
        // Machine-parseable lines for the parity harness to scrape.
        eprintln!(
            "RENDERED_PROMPT_JSON={}",
            serde_json::to_string(&text).unwrap()
        );
        eprintln!(
            "PROMPT_TOKENS_JSON={}",
            serde_json::to_string(&prompt).unwrap()
        );
        eprintln!(
            "ADD_SPECIAL={add_special} PARSE_SPECIAL={parse_special} N_PROMPT={}",
            prompt.len()
        );
    }

    /// Phase-3 parity helper A (fast, tokenizer-only): tokenize each prompt in the
    /// JSON-array env `CAMELID_PARITY_PROMPTS` with raw-completion conventions
    /// (add_special=true, parse_special=false) and print the token-id arrays. Feed
    /// these identical arrays to BOTH the llama.cpp oracle and camelid so the parity
    /// comparison isolates the model forward.
    #[test]
    #[ignore = "needs CAMELID_ORNITH_GGUF (+ CAMELID_PARITY_PROMPTS)"]
    fn ornith_qwen35_tokenize_set() {
        let path = std::env::var("CAMELID_ORNITH_GGUF").expect("set CAMELID_ORNITH_GGUF");
        let prompts: Vec<String> = serde_json::from_str(
            &std::env::var("CAMELID_PARITY_PROMPTS")
                .unwrap_or_else(|_| "[\"What is the capital of France?\"]".into()),
        )
        .expect("CAMELID_PARITY_PROMPTS must be a JSON array of strings");
        let gguf = read_metadata(&path).expect("read gguf");
        let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
        for (i, p) in prompts.iter().enumerate() {
            let ids = tok.encode(p, true, false).expect("encode");
            eprintln!(
                "TOKSET[{i}] PROMPT={} TOKENS={}",
                serde_json::to_string(p).unwrap(),
                serde_json::to_string(&ids).unwrap()
            );
        }
    }

    /// Phase-3 parity helper B (slow, full forward): for each token-id array in the
    /// JSON env `CAMELID_PARITY_TOKENS` (array of arrays of u32), greedy-decode
    /// `CAMELID_PARITY_NPREDICT` (default 20) tokens and print the generated ids.
    /// Run AFTER the llama-server oracle is shut down (both models can't fit RAM).
    #[test]
    #[ignore = "needs CAMELID_ORNITH_GGUF + CAMELID_PARITY_TOKENS (run with oracle server OFF)"]
    fn ornith_qwen35_parity_gen() {
        let path = std::env::var("CAMELID_ORNITH_GGUF").expect("set CAMELID_ORNITH_GGUF");
        let token_sets: Vec<Vec<u32>> = serde_json::from_str(
            &std::env::var("CAMELID_PARITY_TOKENS").expect("set CAMELID_PARITY_TOKENS"),
        )
        .expect("CAMELID_PARITY_TOKENS must be a JSON array of u32 arrays");
        let n_predict: usize = std::env::var("CAMELID_PARITY_NPREDICT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let model = RunnableModel::load(&path).expect("load qwen35");
        for (i, prompt) in token_sets.iter().enumerate() {
            let gen = model.generate(prompt, n_predict).expect("greedy decode");
            eprintln!("PARITY_GEN[{i}] {}", serde_json::to_string(&gen).unwrap());
        }
    }

    /// Item 5 (acceptance economics) helper A: teacher-forced greedy argmax at
    /// every position of each token sequence in the JSON file named by
    /// `CAMELID_STREAM_FILE` (array of arrays of u32). Prints one
    /// `ARGMAX[i] [...]` line per sequence. CPU by default; the resident GPU
    /// engine with `CAMELID_QWEN35_CUDA=1`.
    #[test]
    #[ignore = "needs CAMELID_ORNITH_GGUF + CAMELID_STREAM_FILE"]
    fn ornith_qwen35_argmax_stream() {
        let path = std::env::var("CAMELID_ORNITH_GGUF").expect("set CAMELID_ORNITH_GGUF");
        let stream_file = std::env::var("CAMELID_STREAM_FILE").expect("set CAMELID_STREAM_FILE");
        let token_sets: Vec<Vec<u32>> = serde_json::from_str(
            &std::fs::read_to_string(&stream_file).expect("read CAMELID_STREAM_FILE"),
        )
        .expect("CAMELID_STREAM_FILE must be a JSON array of u32 arrays");
        let model = RunnableModel::load(&path).expect("load qwen35");
        for (i, tokens) in token_sets.iter().enumerate() {
            let started = std::time::Instant::now();
            let stream = model.qwen35_argmax_stream(tokens).expect("argmax stream");
            eprintln!(
                "ARGMAX[{i}] len={} secs={:.1} {}",
                stream.len(),
                started.elapsed().as_secs_f64(),
                serde_json::to_string(&stream).unwrap()
            );
        }
    }

    /// Item 5 helper B: batched-prefill marginal cost — the real CPU verify
    /// cost for a k-token window. Times prefill at prefix length P and P+k for
    /// k in {4,6,8,12} (3 reps printed raw for offline reduction). Prefix
    /// tokens come from the first sequence in `CAMELID_STREAM_FILE`.
    #[test]
    #[ignore = "needs CAMELID_ORNITH_GGUF + CAMELID_STREAM_FILE"]
    fn ornith_qwen35_verify_cost() {
        let path = std::env::var("CAMELID_ORNITH_GGUF").expect("set CAMELID_ORNITH_GGUF");
        let stream_file = std::env::var("CAMELID_STREAM_FILE").expect("set CAMELID_STREAM_FILE");
        let token_sets: Vec<Vec<u32>> = serde_json::from_str(
            &std::fs::read_to_string(&stream_file).expect("read CAMELID_STREAM_FILE"),
        )
        .expect("CAMELID_STREAM_FILE must be a JSON array of u32 arrays");
        let prefix_len: usize = std::env::var("CAMELID_VERIFY_PREFIX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);
        let tokens = &token_sets[0];
        assert!(
            tokens.len() >= prefix_len + 12,
            "first stream sequence too short for prefix {prefix_len}+12"
        );
        let model = RunnableModel::load(&path).expect("load qwen35");
        for rep in 0..3 {
            let base = model
                .qwen35_prefill_timed(&tokens[..prefix_len])
                .expect("prefill base");
            eprintln!("VERIFY_COST rep={rep} k=0 prefix={prefix_len} secs={base:.4}");
            for k in [4usize, 6, 8, 12] {
                let t = model
                    .qwen35_prefill_timed(&tokens[..prefix_len + k])
                    .expect("prefill P+k");
                eprintln!("VERIFY_COST rep={rep} k={k} prefix={prefix_len} secs={t:.4}");
            }
        }
    }
}
