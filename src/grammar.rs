//! Token-level structured-output constraints powered by LLGuidance.
//!
//! A [`ConstraintSpec`] is validated when the request is parsed, then compiled
//! against the loaded GGUF tokenizer's exact byte pieces before inference. The
//! generation loop asks [`ConstraintState`] for a token mask and commits the
//! sampled token. No token is decoded as a standalone UTF-8 string, so byte-level
//! BPE and byte-fallback vocabularies retain non-ASCII coverage.

use std::sync::Arc;

use llguidance::{api::TopLevelGrammar, Constraint, ParserFactory};
use toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};

use crate::tokenizer::Tokenizer;

/// A validated, cheaply clonable structured-output constraint.
#[derive(Clone, Debug)]
pub struct ConstraintSpec(ConstraintKind);

#[derive(Clone, Debug)]
enum ConstraintKind {
    JsonObject,
    JsonSchema(serde_json::Value),
    Lark(String),
}

impl ConstraintSpec {
    /// `response_format: {"type":"json_object"}`.
    pub fn json_object() -> Self {
        Self::validated(ConstraintKind::JsonObject)
            .expect("LLGuidance's built-in JSON object schema must compile")
    }

    /// Compile any JSON Schema supported by LLGuidance.
    ///
    /// Compilation happens immediately against a canonical byte tokenizer so
    /// unsupported schemas are typed request errors, never silently ignored
    /// constraints. The same grammar is compiled again against the model's real
    /// token trie immediately before generation.
    pub fn from_schema(schema: &serde_json::Value) -> Result<Self, ConstraintError> {
        Self::validated(ConstraintKind::JsonSchema(schema.clone()))
    }

    /// Compile a Lark/LLGuidance context-free grammar.
    ///
    /// Plain Lark, `%llguidance` documents, and LLGuidance serialized grammar
    /// lists are accepted by the upstream parser.
    pub fn from_lark(grammar: &str) -> Result<Self, ConstraintError> {
        if grammar.trim().is_empty() {
            return Err(ConstraintError("grammar must not be empty".to_string()));
        }
        Self::validated(ConstraintKind::Lark(grammar.to_string()))
    }

    fn validated(kind: ConstraintKind) -> Result<Self, ConstraintError> {
        let spec = Self(kind);
        spec.build_with_env(ApproximateTokEnv::single_byte_env())?;
        Ok(spec)
    }

    /// Build one generation session against the exact loaded tokenizer.
    pub fn build(&self, tokenizer: &Tokenizer) -> Result<ConstraintState, ConstraintError> {
        self.build_with_env(token_env(tokenizer)?)
    }

    fn grammar(&self) -> Result<TopLevelGrammar, ConstraintError> {
        match &self.0 {
            ConstraintKind::JsonObject => Ok(TopLevelGrammar::from_json_schema(
                serde_json::json!({"type": "object"}),
            )),
            ConstraintKind::JsonSchema(schema) => {
                Ok(TopLevelGrammar::from_json_schema(schema.clone()))
            }
            ConstraintKind::Lark(grammar) => TopLevelGrammar::from_lark_or_grammar_list(grammar)
                .map_err(ConstraintError::from_anyhow),
        }
    }

    fn build_with_env(&self, env: TokEnv) -> Result<ConstraintState, ConstraintError> {
        let mut factory = ParserFactory::new_simple(&env).map_err(ConstraintError::from_anyhow)?;
        // Compiler/evaluator failures are returned through Camelid's typed API;
        // never leak an upstream diagnostic to process stderr as a side channel.
        factory.quiet();
        let parser = factory
            .create_parser(self.grammar()?)
            .map_err(ConstraintError::from_anyhow)?;
        Ok(ConstraintState {
            inner: Constraint::new(parser),
        })
    }
}

/// One token-level LLGuidance session.
pub struct ConstraintState {
    inner: Constraint,
}

impl std::fmt::Debug for ConstraintState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintState")
            .field("engine", &"llguidance")
            .finish_non_exhaustive()
    }
}

impl ConstraintState {
    /// Fill the next-token mask.
    ///
    /// `Ok(true)` means the grammar has completed and generation should stop.
    /// Otherwise `out[id]` reports whether token `id` may be sampled.
    pub fn compute_mask(&mut self, out: &mut [bool]) -> Result<bool, ConstraintError> {
        let result = self
            .inner
            .compute_mask()
            .map_err(ConstraintError::from_anyhow)?;
        if result.is_stop() {
            out.fill(false);
            return Ok(true);
        }

        let mask = result.sample_mask.as_ref().ok_or_else(|| {
            ConstraintError(
                "LLGuidance returned a token splice although fast-forward tokens are disabled"
                    .to_string(),
            )
        })?;
        if mask.len() != out.len() {
            return Err(ConstraintError(format!(
                "LLGuidance mask length {} does not match model vocabulary {}",
                mask.len(),
                out.len()
            )));
        }
        for (id, allowed) in out.iter_mut().enumerate() {
            *allowed = mask.is_allowed(id as u32);
        }
        Ok(false)
    }

    /// Commit the sampled token.
    ///
    /// Returns true as soon as LLGuidance has accepted a complete value and no
    /// additional token is required.
    pub fn commit_token(&mut self, token: u32) -> Result<bool, ConstraintError> {
        let committed = self
            .inner
            .commit_token(Some(token))
            .map_err(ConstraintError::from_anyhow)?;
        Ok(committed.stop || self.inner.has_pending_stop())
    }
}

/// Fail-closed structured-output compilation/evaluation diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintError(String);

impl ConstraintError {
    fn from_anyhow(err: anyhow::Error) -> Self {
        Self(err.to_string())
    }
}

impl std::fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for ConstraintError {}

fn token_env(tokenizer: &Tokenizer) -> Result<TokEnv, ConstraintError> {
    let words = (0..tokenizer.tokens.len() as u32)
        .map(|id| {
            tokenizer
                .constraint_token_bytes(id)
                .map_err(|err| ConstraintError(err.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let primary_eos = tokenizer
        .special
        .eot
        .or(tokenizer.special.eos)
        .or_else(|| tokenizer.special.eog.iter().next().copied())
        .ok_or_else(|| ConstraintError("tokenizer has no EOG/EOS token".to_string()))?;
    let info = TokRxInfo {
        vocab_size: words.len() as u32,
        tok_eos: primary_eos,
        tok_bos: tokenizer.special.bos,
        tok_pad: tokenizer.special.pad,
        tok_unk: tokenizer.special.unk,
        tok_end_of_turn: tokenizer.special.eot,
    };
    let mut trie = TokTrie::from(&info, &words);
    let mut eos_tokens = tokenizer.special.eog.iter().copied().collect::<Vec<_>>();
    if !eos_tokens.contains(&primary_eos) {
        eos_tokens.push(primary_eos);
    }
    trie = trie.with_eos_tokens(&eos_tokens);
    Ok(Arc::new(ApproximateTokEnv::new(trie)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BYTE_VOCAB: usize = 262;
    const BYTE_EOS: usize = BYTE_VOCAB - 1;

    fn build(spec: &ConstraintSpec) -> ConstraintState {
        spec.build_with_env(ApproximateTokEnv::single_byte_env())
            .unwrap()
    }

    fn feed(state: &mut ConstraintState, bytes: &[u8]) -> bool {
        let mut mask = vec![false; BYTE_VOCAB];
        let mut done = false;
        for &byte in bytes {
            assert!(!state.compute_mask(&mut mask).unwrap());
            assert!(
                mask[byte as usize],
                "byte {byte:#04x} is unexpectedly masked"
            );
            done = state.commit_token(byte as u32).unwrap();
        }
        done
    }

    #[test]
    fn json_object_masks_non_object_root_and_stops_at_close() {
        let mut state = build(&ConstraintSpec::json_object());
        let mut mask = vec![false; BYTE_VOCAB];
        assert!(!state.compute_mask(&mut mask).unwrap());
        assert!(mask[b'{' as usize]);
        assert!(!mask[b'[' as usize]);
        assert!(!mask[BYTE_EOS]);

        assert!(feed(&mut state, br#"{"a":1}"#));
    }

    #[test]
    fn broad_json_schema_accepts_scalar_pattern_bounds_and_any_of() {
        let spec = ConstraintSpec::from_schema(&serde_json::json!({
            "anyOf": [
                {"type": "string", "pattern": "^[a-z]{2,8}$"},
                {"type": "integer", "minimum": 1, "maximum": 9}
            ]
        }))
        .unwrap();

        let mut string_state = build(&spec);
        assert!(feed(&mut string_state, br#""camel""#));

        let mut number_state = build(&spec);
        assert!(feed(&mut number_state, b"7"));
    }

    #[test]
    fn lark_cfg_masks_and_commits_tokens() {
        let spec = ConstraintSpec::from_lark(r#"start: "yes" | "no""#).unwrap();
        let mut state = build(&spec);
        let mut mask = vec![false; BYTE_VOCAB];
        assert!(!state.compute_mask(&mut mask).unwrap());
        assert!(mask[b'y' as usize]);
        assert!(mask[b'n' as usize]);
        assert!(!mask[b'x' as usize]);
        assert!(feed(&mut state, b"yes"));
    }

    #[test]
    fn malformed_or_unsupported_constraints_fail_closed() {
        assert!(ConstraintSpec::from_lark("").is_err());
        assert!(ConstraintSpec::from_lark("start: (").is_err());
        assert!(ConstraintSpec::from_schema(&serde_json::json!({"type": "camel"})).is_err());
    }

    #[test]
    fn split_utf8_tokens_remain_reachable() {
        let words = vec![vec![0xc3], vec![0xa9], b"\xff<eos>".to_vec()];
        let trie = TokTrie::from(&TokRxInfo::new(words.len() as u32, 2), &words);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let spec = ConstraintSpec::from_lark("start: \"é\"").unwrap();
        let mut state = spec.build_with_env(env).unwrap();
        let mut mask = vec![false; words.len()];

        assert!(!state.compute_mask(&mut mask).unwrap());
        assert!(mask[0], "the leading UTF-8 fragment must remain reachable");
        assert!(!mask[1]);
        assert!(!state.commit_token(0).unwrap());

        assert!(!state.compute_mask(&mut mask).unwrap());
        assert!(mask[1], "the continuation UTF-8 fragment must be allowed");
        assert!(state.commit_token(1).unwrap());
    }

    #[test]
    fn mask_size_mismatch_is_typed_error() {
        let mut state = build(&ConstraintSpec::json_object());
        let err = state.compute_mask(&mut [false; 1]).unwrap_err();
        assert!(err.to_string().contains("mask length"));
    }
}
