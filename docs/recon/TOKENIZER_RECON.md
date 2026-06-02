# Tokenizer Reverse Engineering

> [!NOTE]
> This document is a design or recon note, not the public support ledger. For current support truth and release status, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../../STATUS.md).

Source: tokenizer recon focused on GGUF tokenizer metadata keys and common LLaMA-family tokenizer behavior.

## Summary

camelid implements tokenizer support as a separate module from GGUF parsing, model execution, and chat-template formatting. The proven baseline remains LLaMA-style SPM from GGUF metadata for TinyLlama, and the current Phase 12 expansion accepts `tokenizer.ggml.model = "gpt2"` only when `tokenizer.ggml.pre = "llama-bpe"` for Llama 3-style GPT-2/BPE tokenizers. WPM/BERT, UGM/T5, RWKV, Gemma variants, Plamo, and other tokenizer models should continue to return explicit unsupported errors until they get their own fixtures and parity evidence.

## Current Llama 3 BPE implementation notes

- `BpeRegistry` stores GGUF `tokenizer.ggml.merges` ranks and uses a std-only `BinaryHeap` priority queue so lower-rank merges win before leftmost fallback behavior.
- The llama-bpe pre-tokenizer is a manual UTF-8/Unicode char scanner, not a regex dependency. It follows the ordered tiktoken-style branches for contraction suffixes, optional-prefix letter groups, 1-3 digit chunks, punctuation with trailing newlines, newline whitespace, trailing whitespace, and general whitespace.
- Real-artifact tokenizer tests load `$CAMELID_MODEL_DIR/Meta-Llama-3-8B-Instruct-Q8_0.gguf` when present and verify all `128256` `tokenizer.ggml.tokens` entries are mirrored exactly into Camelid token IDs/text before any 8B generation claim.
- TinyLlama SPM remains the regression baseline. Keep the 29907 / `"C"` one-token smoke green whenever tokenizer code changes.

## Public tokenizer surface goals

Relevant public APIs:

- `llama_tokenize(vocab, text, len, tokens, max, add_special, parse_special)`
- `llama_token_to_piece(vocab, token, buf, len, lstrip, special)`
- `llama_detokenize(vocab, tokens, n, text, max, remove_special, unparse_special)`
- special token accessors:
  - `llama_vocab_bos`
  - `llama_vocab_eos`
  - `llama_vocab_eot`
  - `llama_vocab_sep`
  - `llama_vocab_nl`
  - `llama_vocab_pad`
  - `llama_vocab_mask`
- config accessors:
  - `llama_vocab_get_add_bos`
  - `llama_vocab_get_add_eos`
  - `llama_vocab_get_add_sep`

## Tokenizer model families

Common local LLM runtimes support these broad tokenizer families:

```text
none
spm     tokenizer.ggml.model = "llama"
bpe     tokenizer.ggml.model = "gpt2" or "gemma4"
wpm     tokenizer.ggml.model = "bert"
ugm     tokenizer.ggml.model = "t5"
rwkv
plamo2
```

Phase 3 SPM support is complete for the TinyLlama baseline. Current Phase 12 scope adds guarded Llama 3-style `gpt2`/`llama-bpe` support only; other BPE/pre-tokenizer families remain unsupported until proven separately.

## Token types

`tokenizer.ggml.token_type` maps integer values to token attributes:

```text
0 undefined
1 normal
2 unknown
3 control
4 user_defined
5 unused
6 byte
```

Represent this as a Rust enum rather than raw integers.

## GGUF metadata keys

Known tokenizer-related keys:

```text
tokenizer.ggml.model
tokenizer.ggml.pre
tokenizer.ggml.tokens
tokenizer.ggml.token_type
tokenizer.ggml.token_type_count
tokenizer.ggml.scores
tokenizer.ggml.merges
tokenizer.ggml.bos_token_id
tokenizer.ggml.eos_token_id
tokenizer.ggml.eot_token_id
tokenizer.ggml.eom_token_id
tokenizer.ggml.unknown_token_id
tokenizer.ggml.seperator_token_id
tokenizer.ggml.padding_token_id
tokenizer.ggml.cls_token_id
tokenizer.ggml.mask_token_id
tokenizer.ggml.add_bos_token
tokenizer.ggml.add_eos_token
tokenizer.ggml.add_sep_token
tokenizer.ggml.add_space_prefix
tokenizer.ggml.remove_extra_whitespaces
tokenizer.ggml.precompiled_charsmap
tokenizer.chat_template
tokenizer.ggml.fim_pre_token_id
tokenizer.ggml.fim_suf_token_id
tokenizer.ggml.fim_mid_token_id
tokenizer.ggml.fim_pad_token_id
tokenizer.ggml.fim_rep_token_id
tokenizer.ggml.fim_sep_token_id
```

Minimum useful subset for first SPM implementation:

```text
tokenizer.ggml.model = "llama"
tokenizer.ggml.tokens: array<string>
tokenizer.ggml.scores: array<float> recommended
tokenizer.ggml.token_type: array<int> recommended
tokenizer.ggml.bos_token_id optional, default 1
tokenizer.ggml.eos_token_id optional, default 2
tokenizer.ggml.unknown_token_id optional, default 0
tokenizer.ggml.add_bos_token optional, default true for SPM
tokenizer.ggml.add_eos_token optional, default false for SPM
tokenizer.ggml.add_space_prefix optional, default true for SPM
```

## SPM/LLaMA behavior to implement first

High-level behavior:

1. If `add_space_prefix` applies, prefix whitespace before raw text.
2. Escape whitespace: spaces become SentencePiece marker `▁`.
3. Split input into UTF-8 characters/symbols.
4. Greedily merge adjacent symbols using vocab token scores.
5. If no token matches a segment, fallback to byte tokens using `<0xXX>` tokens.
6. If `add_special && add_bos`, prepend BOS.
7. If `add_special && add_eos`, append EOS.
8. Do not silently deduplicate double BOS/EOS unless a later decision explicitly chooses that behavior.

Important: this is not GPT-2 BPE. Start with LLaMA/SPM byte-fallback behavior only.

## Recommended Rust model

```rust
pub struct Tokenizer {
    pub model: TokenizerModel,
    pub tokens: Vec<Token>,
    pub token_to_id: HashMap<String, TokenId>,
    pub special: SpecialTokens,
    pub config: TokenizerConfig,
}

pub enum TokenizerModel {
    LlamaSpm,
    Gpt2Bpe,
    BertWpm,
    T5Ugm,
    Rwkv,
    Plamo2,
    None,
}

pub struct Token {
    pub id: u32,
    pub text: String,
    pub score: f32,
    pub kind: TokenKind,
}

pub enum TokenKind {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

pub struct SpecialTokens {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub eot: Option<u32>,
    pub eom: Option<u32>,
    pub unk: Option<u32>,
    pub sep: Option<u32>,
    pub pad: Option<u32>,
    pub mask: Option<u32>,
    pub fim_pre: Option<u32>,
    pub fim_suf: Option<u32>,
    pub fim_mid: Option<u32>,
    pub fim_pad: Option<u32>,
    pub fim_rep: Option<u32>,
    pub fim_sep: Option<u32>,
    pub eog: BTreeSet<u32>,
}

pub struct TokenizerConfig {
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_sep: bool,
    pub add_space_prefix: bool,
    pub remove_extra_whitespaces: bool,
}
```

Suggested API:

```rust
impl Tokenizer {
    pub fn from_gguf(file: &GgufFile) -> Result<Self>;
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<u32>>;
    pub fn decode(&self, tokens: &[u32], remove_special: bool) -> Result<String>;
}
```

## Error handling

Add typed errors:

```rust
UnsupportedTokenizer(String)
InvalidTokenizerMetadata(String)
TokenizerNotAvailable
```

Examples:

- `unsupported tokenizer model "gpt2"; currently supported: llama/SPM`
- `tokenizer.ggml.tokens is required and must be array<string>`
- `tokenizer.ggml.scores length 123 < token count 456`
- `SPM byte fallback token <0x0A> is missing`
- `token id 9999 out of range`

## Required GGUF helpers

Add helpers to `GgufFile`:

- `metadata_bool`
- `metadata_u32`
- `metadata_i32`
- `metadata_array_strings`
- `metadata_array_f32`
- `metadata_array_i32`

## Exact next code tasks

1. Add `src/tokenizer/mod.rs`.
2. Export tokenizer module in `src/lib.rs`.
3. Add tokenizer-related variants to `BackendError`.
4. Add typed metadata helper methods to `GgufFile`.
5. Implement `Tokenizer::from_gguf` for `tokenizer.ggml.model = "llama"`.
6. Implement SPM encode:
   - whitespace escape
   - UTF-8 symbol split
   - score-based bigram merge
   - `<0xXX>` byte fallback
   - BOS/EOS insertion
7. Implement minimal decode:
   - skip BOS/EOS when `remove_special`
   - byte tokens decode to bytes
   - unescape `▁` to spaces
   - suppress control tokens unless explicitly rendering specials later
8. Add optional API endpoints later:
   - `POST /api/tokenize`
   - `POST /api/detokenize`
9. Update `/api/capabilities` once tokenizer support is actually available.

## Recommended tests

1. `loads_llama_spm_tokenizer_metadata`
2. `rejects_missing_tokenizer_tokens`
3. `rejects_unsupported_gpt2_bpe_for_now`
4. `encodes_empty_with_add_special_to_bos`
5. `encodes_empty_without_special_to_empty`
6. `encodes_known_piece_with_space_prefix`
7. `falls_back_to_byte_tokens`
8. `decodes_byte_tokens`
9. `honors_add_bos_add_eos_metadata`
10. `rejects_score_or_token_type_array_shorter_than_vocab`

## Chat template boundary

Keep raw tokenization separate from chat template rendering. `tokenizer.chat_template` should be parsed/preserved as metadata for a future prompt-formatting layer, not mixed into `encode`.
