use std::{
    collections::{BTreeSet, BinaryHeap, HashMap},
    fs::File,
    io::Read,
};

use sha2::Digest;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;

use crate::{gguf::GgufFile, BackendError, Result};

pub type TokenId = u32;

const SPM_SPACE: char = '▁';

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    LlamaSpm,
    Gpt2Bpe,
    BertWordPiece,
}

impl TokenizerModel {
    pub fn as_summary_model(self) -> &'static str {
        match self {
            Self::LlamaSpm => "llama_spm",
            Self::Gpt2Bpe => "gpt2_bpe",
            Self::BertWordPiece => "bert_wordpiece",
        }
    }
}

/// GPT-2/BPE pre-tokenizer dialect (`tokenizer.ggml.pre`). The byte-level BPE
/// merge step is identical across these; the pre-tokenization regex that splits
/// raw text into pieces differs in exactly one branch — digit grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BpePreTokenizer {
    /// llama.cpp `llama-bpe` (Llama 3 / GPT-4 tiktoken): digits group in runs of
    /// up to three (`\p{N}{1,3}`).
    #[default]
    Llama3,
    /// command-r pre-tokenizer, standard byte-fallback tiktoken behavior.
    CommandR,
    /// llama.cpp `qwen2` (Qwen2/Qwen3): each digit is its own piece (`\p{N}`).
    /// Byte-for-byte identical to `llama-bpe` in every other branch — verified
    /// against `llama-vocab.cpp` LLAMA_VOCAB_PRE_TYPE_LLAMA3 vs _QWEN2.
    Qwen2,
    /// llama.cpp `qwen35` (Qwen3.5 / Ornith): single-digit grouping like `qwen2`,
    /// and `LLAMA_VOCAB_PRE_TYPE_QWEN35`'s regex additionally folds Unicode
    /// combining marks `\p{M}` into the letter class (`\p{L}+` → `[\p{L}\p{M}]+`
    /// and the punctuation class excludes `\p{M}`). Mark-folding is implemented
    /// via a generated `\p{M}` range table (`mark_ranges.rs.inc`) — see
    /// [`fold_marks`](Self::fold_marks). Byte-exactness vs the oracle is held by
    /// the ITEM1 tokenizer gate (qa/ornith/constrained-vram), which covers NFD,
    /// Devanagari (incl. virama clusters), Arabic harakat, and zalgo inputs.
    Qwen35,
    /// llama.cpp `gpt-4o`: its collapsed-regex dialect uses ASCII-only case
    /// classes over Unicode General Category `L` code points.
    Gpt4o,
}

impl BpePreTokenizer {
    /// Maximum number of consecutive digits the pre-tokenizer keeps in one piece.
    fn digit_group_max(self) -> usize {
        match self {
            Self::Llama3 | Self::CommandR | Self::Gpt4o => 3,
            Self::Qwen2 | Self::Qwen35 => 1,
        }
    }

    /// Whether `\p{M}` combining marks fold into the letter class (and are
    /// excluded from the punctuation class) — the qwen35 regex dialect.
    fn fold_marks(self) -> bool {
        matches!(self, Self::Qwen35)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenKind {
    fn from_i32(value: i32) -> Result<Self> {
        Ok(match value {
            0 => Self::Undefined,
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            other => {
                return Err(BackendError::InvalidTokenizerMetadata(format!(
                    "unknown tokenizer token type {other}"
                )))
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: TokenId,
    pub text: String,
    pub score: f32,
    pub kind: TokenKind,
}

/// True for `<|...|>` chat-control markers (e.g. Phi-3's `<|end|>`/`<|assistant|>`,
/// ChatML's `<|im_start|>`) carried as `UserDefined` tokens. These are turn
/// scaffolding, never user-visible content, so they are stripped from decoded
/// output under `remove_special` — exactly like `Control` tokens. Content-bearing
/// `UserDefined` tokens that are NOT this shape (e.g. Qwen3's `<think>`/`</think>`,
/// which begin `<` but not `<|`) are preserved.
fn is_chat_control_marker(token: &Token) -> bool {
    token.kind == TokenKind::UserDefined
        && token.text.starts_with("<|")
        && token.text.ends_with("|>")
}

#[derive(Debug, Clone, Default)]
pub struct BpeRegistry {
    ranks: HashMap<(String, String), usize>,
}

impl BpeRegistry {
    fn from_merges(merges: Vec<String>) -> Self {
        let ranks = merges
            .into_iter()
            .enumerate()
            .filter_map(|(rank, merge)| {
                let (left, right) = merge.split_once(' ')?;
                Some(((left.to_string(), right.to_string()), rank))
            })
            .collect();
        Self { ranks }
    }

    pub fn len(&self) -> usize {
        self.ranks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    fn rank(&self, left: &str, right: &str) -> Option<usize> {
        self.ranks
            .get(&(left.to_string(), right.to_string()))
            .copied()
    }

    fn ranks(&self) -> &HashMap<(String, String), usize> {
        &self.ranks
    }

    fn merge_symbols(&self, mut symbols: Vec<String>) -> Vec<String> {
        while symbols.len() > 1 {
            let mut heap = BinaryHeap::new();
            for idx in 0..symbols.len() - 1 {
                if let Some(rank) = self.rank(&symbols[idx], &symbols[idx + 1]) {
                    heap.push(BpeMergeCandidate { rank, index: idx });
                }
            }

            let Some(best) = heap.pop() else { break };
            let left = symbols[best.index].clone();
            let right = symbols[best.index + 1].clone();
            let mut merged = Vec::with_capacity(symbols.len() - 1);
            let mut idx = 0;
            while idx < symbols.len() {
                if idx + 1 < symbols.len() && symbols[idx] == left && symbols[idx + 1] == right {
                    merged.push(format!("{}{}", symbols[idx], symbols[idx + 1]));
                    idx += 2;
                } else {
                    merged.push(symbols[idx].clone());
                    idx += 1;
                }
            }
            symbols = merged;
        }
        symbols
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BpeMergeCandidate {
    rank: usize,
    index: usize,
}

impl Ord for BpeMergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for BpeMergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    pub bos: Option<TokenId>,
    pub eos: Option<TokenId>,
    pub eot: Option<TokenId>,
    pub eom: Option<TokenId>,
    pub unk: Option<TokenId>,
    pub sep: Option<TokenId>,
    pub pad: Option<TokenId>,
    pub mask: Option<TokenId>,
    pub eog: BTreeSet<TokenId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerConfig {
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_sep: bool,
    pub add_space_prefix: bool,
    pub remove_extra_whitespaces: bool,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub model: TokenizerModel,
    /// GPT-2/BPE pre-tokenizer dialect. Only consulted on the [`TokenizerModel::Gpt2Bpe`]
    /// path; defaults to [`BpePreTokenizer::Llama3`] and is ignored for SPM.
    pub bpe_pre_tokenizer: BpePreTokenizer,
    pub tokens: Vec<Token>,
    pub token_to_id: HashMap<String, TokenId>,
    pub byte_token_to_id: HashMap<u8, TokenId>,
    pub bpe_ranks: HashMap<(String, String), usize>,
    pub bpe_registry: BpeRegistry,
    pub special: SpecialTokens,
    pub config: TokenizerConfig,
    pub chat_template: Option<String>,
}

/// The Llama-3 tiktoken tokenizer's special-token signature. Llama 3 / 3.1 / 3.2 all
/// place these five stable chat markers at these exact ids in a 128,256-token vocab,
/// and no other tokenizer family does — so a GPT-2/BPE GGUF carrying them IS the
/// llama-bpe tokenizer. Used only to recover a MISSING `tokenizer.ggml.pre` (see
/// `Tokenizer::from_gguf`); the checked ids deliberately exclude the reserved slots
/// that were renamed between Llama-3 and 3.2 (e.g. 128008 `<|eom_id|>`). Proven
/// byte-identical base vocab `[0, 128000)` and merges against a `pre=llama-bpe`
/// Llama-3.2 GGUF.
fn is_llama3_bpe_signature(token_texts: &[String]) -> bool {
    token_texts.len() == 128_256
        && token_texts.get(128_000).map(String::as_str) == Some("<|begin_of_text|>")
        && token_texts.get(128_001).map(String::as_str) == Some("<|end_of_text|>")
        && token_texts.get(128_006).map(String::as_str) == Some("<|start_header_id|>")
        && token_texts.get(128_007).map(String::as_str) == Some("<|end_header_id|>")
        && token_texts.get(128_009).map(String::as_str) == Some("<|eot_id|>")
}

/// Resolve the GPT-2/BPE pre-tokenizer dialect from `tokenizer.ggml.pre`. The three
/// known dialects differ only in the split regex (digit grouping / mark folding); the
/// byte-BPE merge step is identical (verified against llama.cpp llama-vocab.cpp). When
/// the key is ABSENT, recover llama-bpe iff the vocab carries the Llama-3 signature —
/// some Llama-3 GGUF conversions omit the key, and llama.cpp then silently mis-tokenizes
/// them under a raw GPT-2 fallback ("GENERATION QUALITY WILL BE DEGRADED"). An
/// explicit-but-unknown `pre`, or a missing key without the signature (e.g. a de-labeled
/// Qwen), is refused. Extracted from `from_gguf` so the decision is unit-testable.
fn resolve_gpt2_pre_tokenizer(
    pre: Option<&str>,
    token_texts: &[String],
    allow_gpt4o: bool,
) -> Result<BpePreTokenizer> {
    match pre {
        Some("llama-bpe") => Ok(BpePreTokenizer::Llama3),
        Some("command-r") => Ok(BpePreTokenizer::CommandR),
        Some("qwen2") => Ok(BpePreTokenizer::Qwen2),
        Some("qwen35") => Ok(BpePreTokenizer::Qwen35),
        Some("gpt-4o") if allow_gpt4o => Ok(BpePreTokenizer::Gpt4o),
        Some("gpt-4o") => Err(BackendError::UnsupportedTokenizer(
            "gpt-4o is admitted only for the exact Phi-4-mini-instruct-Q4_K_M artifact"
                .to_string(),
        )),
        None if is_llama3_bpe_signature(token_texts) => Ok(BpePreTokenizer::Llama3),
        other => Err(BackendError::UnsupportedTokenizer(format!(
            "unsupported GPT-2/BPE pre-tokenizer {other:?}; currently supported: llama-bpe, command-r, qwen2, qwen35"
        ))),
    }
}

const PHI4_MINI_Q4KM_SHA256: &str =
    "88c00229914083cd112853aab84ed51b87bdf6b9ce42f532d8c85c7c63b1730a";

fn is_exact_phi4_mini_q4km(file: &GgufFile) -> bool {
    let named_phi4 = file.architecture() == Some("phi3")
        && file.model_name() == Some("Phi 4 Mini Instruct")
        && file.path.file_name().and_then(|name| name.to_str())
            == Some("Phi-4-mini-instruct-Q4_K_M.gguf");
    named_phi4 && sha256_file(&file.path).is_some_and(|sha256| sha256 == PHI4_MINI_Q4KM_SHA256)
}

fn sha256_file(path: &std::path::Path) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let mut digest = sha2::Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Some(format!("{:x}", digest.finalize()))
}

impl Tokenizer {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let model_name = file
            .metadata_string("tokenizer.ggml.model")
            .ok_or(BackendError::TokenizerNotAvailable)?;
        let model = match model_name {
            // Gemma uses a SentencePiece (unigram) tokenizer, the same mechanism as
            // Llama SPM — tokens, scores, and the bos/eos/unk ids are all read from
            // the GGUF below. (Gemma sets tokenizer.ggml.add_space_prefix=0; exact
            // leading-space parity is a follow-up, construction works on the SPM path.)
            "llama" | "gemma2" | "gemma3" | "gemma4" => TokenizerModel::LlamaSpm,
            "gpt2" => TokenizerModel::Gpt2Bpe,
            "bert" => TokenizerModel::BertWordPiece,
            other => {
                return Err(BackendError::UnsupportedTokenizer(format!(
                    "unsupported tokenizer model {other:?}; currently supported: llama/SPM, gemma2/SPM, gemma3/SPM, gemma4/SPM, GPT-2/BPE llama-bpe, and BERT WordPiece"
                )))
            }
        };
        // Read the token list up front: it is needed both to recover a missing
        // pre-tokenizer from the vocab signature (below) and to build the vocab.
        let token_texts = file.metadata_array_strings("tokenizer.ggml.tokens")?;
        if token_texts.is_empty() {
            return Err(BackendError::InvalidTokenizerMetadata(
                "tokenizer.ggml.tokens must not be empty".to_string(),
            ));
        }

        let bpe_pre_tokenizer = if model == TokenizerModel::Gpt2Bpe {
            resolve_gpt2_pre_tokenizer(
                file.metadata_string("tokenizer.ggml.pre"),
                &token_texts,
                is_exact_phi4_mini_q4km(file),
            )?
        } else {
            BpePreTokenizer::default()
        };

        let scores = file
            .metadata_array_f32_optional("tokenizer.ggml.scores")?
            .unwrap_or_else(|| vec![0.0; token_texts.len()]);
        if scores.len() < token_texts.len() {
            return Err(BackendError::InvalidTokenizerMetadata(format!(
                "tokenizer.ggml.scores length {} is shorter than token count {}",
                scores.len(),
                token_texts.len()
            )));
        }

        let kinds_raw = file
            .metadata_array_i32_optional("tokenizer.ggml.token_type")?
            .unwrap_or_else(|| vec![1; token_texts.len()]);
        if kinds_raw.len() < token_texts.len() {
            return Err(BackendError::InvalidTokenizerMetadata(format!(
                "tokenizer.ggml.token_type length {} is shorter than token count {}",
                kinds_raw.len(),
                token_texts.len()
            )));
        }

        // `tokenizer.ggml.model = "llama"` is SPM proper: the reference segments it
        // by token SCORE and ignores any merge list the converter happened to embed.
        // Some Llama-family GGUFs ship merges anyway (TinyLlama carries 61k), and
        // honouring them silently switched that model to rank-based BPE, segmenting
        // ordinary words differently from the reference (`thunderstorm` -> `st|orm`
        // instead of `stor|m`). Drop merges for this model only.
        //
        // Scoped to the raw metadata string, NOT `TokenizerModel::LlamaSpm`: gemma2/
        // gemma3/gemma4 also map onto that enum but genuinely are merge-driven here
        // (the DiffusionGemma tokenizer-parity gate pins gemma4's merge behaviour),
        // and they declare their own tokenizer model names.
        let spm_ignores_merges = model_name == "llama";
        let bpe_registry = BpeRegistry::from_merges(if spm_ignores_merges {
            Vec::new()
        } else {
            file.metadata_array_strings_optional("tokenizer.ggml.merges")?
                .unwrap_or_default()
        });
        let bpe_ranks = bpe_registry.ranks().clone();

        let mut tokens = Vec::with_capacity(token_texts.len());
        let mut token_to_id = HashMap::with_capacity(token_texts.len());
        let mut byte_token_to_id = HashMap::new();
        for (idx, text) in token_texts.into_iter().enumerate() {
            let id = idx as TokenId;
            let kind = TokenKind::from_i32(kinds_raw[idx])?;
            if let Some(byte) = parse_byte_token(&text) {
                byte_token_to_id.insert(byte, id);
            }
            token_to_id.insert(text.clone(), id);
            tokens.push(Token {
                id,
                text,
                score: scores[idx],
                kind,
            });
        }

        let default_bos = match model {
            TokenizerModel::LlamaSpm => Some(1),
            TokenizerModel::Gpt2Bpe => token_to_id.get("<|begin_of_text|>").copied(),
            TokenizerModel::BertWordPiece => token_to_id.get("[CLS]").copied().or(Some(101)),
        };
        let default_eos = match model {
            TokenizerModel::LlamaSpm => Some(2),
            TokenizerModel::Gpt2Bpe => token_to_id.get("<|end_of_text|>").copied(),
            TokenizerModel::BertWordPiece => token_to_id.get("[SEP]").copied().or(Some(102)),
        };
        let default_unk = match model {
            TokenizerModel::LlamaSpm => Some(0),
            TokenizerModel::Gpt2Bpe => None,
            TokenizerModel::BertWordPiece => token_to_id.get("[UNK]").copied().or(Some(100)),
        };

        let bos = file
            .metadata_u32("tokenizer.ggml.bos_token_id")
            .or(default_bos);
        let eos = file
            .metadata_u32("tokenizer.ggml.eos_token_id")
            .or(default_eos);
        let unk = file
            .metadata_u32("tokenizer.ggml.unknown_token_id")
            .or(default_unk);
        let eot = file
            .metadata_u32("tokenizer.ggml.eot_token_id")
            .or_else(|| token_to_id.get("<|eot_id|>").copied());
        let eom = file.metadata_u32("tokenizer.ggml.eom_token_id");
        let sep = file
            .metadata_u32("tokenizer.ggml.separator_token_id")
            .or_else(|| file.metadata_u32("tokenizer.ggml.seperator_token_id"));
        let pad = file.metadata_u32("tokenizer.ggml.padding_token_id");
        let mask = file.metadata_u32("tokenizer.ggml.mask_token_id");
        // Well-known end-of-turn markers used by chat templates. Some GGUFs set
        // `eos` to a generic `<|endoftext|>` but END EACH CHAT TURN with a distinct
        // token and never populate `eot_token_id` — notably Phi-3 (`<|end|>`), so
        // without this its chat turns never stop and the model rambles into new
        // turns. llama.cpp likewise flags these as EOG. Purely additive: only ids
        // that genuinely exist in this vocab are added, and a supported row's
        // turn-end is already its `eos`/`eot`, so its stop set is unchanged.
        const EOG_MARKER_TEXTS: &[&str] = &[
            "<|end|>",       // Phi-3
            "<|eot_id|>",    // Llama 3
            "<|im_end|>",    // ChatML / Qwen
            "<end_of_turn>", // Gemma
            "<|eom_id|>",
        ];
        let mut eog: std::collections::BTreeSet<TokenId> =
            [eos, eot, eom].into_iter().flatten().collect();
        for marker in EOG_MARKER_TEXTS {
            if let Some(&id) = token_to_id.get(*marker) {
                eog.insert(id);
            }
        }

        validate_token_id("bos", bos, tokens.len())?;
        validate_token_id("eos", eos, tokens.len())?;
        validate_token_id("unk", unk, tokens.len())?;
        validate_token_id("eot", eot, tokens.len())?;
        validate_token_id("eom", eom, tokens.len())?;
        validate_token_id("sep", sep, tokens.len())?;
        validate_token_id("pad", pad, tokens.len())?;
        validate_token_id("mask", mask, tokens.len())?;

        Ok(Self {
            model,
            bpe_pre_tokenizer,
            tokens,
            token_to_id,
            byte_token_to_id,
            bpe_ranks,
            bpe_registry,
            special: SpecialTokens {
                bos,
                eos,
                eot,
                eom,
                unk,
                sep,
                pad,
                mask,
                eog,
            },
            config: TokenizerConfig {
                // Gemma 4 workaround (matches llama.cpp PR #21500): some gemma4
                // exports — notably the 26B A4B QAT GGUF — ship an incorrect
                // `add_bos_token = false`, but the model is always run with a
                // leading BOS. llama.cpp force-overrides it to true for gemma4;
                // do the same so the prompt token stream matches the reference
                // (without this, the BOS is dropped and the whole forward
                // diverges). E-series/12B already ship true, so this is a no-op
                // for them.
                add_bos: if model_name.starts_with("gemma") {
                    true
                } else {
                    file.metadata_bool("tokenizer.ggml.add_bos_token")
                        .unwrap_or(true)
                },
                add_eos: file
                    .metadata_bool("tokenizer.ggml.add_eos_token")
                    .unwrap_or(false),
                add_sep: file
                    .metadata_bool("tokenizer.ggml.add_sep_token")
                    .unwrap_or(model == TokenizerModel::BertWordPiece),
                add_space_prefix: file
                    .metadata_bool("tokenizer.ggml.add_space_prefix")
                    .unwrap_or(true),
                remove_extra_whitespaces: file
                    .metadata_bool("tokenizer.ggml.remove_extra_whitespaces")
                    .unwrap_or(false),
            },
            chat_template: file
                .metadata_string("tokenizer.chat_template")
                .map(str::to_owned),
        })
    }

    pub fn token_text(&self, id: Option<TokenId>) -> Option<&str> {
        id.and_then(|id| self.tokens.get(id as usize))
            .map(|token| token.text.as_str())
    }

    pub fn token_id(&self, text: &str) -> Option<TokenId> {
        self.token_to_id.get(text).copied()
    }

    pub fn encode(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        if add_special && self.config.add_bos {
            if let Some(bos) = self.special.bos {
                out.push(bos);
            }
        }

        match self.model {
            TokenizerModel::LlamaSpm => {
                let normalized = self.normalize_spm_text(text, parse_special);
                if !normalized.is_empty() {
                    out.extend(self.encode_piece(&normalized, parse_special)?);
                }
            }
            TokenizerModel::Gpt2Bpe => {
                if !text.is_empty() {
                    out.extend(self.encode_bpe_text(text, parse_special)?);
                }
            }
            TokenizerModel::BertWordPiece => {
                if !text.is_empty() {
                    out.extend(self.encode_wordpiece_text(text)?);
                }
            }
        }

        if add_special {
            if self.config.add_sep {
                if let Some(sep) = self.special.sep.or(self.special.eos) {
                    out.push(sep);
                }
            } else if self.config.add_eos {
                if let Some(eos) = self.special.eos {
                    out.push(eos);
                }
            }
        }
        Ok(out)
    }

    pub fn decode(&self, token_ids: &[TokenId], remove_special: bool) -> Result<String> {
        if self.model == TokenizerModel::Gpt2Bpe {
            return self.decode_bpe(token_ids, remove_special);
        }
        if self.model == TokenizerModel::BertWordPiece {
            return self.decode_wordpiece(token_ids, remove_special);
        }

        let mut bytes = Vec::new();
        let mut text = String::new();

        for id in token_ids {
            if remove_special && self.is_special(*id) {
                continue;
            }
            let token = self.tokens.get(*id as usize).ok_or_else(|| {
                BackendError::InvalidTokenizerMetadata(format!("token id {id} out of range"))
            })?;
            if remove_special && (token.kind == TokenKind::Control || is_chat_control_marker(token))
            {
                continue;
            }
            if let Some(byte) = parse_byte_token(&token.text) {
                bytes.push(byte);
                continue;
            }
            flush_bytes(&mut bytes, &mut text)?;
            text.push_str(&token.text.replace(SPM_SPACE, " "));
        }
        flush_bytes(&mut bytes, &mut text)?;
        Ok(text)
    }

    /// Return the exact byte piece LLGuidance must associate with one token.
    ///
    /// This deliberately does not round-trip through [`Self::decode`]. A
    /// byte-fallback or byte-level BPE token may contain only one byte of a
    /// multi-byte UTF-8 scalar, which is valid tokenizer state but not a valid
    /// standalone Rust `String`. Decoding tokens one at a time therefore loses
    /// precisely the fragments a token-trie constraint engine needs.
    ///
    /// Control/EOG tokens use toktrie's `0xff` marker, matching llama.cpp's
    /// LLGuidance adapter: normal detokenization is attempted first; tokens that
    /// exist only when special-token rendering is enabled are marker-prefixed.
    pub(crate) fn constraint_token_bytes(&self, id: TokenId) -> Result<Vec<u8>> {
        let token = self.tokens.get(id as usize).ok_or_else(|| {
            BackendError::InvalidTokenizerMetadata(format!("token id {id} out of range"))
        })?;

        if token.kind == TokenKind::Control
            || self.special.eog.contains(&id)
            || self.is_special(id)
            || is_chat_control_marker(token)
        {
            let mut bytes = Vec::with_capacity(token.text.len() + 1);
            bytes.push(toktrie::TokTrie::SPECIAL_TOKEN_MARKER);
            bytes.extend_from_slice(token.text.as_bytes());
            return Ok(bytes);
        }

        match self.model {
            TokenizerModel::LlamaSpm => {
                if let Some(byte) = parse_byte_token(&token.text) {
                    Ok(vec![byte])
                } else {
                    Ok(token.text.replace(SPM_SPACE, " ").into_bytes())
                }
            }
            TokenizerModel::Gpt2Bpe => token
                .text
                .chars()
                .map(|ch| {
                    bpe_char_to_byte(ch).ok_or_else(|| {
                        BackendError::InvalidTokenizerMetadata(format!(
                            "GPT-2/BPE token {:?} contains non-byte character {ch:?}",
                            token.text
                        ))
                    })
                })
                .collect(),
            TokenizerModel::BertWordPiece => {
                let piece = token.text.strip_prefix("##").unwrap_or(&token.text);
                Ok(piece
                    .strip_prefix(SPM_SPACE)
                    .unwrap_or(piece)
                    .as_bytes()
                    .to_vec())
            }
        }
    }

    /// Chat prompts are tokenized with special-token parsing for every model:
    /// llama-server tokenizes rendered chat templates with specials enabled,
    /// so a template's control markers (e.g. SPM `</s>` between turns) must
    /// become control token ids, not literal text. The committed TinyLlama
    /// parity evidence records exactly this shape (`..., 12199, 2, 29871,
    /// ...`). Raw completion text is unaffected and keeps
    /// `parse_special: false` — special parsing does not spread to raw text.
    pub fn chat_prompt_parse_special(&self) -> bool {
        true
    }

    fn encode_wordpiece_text(&self, text: &str) -> Result<Vec<TokenId>> {
        let normalized: String = text
            .nfd()
            .flat_map(char::to_lowercase)
            .filter(|ch| {
                !matches!(
                    get_general_category(*ch),
                    GeneralCategory::NonspacingMark
                        | GeneralCategory::SpacingMark
                        | GeneralCategory::EnclosingMark
                )
            })
            .collect();

        let mut basic = Vec::new();
        let mut current = String::new();
        for ch in normalized.chars() {
            if ch.is_whitespace() || ch.is_control() {
                if !current.is_empty() {
                    basic.push(std::mem::take(&mut current));
                }
            } else if is_bert_punctuation(ch) || is_cjk(ch) {
                if !current.is_empty() {
                    basic.push(std::mem::take(&mut current));
                }
                basic.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            basic.push(current);
        }

        let unk = self.special.unk.ok_or_else(|| {
            BackendError::InvalidTokenizerMetadata(
                "BERT WordPiece tokenizer has no unknown token".to_string(),
            )
        })?;
        // llama.cpp's GGUF BERT conversion uses a SentencePiece-like leading
        // `▁` for the first piece of every basic token and stores continuation
        // pieces without Hugging Face's textual `##` prefix. Keep a fallback
        // to ordinary `piece`/`##piece` vocabularies for older GGUFs and unit
        // fixtures, but prefer the representation this vocabulary proves.
        let uses_gguf_word_start = self
            .tokens
            .iter()
            .any(|token| token.text.starts_with(SPM_SPACE));
        let mut output = Vec::new();
        for word in basic {
            let chars = word.chars().collect::<Vec<_>>();
            if chars.len() > 100 {
                output.push(unk);
                continue;
            }
            let mut start = 0;
            let mut pieces = Vec::new();
            let mut failed = false;
            while start < chars.len() {
                let mut end = chars.len();
                let mut found = None;
                while start < end {
                    let body = chars[start..end].iter().collect::<String>();
                    let primary = if uses_gguf_word_start {
                        if start == 0 {
                            format!("{SPM_SPACE}{body}")
                        } else {
                            body.clone()
                        }
                    } else if start == 0 {
                        body.clone()
                    } else {
                        format!("##{body}")
                    };
                    let fallback = if start == 0 {
                        body
                    } else {
                        format!("##{body}")
                    };
                    if let Some(id) = self
                        .token_to_id
                        .get(&primary)
                        .or_else(|| self.token_to_id.get(&fallback))
                        .copied()
                    {
                        found = Some((id, end));
                        break;
                    }
                    end -= 1;
                }
                match found {
                    Some((id, next)) => {
                        pieces.push(id);
                        start = next;
                    }
                    None => {
                        failed = true;
                        break;
                    }
                }
            }
            if failed {
                output.push(unk);
            } else {
                output.extend(pieces);
            }
        }
        Ok(output)
    }

    fn decode_wordpiece(&self, token_ids: &[TokenId], remove_special: bool) -> Result<String> {
        let mut text = String::new();
        let uses_gguf_word_start = self
            .tokens
            .iter()
            .any(|token| token.text.starts_with(SPM_SPACE));
        for &id in token_ids {
            let token = self.tokens.get(id as usize).ok_or_else(|| {
                BackendError::InvalidTokenizerMetadata(format!("token id {id} out of range"))
            })?;
            if remove_special
                && (self.is_special(id)
                    || token.kind == TokenKind::Control
                    || token.text.starts_with('[') && token.text.ends_with(']'))
            {
                continue;
            }
            if let Some(suffix) = token.text.strip_prefix("##") {
                text.push_str(suffix);
            } else if let Some(word) = token.text.strip_prefix(SPM_SPACE) {
                if is_single_bert_punctuation(word) {
                    text.push_str(word);
                } else {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(word);
                }
            } else if is_single_bert_punctuation(&token.text) || uses_gguf_word_start {
                text.push_str(&token.text);
            } else {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&token.text);
            }
        }
        Ok(text)
    }

    fn encode_bpe_text(&self, text: &str, parse_special: bool) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        let mut byte_start = 0;

        while byte_start < text.len() {
            // llama.cpp's special-token partition runs in BOTH modes: USER_DEFINED
            // ("added") tokens are matched in raw text unconditionally; only
            // CONTROL tokens are gated by `parse_special`. Verified against the
            // qwen35 oracle (ITEM1 tokenizer gate): `--no-parse-special` still
            // yields single ids for <think>/<tool_call> (USER_DEFINED), while
            // <|im_start|> (CONTROL) tokenizes as text.
            if let Some((token_text, token_len)) =
                self.longest_control_token_at(text, byte_start, parse_special)
            {
                if let Some(id) = self.token_to_id.get(token_text) {
                    out.push(*id);
                    byte_start += token_len;
                    continue;
                }
            }

            let byte_end = self
                .next_control_token_start(text, byte_start, parse_special)
                .unwrap_or(text.len());

            let segments = match self.bpe_pre_tokenizer {
                BpePreTokenizer::Gpt4o => bpe_pretokenize_gpt4o(
                    &text[byte_start..byte_end],
                    self.bpe_pre_tokenizer.digit_group_max(),
                ),
                pre_tokenizer => bpe_pretokenize_with(
                    &text[byte_start..byte_end],
                    pre_tokenizer.digit_group_max(),
                    pre_tokenizer.fold_marks(),
                ),
            };
            for segment in segments {
                self.encode_bpe_segment(segment, &mut out)?;
            }
            byte_start = byte_end;
        }

        Ok(out)
    }

    fn encode_bpe_segment(&self, segment: &str, out: &mut Vec<TokenId>) -> Result<()> {
        if segment.is_empty() {
            return Ok(());
        }

        let mut symbols: Vec<String> = segment
            .as_bytes()
            .iter()
            .map(|byte| bpe_byte_to_char(*byte).to_string())
            .collect();

        symbols = self.bpe_registry.merge_symbols(symbols);

        for symbol in symbols {
            let id = self.token_to_id.get(&symbol).copied().ok_or_else(|| {
                BackendError::InvalidTokenizerMetadata(format!(
                    "GPT-2/BPE token {symbol:?} is missing from tokenizer.ggml.tokens"
                ))
            })?;
            out.push(id);
        }
        Ok(())
    }

    fn decode_bpe(&self, token_ids: &[TokenId], remove_special: bool) -> Result<String> {
        let mut bytes = Vec::new();
        for id in token_ids {
            if remove_special && self.is_special(*id) {
                continue;
            }
            let token = self.tokens.get(*id as usize).ok_or_else(|| {
                BackendError::InvalidTokenizerMetadata(format!("token id {id} out of range"))
            })?;
            if remove_special && (token.kind == TokenKind::Control || is_chat_control_marker(token))
            {
                continue;
            }
            for ch in token.text.chars() {
                if let Some(byte) = bpe_char_to_byte(ch) {
                    bytes.push(byte);
                } else if !remove_special || token.kind != TokenKind::Control {
                    return Err(BackendError::InvalidTokenizerMetadata(format!(
                        "GPT-2/BPE token {:?} contains non-byte character {ch:?}",
                        token.text
                    )));
                }
            }
        }

        // A generated sequence can stop mid-multi-byte-character — e.g. truncated by
        // max_tokens partway through an emoji — leaving valid byte-tokens that don't yet
        // form complete UTF-8. That is normal model output, not corrupt tokenizer
        // metadata, so return the valid UTF-8 prefix and hold back the incomplete
        // trailing bytes instead of failing the whole request (the strict decode
        // surfaced as a 503 that hung the chat UI). Holding the bytes back — rather than
        // emitting a U+FFFD — lets the streaming re-decode append the character cleanly
        // once the next token completes it (a transient U+FFFD would break the
        // strip_prefix delta diff and duplicate the line). For complete sequences the
        // valid prefix is the whole string, byte-for-byte identical to a strict decode,
        // so token-AND-text parity is unaffected.
        match std::str::from_utf8(&bytes) {
            Ok(text) => Ok(text.to_string()),
            Err(err) => Ok(std::str::from_utf8(&bytes[..err.valid_up_to()])
                .unwrap_or("")
                .to_string()),
        }
    }

    fn normalize_spm_text(&self, text: &str, parse_special: bool) -> String {
        let mut normalized = String::new();
        if text.is_empty() {
            return normalized;
        }
        // Always prepend the dummy `▁` when add_space_prefix is set, including when
        // the text already begins with whitespace. This matches HF's SentencePiece
        // (Metaspace) tokenizer, which prepends unconditionally — verified bit-exact
        // against HF `tokenizers` for llama SPM (tests/runnable_tokenizer.rs). A prior
        // `!text.starts_with(char::is_whitespace)` guard suppressed the prefix on
        // leading-whitespace input and diverged from HF on those cases (RA-5).
        if self.config.add_space_prefix
            && !(parse_special && self.longest_control_token_at(text, 0, true).is_some())
        {
            normalized.push(SPM_SPACE);
        }
        for ch in text.chars() {
            if ch == ' ' {
                normalized.push(SPM_SPACE);
            } else {
                normalized.push(ch);
            }
        }
        if parse_special {
            normalized
        } else {
            self.add_dummy_prefix_after_control_tokens(&normalized)
        }
    }

    fn add_dummy_prefix_after_control_tokens(&self, text: &str) -> String {
        if !self.config.add_space_prefix || text.is_empty() {
            return text.to_string();
        }

        let mut normalized = String::with_capacity(text.len());
        let mut byte_start = 0;
        while byte_start < text.len() {
            if let Some((token_text, token_len)) =
                self.longest_control_token_at(text, byte_start, true)
            {
                normalized.push_str(token_text);
                byte_start += token_len;

                let rest = &text[byte_start..];
                let next_is_control = self
                    .longest_control_token_at(text, byte_start, true)
                    .is_some();
                let should_insert_dummy_prefix =
                    self.should_insert_dummy_after_control(token_text, rest, next_is_control);
                if should_insert_dummy_prefix {
                    normalized.push(SPM_SPACE);
                }
                continue;
            }

            let ch = text[byte_start..]
                .chars()
                .next()
                .expect("byte_start is in-bounds");
            normalized.push(ch);
            byte_start += ch.len_utf8();
        }
        normalized
    }

    /// True when `token_text` is a `<|…|>` chat-control marker whose trailing
    /// whitespace the reference strips. Deliberately narrow: SPM models whose turn
    /// scaffolding is `[INST]`/`<s>` (TinyLlama, Llama 2, Mistral) do not match the
    /// `<|…|>` shape, and BPE families (Qwen3, Llama 3) never reach this SPM path,
    /// so their committed tokenizations are untouched.
    fn chat_control_marker_rstrips(&self, token_text: &str) -> bool {
        self.tokens
            .iter()
            .any(|token| token.text == token_text && is_chat_control_marker(token))
    }

    fn should_insert_dummy_after_control(
        &self,
        token_text: &str,
        rest: &str,
        next_is_control: bool,
    ) -> bool {
        if rest.is_empty() || next_is_control {
            return false;
        }

        if self
            .token_text(self.special.bos)
            .is_some_and(|bos| token_text == bos)
            && rest.starts_with("[INST]")
            && self.token_to_id.contains_key("[INST]")
            && self.token_to_id.contains_key("[/INST]")
        {
            return false;
        }

        if token_text == "[INST]"
            && self.token_to_id.contains_key("[INST]")
            && self.token_to_id.contains_key("[/INST]")
        {
            return true;
        }

        !rest.starts_with(SPM_SPACE)
    }

    /// Longest special token whose text starts at `byte_start`. USER_DEFINED
    /// ("added") tokens always participate; CONTROL tokens only when
    /// `include_control` (llama.cpp's `parse_special` partition rule).
    fn longest_control_token_at<'a>(
        &'a self,
        text: &str,
        byte_start: usize,
        include_control: bool,
    ) -> Option<(&'a str, usize)> {
        if !text.is_char_boundary(byte_start) {
            return None;
        }

        // USER_DEFINED ("added") tokens always match, mirroring llama.cpp's
        // special-token partition. Qwen3/qwen35 mark <think>/</think>
        // (and many <|...|> markers) as USER_DEFINED (type 4) rather than CONTROL
        // (type 3); without matching USER_DEFINED, a rendered ChatML template's
        // literal "</think>" tokenizes as text instead of the single special
        // token, and chat generation diverges from the reference. CONTROL tokens
        // participate only under `include_control` (= `parse_special`).
        self.tokens
            .iter()
            .filter(|token| {
                matches!(token.kind, TokenKind::UserDefined)
                    || (include_control && matches!(token.kind, TokenKind::Control))
            })
            .filter(|token| !token.text.is_empty())
            .filter(|token| text[byte_start..].starts_with(&token.text))
            .max_by_key(|token| token.text.len())
            .map(|token| (token.text.as_str(), token.text.len()))
    }

    fn encode_piece(&self, piece: &str, parse_special: bool) -> Result<Vec<TokenId>> {
        // SPM used to short-circuit to `encode_piece_greedy` here whenever specials
        // were not being parsed. That encoder is longest-match-first, which is a
        // different algorithm from the reference's score-ordered bigram merge and
        // segments ordinary words differently (`thunderstorm` -> `stor|m` instead of
        // `st|orm`). Fall through so every SPM segment reaches the ported session.

        let mut out = Vec::new();
        let mut byte_start = 0;
        // Set when a control token asked for the SPM dummy prefix. It is applied by
        // PREPENDING `▁` to the next raw segment, never by pushing the bare `▁` id:
        // the reference merges the prefix into the following word (`▁What` = one
        // token), whereas emitting the standalone id splits it into `▁` + `What` and
        // diverges on every fragment that follows a special. `normalize_spm_text`'s
        // non-special path already prepends the character; this keeps both in step.
        let mut pending_dummy_prefix = false;
        while byte_start < piece.len() {
            if parse_special {
                if let Some((token_text, token_len)) =
                    self.longest_control_token_at(piece, byte_start, true)
                {
                    if let Some(id) = self.token_to_id.get(token_text) {
                        out.push(*id);
                        byte_start += token_len;
                        // `<|...|>` chat-control markers carry rstrip semantics: the
                        // reference drops the whitespace run that follows the marker
                        // (Phi-3 renders `<|user|>\n…`, and the reference tokenizes
                        // that as `<|user|>` + `▁What`, with no `<0x0A>`). Consume it
                        // before deciding on the dummy prefix, so the prefix attaches
                        // to the first real character of the fragment.
                        if self.chat_control_marker_rstrips(token_text) {
                            let trimmed = piece[byte_start..].trim_start_matches(|c: char| {
                                c == SPM_SPACE || c.is_ascii_whitespace()
                            });
                            byte_start = piece.len() - trimmed.len();
                        }
                        let rest = &piece[byte_start..];
                        let next_is_control = self
                            .longest_control_token_at(piece, byte_start, true)
                            .is_some();
                        if self.config.add_space_prefix
                            && self.should_insert_dummy_after_control(
                                token_text,
                                rest,
                                next_is_control,
                            )
                        {
                            pending_dummy_prefix = true;
                        }
                        continue;
                    }
                }
            }

            let byte_end = if parse_special {
                self.next_control_token_start(piece, byte_start, true)
                    .unwrap_or(piece.len())
            } else {
                piece.len()
            };
            let segment = &piece[byte_start..byte_end];
            let prefixed;
            let segment = if pending_dummy_prefix {
                pending_dummy_prefix = false;
                prefixed = format!("{SPM_SPACE}{segment}");
                prefixed.as_str()
            } else {
                segment
            };
            self.encode_spm_segment(segment, &mut out)?;
            byte_start = byte_end;
        }
        Ok(out)
    }

    fn next_control_token_start(
        &self,
        text: &str,
        byte_start: usize,
        include_control: bool,
    ) -> Option<usize> {
        text[byte_start..]
            .char_indices()
            .map(|(offset, _)| byte_start + offset)
            .find(|idx| {
                self.longest_control_token_at(text, *idx, include_control)
                    .is_some()
            })
    }

    fn encode_spm_segment(&self, segment: &str, out: &mut Vec<TokenId>) -> Result<()> {
        if segment.is_empty() {
            return Ok(());
        }

        // SPM (no merge ranks) follows the reference session verbatim; only the
        // rank-based BPE families (gemma4 et al) use the symbol-merge path below.
        if self.bpe_ranks.is_empty() {
            return self.encode_spm_segment_reference(segment, out);
        }

        let symbols = self
            .bpe_registry
            .merge_symbols(segment.chars().map(|ch| ch.to_string()).collect());

        let mut unresolved = String::new();
        for symbol in symbols {
            // The multi-space (▁▁) deferral belongs to the score-merge path
            // only. Rank-based BPE (merges present, e.g. the gemma4 family)
            // merges multi-space runs into single vocab tokens — llama.cpp's
            // GEMMA4 BPE emits e.g. ▁▁ / ▁▁▁ tokens, proven by the
            // DiffusionGemma tokenizer-parity gate (tests/dg_tokenizer_parity.rs),
            // and deferring them here diverges from the reference.
            if self.bpe_ranks.is_empty() && symbol.contains("▁▁") {
                unresolved.push_str(&symbol);
                continue;
            }

            if let Some(id) = self.token_to_id.get(&symbol).copied() {
                if !unresolved.is_empty() {
                    out.extend(self.encode_piece_greedy(&unresolved)?);
                    unresolved.clear();
                }
                out.push(id);
            } else {
                unresolved.push_str(&symbol);
            }
        }
        if !unresolved.is_empty() {
            out.extend(self.encode_piece_greedy(&unresolved)?);
        }
        Ok(())
    }

    /// SPM segmentation, ported from the reference `llm_tokenizer_spm_session`.
    ///
    /// A global "merge the best-scoring adjacent pair, recompute everything" loop is
    /// NOT equivalent to the reference and diverges on ordinary words (`thunderstorm`
    /// segmented `stor|m` instead of `st|orm`, `LRUCache` as `LR|UC|ache` instead of
    /// `L|RU|Cache`). Four properties have to hold together:
    ///
    /// 1. Symbols are a doubly-linked list merged IN PLACE (`left.n += right.n`,
    ///    `right.n = 0`), so indices stay stable and `rev_merge` can address them.
    /// 2. The queue is seeded ONCE with every adjacent pair; after a merge only
    ///    `(prev, left)` and `(left, next)` are offered. Re-deriving all pairs each
    ///    pass invents merges the reference never queued.
    /// 3. Popped entries are validated against the CURRENT symbol widths
    ///    (`left.n + right.n != size` -> stale, skip), so a pair whose operands were
    ///    already consumed cannot fire late.
    /// 4. Output goes through `resegment`: a merged span that is not itself a vocab
    ///    token is split back into the two operands that formed it via `rev_merge`,
    ///    recursively, and only a span with no recorded merge falls back to bytes.
    ///
    /// Tie-break matches the reference comparator: highest score wins; on equal
    /// scores the smaller left index wins.
    fn encode_spm_segment_reference(&self, segment: &str, out: &mut Vec<TokenId>) -> Result<()> {
        #[derive(Clone, Copy)]
        struct Symbol {
            start: usize,
            n: usize,
            prev: i64,
            next: i64,
        }

        #[derive(Clone, Copy)]
        struct Bigram {
            left: usize,
            right: usize,
            score: f32,
            size: usize,
        }
        impl PartialEq for Bigram {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score && self.left == other.left
            }
        }
        impl Eq for Bigram {}
        impl Ord for Bigram {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Max-heap: greatest score pops first; equal scores favour the
                // SMALLER left index, so reverse that half of the comparison.
                self.score
                    .partial_cmp(&other.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| other.left.cmp(&self.left))
            }
        }
        impl PartialOrd for Bigram {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut symbols: Vec<Symbol> = Vec::new();
        for (offset, ch) in segment.char_indices() {
            let index = symbols.len() as i64;
            let n = ch.len_utf8();
            symbols.push(Symbol {
                start: offset,
                n,
                prev: index - 1,
                next: if offset + n == segment.len() {
                    -1
                } else {
                    index + 1
                },
            });
        }
        if symbols.is_empty() {
            return Ok(());
        }

        let mut rev_merge: HashMap<String, (usize, usize)> = HashMap::new();
        let mut work: BinaryHeap<Bigram> = BinaryHeap::new();

        let try_add_bigram = |symbols: &Vec<Symbol>,
                              work: &mut BinaryHeap<Bigram>,
                              rev_merge: &mut HashMap<String, (usize, usize)>,
                              left: i64,
                              right: i64| {
            if left == -1 || right == -1 {
                return;
            }
            let (left, right) = (left as usize, right as usize);
            let start = symbols[left].start;
            let end = start + symbols[left].n + symbols[right].n;
            let Some(text) = segment.get(start..end) else {
                return;
            };
            let Some(id) = self.token_to_id.get(text).copied() else {
                return;
            };
            let Some(token) = self.tokens.get(id as usize) else {
                return;
            };
            work.push(Bigram {
                left,
                right,
                score: token.score,
                size: text.len(),
            });
            rev_merge.insert(text.to_string(), (left, right));
        };

        for i in 1..symbols.len() {
            try_add_bigram(&symbols, &mut work, &mut rev_merge, i as i64 - 1, i as i64);
        }

        while let Some(bigram) = work.pop() {
            let (left, right) = (bigram.left, bigram.right);
            // Stale: an operand was already absorbed, or the pair no longer spans
            // the width this entry was queued for.
            if symbols[left].n == 0
                || symbols[right].n == 0
                || symbols[left].n + symbols[right].n != bigram.size
            {
                continue;
            }

            symbols[left].n += symbols[right].n;
            symbols[right].n = 0;
            symbols[left].next = symbols[right].next;
            if symbols[right].next >= 0 {
                let next = symbols[right].next as usize;
                symbols[next].prev = left as i64;
            }

            let (prev, next) = (symbols[left].prev, symbols[left].next);
            try_add_bigram(&symbols, &mut work, &mut rev_merge, prev, left as i64);
            try_add_bigram(&symbols, &mut work, &mut rev_merge, left as i64, next);
        }

        // Walk the surviving chain, resegmenting each span. Iterative rather than
        // recursive (the reference recurses) so a pathological merge tree cannot
        // overflow the stack; the explicit stack reproduces the same left-then-right
        // pre-order.
        let mut index = 0_i64;
        while index != -1 {
            let symbol = symbols[index as usize];
            let mut pending = vec![(symbol.start, symbol.n)];
            while let Some((start, n)) = pending.pop() {
                let Some(text) = segment.get(start..start + n) else {
                    continue;
                };
                if let Some(id) = self.token_to_id.get(text).copied() {
                    out.push(id);
                    continue;
                }
                match rev_merge.get(text) {
                    // A recorded merge whose left operand still spans the whole
                    // parent would re-expand to itself forever. Unreachable for a
                    // well-formed vocab (a merge is only queued when the joined text
                    // IS a token, so the branch above already took it), but fail to
                    // bytes rather than hang if a vocab ever violates that.
                    Some(&(left, right)) if symbols[left].n < n => {
                        // Push right first so left is popped (emitted) first.
                        pending.push((symbols[right].start, symbols[right].n));
                        pending.push((symbols[left].start, symbols[left].n));
                    }
                    _ => self.encode_unknown_symbol_bytes(text, out)?,
                }
            }
            index = symbol.next;
        }
        Ok(())
    }

    fn encode_unknown_symbol_bytes(&self, symbol: &str, out: &mut Vec<TokenId>) -> Result<()> {
        for byte in symbol.as_bytes() {
            let id = self
                .byte_token_to_id
                .get(byte)
                .copied()
                .or(self.special.unk);
            match id {
                Some(id) => out.push(id),
                None => {
                    return Err(BackendError::InvalidTokenizerMetadata(format!(
                        "SPM byte fallback token <0x{byte:02X}> is missing"
                    )))
                }
            }
        }
        Ok(())
    }

    fn encode_piece_greedy(&self, piece: &str) -> Result<Vec<TokenId>> {
        let chars: Vec<(usize, char)> = piece.char_indices().collect();
        let mut out = Vec::new();
        let mut byte_start = 0;

        while byte_start < piece.len() {
            let mut best: Option<(usize, TokenId, f32)> = None;
            for byte_end in piece[byte_start..]
                .char_indices()
                .skip(1)
                .map(|(offset, _)| byte_start + offset)
                .chain(std::iter::once(piece.len()))
            {
                let candidate = &piece[byte_start..byte_end];
                if candidate.contains("▁▁") {
                    continue;
                }
                if let Some(id) = self.token_to_id.get(candidate) {
                    let score = self.tokens[*id as usize].score;
                    let len = byte_end - byte_start;
                    match best {
                        Some((best_len, _, best_score))
                            if len < best_len || (len == best_len && score <= best_score) => {}
                        _ => best = Some((len, *id, score)),
                    }
                }
            }

            if let Some((len, id, _)) = best {
                out.push(id);
                byte_start += len;
                continue;
            }

            let ch = chars
                .iter()
                .find(|(idx, _)| *idx == byte_start)
                .map(|(_, ch)| *ch)
                .ok_or_else(|| {
                    BackendError::InvalidTokenizerMetadata(
                        "internal UTF-8 tokenizer cursor error".to_string(),
                    )
                })?;
            let mut buf = [0u8; 4];
            self.encode_unknown_symbol_bytes(ch.encode_utf8(&mut buf), &mut out)?;
            byte_start += ch.len_utf8();
        }
        Ok(out)
    }

    fn is_special(&self, id: TokenId) -> bool {
        self.special.bos == Some(id)
            || self.special.eos == Some(id)
            || self.special.eot == Some(id)
            || self.special.eom == Some(id)
            || self.special.sep == Some(id)
            || self.special.pad == Some(id)
            || self.special.mask == Some(id)
    }
}

#[cfg(test)]
fn bpe_pretokenize(text: &str) -> Vec<&str> {
    // Test-only convenience wrapper: the default llama-bpe digit grouping.
    bpe_pretokenize_with(text, 3, false)
}

/// GPT4O pre-tokenizer ported from pinned llama.cpp
/// `LLAMA_VOCAB_PRE_TYPE_GPT4O`. llama.cpp executes an ECMAScript regex over
/// collapsed Unicode categories, not the more readable tokenizer.json regex in
/// the adjacent source comment. In particular, marks are not letters and only
/// ASCII `a-z`/`A-Z` affect the two ordered word alternatives.
fn bpe_pretokenize_gpt4o(text: &str, digit_group_max: usize) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut byte_start = 0;

    while byte_start < text.len() {
        let byte_end = next_gpt4o_segment_end(text, byte_start, digit_group_max);
        segments.push(&text[byte_start..byte_end]);
        byte_start = byte_end;
    }

    segments
}

fn next_gpt4o_segment_end(text: &str, byte_start: usize, digit_group_max: usize) -> usize {
    if let Some(end) = consume_gpt4o_word(text, byte_start) {
        return end;
    }

    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if is_gpt4o_number(ch) {
        return consume_digits(text, byte_start, digit_group_max);
    }
    if let Some(end) = consume_gpt4o_punctuation(text, byte_start) {
        return end;
    }
    if let Some(end) = consume_whitespace_with_newline(text, byte_start) {
        return end;
    }
    if is_whitespace(ch) {
        return consume_whitespace_before_nonspace(text, byte_start);
    }

    byte_start + ch.len_utf8()
}

fn consume_gpt4o_word(text: &str, byte_start: usize) -> Option<usize> {
    let first = next_char(text, byte_start)?;
    let word_start = if is_gpt4o_letter(first) {
        byte_start
    } else if first != '\r' && first != '\n' && !is_gpt4o_number(first) {
        let next_start = byte_start + first.len_utf8();
        let next = next_char(text, next_start)?;
        is_gpt4o_letter(next).then_some(next_start)?
    } else {
        return None;
    };

    let word_end = consume_gpt4o_first_word_alternative(text, word_start)
        .or_else(|| consume_gpt4o_second_word_alternative(text, word_start))?;
    Some(consume_contraction(text, word_end).unwrap_or(word_end))
}

fn consume_gpt4o_first_word_alternative(text: &str, byte_start: usize) -> Option<usize> {
    let mut upper_end = byte_start;
    let mut last_lower_like = None;
    while upper_end < text.len() {
        let ch = next_char(text, upper_end)?;
        if !is_gpt4o_upper_like(ch) {
            break;
        }
        if is_gpt4o_lower_like(ch) {
            last_lower_like = Some(upper_end);
        }
        upper_end += ch.len_utf8();
    }

    // The first regex alternative greedily consumes its upper-like run, then
    // backtracks only as far as needed to leave one lower-like code point.
    let lower_start = if next_char(text, upper_end).is_some_and(is_gpt4o_lower_like) {
        upper_end
    } else {
        last_lower_like?
    };

    let mut end = lower_start;
    while end < text.len() {
        let ch = next_char(text, end)?;
        if !is_gpt4o_lower_like(ch) {
            break;
        }
        end += ch.len_utf8();
    }
    (end > lower_start).then_some(end)
}

fn consume_gpt4o_second_word_alternative(text: &str, byte_start: usize) -> Option<usize> {
    let mut end = byte_start;
    while end < text.len() {
        let ch = next_char(text, end)?;
        if !is_gpt4o_upper_like(ch) {
            break;
        }
        end += ch.len_utf8();
    }
    if end == byte_start {
        return None;
    }
    while end < text.len() {
        let ch = next_char(text, end)?;
        if !is_gpt4o_lower_like(ch) {
            break;
        }
        end += ch.len_utf8();
    }
    Some(end)
}

fn is_gpt4o_upper_like(ch: char) -> bool {
    is_gpt4o_letter(ch) && !ch.is_ascii_lowercase()
}

fn is_gpt4o_lower_like(ch: char) -> bool {
    is_gpt4o_letter(ch) && !ch.is_ascii_uppercase()
}

fn is_gpt4o_letter(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

fn is_gpt4o_number(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

fn consume_gpt4o_punctuation(text: &str, byte_start: usize) -> Option<usize> {
    let first = next_char(text, byte_start)?;
    let mut cursor = if first == ' ' {
        let next_start = byte_start + first.len_utf8();
        let next = next_char(text, next_start)?;
        is_gpt4o_punctuation(next).then_some(next_start)?
    } else if is_gpt4o_punctuation(first) {
        byte_start
    } else {
        return None;
    };

    while cursor < text.len() {
        let ch = next_char(text, cursor)?;
        if !is_gpt4o_punctuation(ch) {
            break;
        }
        cursor += ch.len_utf8();
    }
    while cursor < text.len() {
        let ch = next_char(text, cursor)?;
        if !matches!(ch, '\r' | '\n' | '/') {
            break;
        }
        cursor += ch.len_utf8();
    }
    Some(cursor)
}

fn is_gpt4o_punctuation(ch: char) -> bool {
    !is_whitespace(ch) && !is_gpt4o_letter(ch) && !is_gpt4o_number(ch)
}

fn bpe_pretokenize_with(text: &str, digit_group_max: usize, fold_marks: bool) -> Vec<&str> {
    // GPT-2/BPE pre-tokenizer, mirroring llama.cpp's tiktoken-style regex without
    // pulling in a regex dependency:
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d)
    //   | [^\r\n\p{L}\p{N}]?\p{L}+
    //   | \p{N}{1,N}              (N = digit_group_max: 3 for llama-bpe, 1 for qwen2)
    //   |  ?[^\s\p{L}\p{N}]+[\r\n]*
    //   | \s*[\r\n]+
    //   | \s+(?!\S)
    //   | \s+
    // Keep the branch order identical: the whitespace branches intentionally
    // leave one prefix byte/char behind when that enables the next token to be
    // an optional-prefix letters or punctuation segment. The ONLY dialect
    // difference (llama-bpe vs qwen2) is the digit-run cap.
    let mut segments = Vec::new();
    let mut byte_start = 0;

    while byte_start < text.len() {
        let byte_end = next_llama_bpe_segment_end(text, byte_start, digit_group_max, fold_marks);
        segments.push(&text[byte_start..byte_end]);
        byte_start = byte_end;
    }

    segments
}

fn next_llama_bpe_segment_end(
    text: &str,
    byte_start: usize,
    digit_group_max: usize,
    fold_marks: bool,
) -> usize {
    if let Some(end) = consume_contraction(text, byte_start) {
        return end;
    }
    if let Some(end) = consume_optional_prefix_letters(text, byte_start, fold_marks) {
        return end;
    }

    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if is_number(ch) {
        return consume_digits(text, byte_start, digit_group_max);
    }
    if let Some(end) = consume_optional_space_punctuation(text, byte_start, fold_marks) {
        return end;
    }
    if let Some(end) = consume_whitespace_with_newline(text, byte_start) {
        return end;
    }
    if is_whitespace(ch) {
        return consume_whitespace_before_nonspace(text, byte_start);
    }

    byte_start + ch.len_utf8()
}

fn consume_contraction(text: &str, byte_start: usize) -> Option<usize> {
    ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
        .into_iter()
        .find_map(|suffix| {
            text[byte_start..]
                .get(..suffix.len())
                .filter(|candidate| candidate.eq_ignore_ascii_case(suffix))
                .map(|_| byte_start + suffix.len())
        })
}

fn consume_optional_prefix_letters(
    text: &str,
    byte_start: usize,
    fold_marks: bool,
) -> Option<usize> {
    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if is_letter_class(ch, fold_marks) {
        return Some(consume_letters(text, byte_start, fold_marks));
    }
    if ch == '\r' || ch == '\n' || is_number(ch) {
        return None;
    }

    let next_idx = byte_start + ch.len_utf8();
    let next = (next_idx < text.len()).then(|| next_char(text, next_idx))??;
    is_letter_class(next, fold_marks).then(|| consume_letters(text, next_idx, fold_marks))
}

fn consume_optional_space_punctuation(
    text: &str,
    byte_start: usize,
    fold_marks: bool,
) -> Option<usize> {
    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    let punctuation_start = if ch == ' ' {
        let next_idx = byte_start + ch.len_utf8();
        let next = (next_idx < text.len()).then(|| next_char(text, next_idx))??;
        if is_punctuation_for_bpe(next, fold_marks) {
            next_idx
        } else {
            return None;
        }
    } else if is_punctuation_for_bpe(ch, fold_marks) {
        byte_start
    } else {
        return None;
    };

    let mut idx = punctuation_start;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_punctuation_for_bpe(ch, fold_marks) {
            break;
        }
        idx += ch.len_utf8();
    }
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if ch != '\n' && ch != '\r' {
            break;
        }
        idx += ch.len_utf8();
    }
    Some(idx)
}

fn consume_whitespace_with_newline(text: &str, byte_start: usize) -> Option<usize> {
    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if !is_whitespace(ch) {
        return None;
    }

    let mut idx = byte_start;
    let mut last_newline_end = None;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_whitespace(ch) {
            break;
        }
        idx += ch.len_utf8();
        if ch == '\n' || ch == '\r' {
            last_newline_end = Some(idx);
        }
    }
    last_newline_end
}

fn consume_whitespace_before_nonspace(text: &str, byte_start: usize) -> usize {
    let whitespace_end = consume_whitespace(text, byte_start);
    if whitespace_end == text.len() {
        return whitespace_end;
    }

    // Implements \s+(?!\S): if a whitespace run is followed by a non-space,
    // leave one horizontal space for the optional-prefix branch that follows.
    let chars: Vec<(usize, char)> = text[byte_start..whitespace_end]
        .char_indices()
        .map(|(offset, ch)| (byte_start + offset, ch))
        .collect();
    if chars.len() > 1 {
        chars[chars.len() - 1].0
    } else {
        whitespace_end
    }
}

fn next_char(text: &str, byte_start: usize) -> Option<char> {
    text[byte_start..].chars().next()
}

fn is_letter(ch: char) -> bool {
    ch.is_alphabetic()
}

fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

fn is_whitespace(ch: char) -> bool {
    ch.is_whitespace()
}

fn is_bert_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            get_general_category(ch),
            GeneralCategory::ConnectorPunctuation
                | GeneralCategory::DashPunctuation
                | GeneralCategory::OpenPunctuation
                | GeneralCategory::ClosePunctuation
                | GeneralCategory::InitialPunctuation
                | GeneralCategory::FinalPunctuation
                | GeneralCategory::OtherPunctuation
        )
}

fn is_single_bert_punctuation(text: &str) -> bool {
    let mut chars = text.chars();
    matches!((chars.next(), chars.next()), (Some(ch), None) if is_bert_punctuation(ch))
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4dbf
            | 0x4e00..=0x9fff
            | 0x20000..=0x2a6df
            | 0x2a700..=0x2b73f
            | 0x2b740..=0x2b81f
            | 0x2b820..=0x2ceaf
            | 0xf900..=0xfaff
            | 0x2f800..=0x2fa1f
    )
}

/// Unicode `\p{M}` (Mn | Mc | Me) via a generated inclusive-range table — the
/// qwen35 pre-tokenizer folds these into the letter class. NOTE `is_letter`
/// (Rust `is_alphabetic` = derived Alphabetic) already covers the
/// Other_Alphabetic subset of marks (e.g. Devanagari matras, Arabic harakat);
/// this table adds the rest (viramas, NFD accents, enclosing marks), matching
/// the oracle's strict general-category `[\p{L}\p{M}]` on all gate fixtures.
fn is_mark(ch: char) -> bool {
    const MARK_RANGES: &[(u32, u32)] = include!("mark_ranges.rs.inc");
    let cp = ch as u32;
    MARK_RANGES
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// The pre-tokenizer's letter class: `\p{L}` for llama-bpe/qwen2,
/// `[\p{L}\p{M}]` for qwen35 (`fold_marks`).
fn is_letter_class(ch: char, fold_marks: bool) -> bool {
    is_letter(ch) || (fold_marks && is_mark(ch))
}

fn is_punctuation_for_bpe(ch: char, fold_marks: bool) -> bool {
    !is_whitespace(ch) && !is_letter_class(ch, fold_marks) && !is_number(ch)
}

fn consume_letters(text: &str, byte_start: usize, fold_marks: bool) -> usize {
    let mut idx = byte_start;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_letter_class(ch, fold_marks) {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn consume_digits(text: &str, byte_start: usize, max_digits: usize) -> usize {
    let mut idx = byte_start;
    let mut count = 0;
    while idx < text.len() && count < max_digits {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_number(ch) {
            break;
        }
        idx += ch.len_utf8();
        count += 1;
    }
    idx
}

fn consume_whitespace(text: &str, byte_start: usize) -> usize {
    let mut idx = byte_start;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_whitespace(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn bpe_byte_to_char(byte: u8) -> char {
    let byte = u32::from(byte);
    if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte) {
        return char::from_u32(byte).expect("visible byte maps to Unicode scalar");
    }

    let offset = (0..byte)
        .filter(|candidate| {
            !((33..=126).contains(candidate)
                || (161..=172).contains(candidate)
                || (174..=255).contains(candidate))
        })
        .count() as u32;
    char::from_u32(256 + offset).expect("GPT-2 byte fallback maps to Unicode scalar")
}

fn bpe_char_to_byte(ch: char) -> Option<u8> {
    (0..=u8::MAX).find(|byte| bpe_byte_to_char(*byte) == ch)
}

fn validate_token_id(name: &str, id: Option<TokenId>, len: usize) -> Result<()> {
    if let Some(id) = id {
        if id as usize >= len {
            return Err(BackendError::InvalidTokenizerMetadata(format!(
                "{name} token id {id} out of range for vocab size {len}"
            )));
        }
    }
    Ok(())
}

fn parse_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn flush_bytes(bytes: &mut Vec<u8>, text: &mut String) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    // SPM byte-fallback can likewise end mid-character when generation is truncated;
    // push the valid UTF-8 prefix and hold back any incomplete trailing bytes rather
    // than erroring. Identical to a strict decode for complete sequences (parity-safe).
    let taken = std::mem::take(bytes);
    match std::str::from_utf8(&taken) {
        Ok(decoded) => text.push_str(decoded),
        Err(err) => text.push_str(std::str::from_utf8(&taken[..err.valid_up_to()]).unwrap_or("")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        bpe_byte_to_char, bpe_pretokenize, bpe_pretokenize_gpt4o, bpe_pretokenize_with,
        is_chat_control_marker, is_exact_phi4_mini_q4km, is_mark, BpePreTokenizer, BpeRegistry,
        SpecialTokens, Token, TokenKind, Tokenizer, TokenizerConfig, TokenizerModel, SPM_SPACE,
    };
    use std::collections::{BTreeSet, HashMap};

    fn tok(text: &str, kind: TokenKind) -> Token {
        Token {
            id: 0,
            text: text.to_string(),
            score: 0.0,
            kind,
        }
    }

    fn tokenizer_with(
        model: TokenizerModel,
        tokens: Vec<Token>,
        special: SpecialTokens,
    ) -> Tokenizer {
        Tokenizer {
            model,
            bpe_pre_tokenizer: BpePreTokenizer::Llama3,
            token_to_id: tokens
                .iter()
                .map(|token| (token.text.clone(), token.id))
                .collect(),
            tokens,
            byte_token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            bpe_registry: BpeRegistry::default(),
            special,
            config: TokenizerConfig {
                add_bos: false,
                add_eos: false,
                add_sep: false,
                add_space_prefix: false,
                remove_extra_whitespaces: false,
            },
            chat_template: None,
        }
    }

    #[test]
    fn constraint_token_bytes_preserve_fragments_spaces_and_specials() {
        let spm_tokens = vec![
            Token {
                id: 0,
                text: "<0xC3>".to_string(),
                score: 0.0,
                kind: TokenKind::Byte,
            },
            Token {
                id: 1,
                text: format!("{SPM_SPACE}hello"),
                score: 0.0,
                kind: TokenKind::Normal,
            },
            Token {
                id: 2,
                text: "</s>".to_string(),
                score: 0.0,
                kind: TokenKind::Control,
            },
        ];
        let special = SpecialTokens {
            eos: Some(2),
            eog: BTreeSet::from([2]),
            ..SpecialTokens::default()
        };
        let spm = tokenizer_with(TokenizerModel::LlamaSpm, spm_tokens, special);
        assert_eq!(spm.constraint_token_bytes(0).unwrap(), vec![0xc3]);
        assert_eq!(spm.constraint_token_bytes(1).unwrap(), b" hello");
        assert_eq!(spm.constraint_token_bytes(2).unwrap(), b"\xff</s>".to_vec());

        let bpe = tokenizer_with(
            TokenizerModel::Gpt2Bpe,
            vec![Token {
                id: 0,
                text: bpe_byte_to_char(0xa9).to_string(),
                score: 0.0,
                kind: TokenKind::Normal,
            }],
            SpecialTokens::default(),
        );
        assert_eq!(bpe.constraint_token_bytes(0).unwrap(), vec![0xa9]);
    }

    #[test]
    fn bert_wordpiece_normalizes_splits_and_wraps_special_tokens() {
        let texts = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "cafe", ",", "camel", "##ids",
        ];
        let tokens = texts
            .iter()
            .enumerate()
            .map(|(id, text)| Token {
                id: id as u32,
                text: (*text).to_string(),
                score: 0.0,
                kind: if id < 4 {
                    TokenKind::Control
                } else {
                    TokenKind::Normal
                },
            })
            .collect();
        let special = SpecialTokens {
            bos: Some(2),
            eos: Some(3),
            sep: Some(3),
            unk: Some(1),
            pad: Some(0),
            eog: BTreeSet::from([3]),
            ..SpecialTokens::default()
        };
        let mut tokenizer = tokenizer_with(TokenizerModel::BertWordPiece, tokens, special);
        tokenizer.config.add_bos = true;
        tokenizer.config.add_sep = true;

        let encoded = tokenizer.encode("Café, camelids", true, false).unwrap();
        assert_eq!(encoded, vec![2, 4, 5, 6, 7, 3]);
        assert_eq!(tokenizer.decode(&encoded, true).unwrap(), "cafe, camelids");
    }

    #[test]
    fn llama3_bpe_signature_matches_only_the_llama3_vocab() {
        use super::is_llama3_bpe_signature;
        // Minimal stand-in for the Llama-3 vocab: 128,256 entries with the five
        // stable chat markers at their canonical ids.
        let mut v = vec![String::new(); 128_256];
        v[128_000] = "<|begin_of_text|>".to_string();
        v[128_001] = "<|end_of_text|>".to_string();
        v[128_006] = "<|start_header_id|>".to_string();
        v[128_007] = "<|end_header_id|>".to_string();
        v[128_009] = "<|eot_id|>".to_string();
        assert!(is_llama3_bpe_signature(&v));

        // Wrong vocab size never matches (Qwen's 151,936, or a truncated vocab).
        assert!(!is_llama3_bpe_signature(&v[..128_255]));
        let qwen_sized = vec!["x".to_string(); 151_936];
        assert!(!is_llama3_bpe_signature(&qwen_sized));

        // Missing any core marker fails — this is what refuses a de-pre'd non-Llama-3
        // vocab that happens to be 128,256 tokens.
        let mut missing = v.clone();
        missing[128_009] = "<|not_eot|>".to_string();
        assert!(!is_llama3_bpe_signature(&missing));
    }

    #[test]
    fn resolve_gpt2_pre_tokenizer_gates_the_missing_pre_recovery() {
        use super::{is_exact_phi4_mini_q4km, resolve_gpt2_pre_tokenizer, BpePreTokenizer};
        use crate::gguf::{GgufFile, GgufMetadataValue};
        use std::{collections::BTreeMap, path::PathBuf};
        let mut sig = vec![String::new(); 128_256];
        sig[128_000] = "<|begin_of_text|>".to_string();
        sig[128_001] = "<|end_of_text|>".to_string();
        sig[128_006] = "<|start_header_id|>".to_string();
        sig[128_007] = "<|end_header_id|>".to_string();
        sig[128_009] = "<|eot_id|>".to_string();
        let no_sig = vec!["x".to_string(); 128_256];

        // Explicit dialects resolve regardless of the vocab.
        assert!(matches!(
            resolve_gpt2_pre_tokenizer(Some("llama-bpe"), &no_sig, false),
            Ok(BpePreTokenizer::Llama3)
        ));
        assert!(matches!(
            resolve_gpt2_pre_tokenizer(Some("qwen2"), &no_sig, false),
            Ok(BpePreTokenizer::Qwen2)
        ));
        assert!(matches!(
            resolve_gpt2_pre_tokenizer(Some("qwen35"), &no_sig, false),
            Ok(BpePreTokenizer::Qwen35)
        ));
        assert!(matches!(
            resolve_gpt2_pre_tokenizer(Some("gpt-4o"), &no_sig, true),
            Ok(BpePreTokenizer::Gpt4o)
        ));
        assert!(resolve_gpt2_pre_tokenizer(Some("gpt-4o"), &no_sig, false).is_err());

        // Missing pre + Llama-3 signature => recovered as Llama3 (the fix).
        assert!(matches!(
            resolve_gpt2_pre_tokenizer(None, &sig, false),
            Ok(BpePreTokenizer::Llama3)
        ));

        // Missing pre WITHOUT the signature => still refused (guards a de-labeled Qwen).
        assert!(resolve_gpt2_pre_tokenizer(None, &no_sig, false).is_err());

        // An explicit-but-unknown pre is refused EVEN WITH the signature — we only
        // rescue an absent key, never override a stated (if unrecognized) dialect.
        assert!(resolve_gpt2_pre_tokenizer(Some("smaug-bpe"), &sig, false).is_err());

        let exact = GgufFile {
            path: PathBuf::from("Phi-4-mini-instruct-Q4_K_M.gguf"),
            version: 3,
            tensor_count: 0,
            metadata_count: 2,
            alignment: 32,
            data_start_offset: 0,
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_string(),
                    GgufMetadataValue::String("phi3".to_string()),
                ),
                (
                    "general.name".to_string(),
                    GgufMetadataValue::String("Phi 4 Mini Instruct".to_string()),
                ),
            ]),
            tensors: Vec::new(),
        };
        assert!(!is_exact_phi4_mini_q4km(&exact));
        let mut renamed = exact.clone();
        renamed.path = PathBuf::from("renamed.gguf");
        assert!(!is_exact_phi4_mini_q4km(&renamed));
    }

    // Parity gate for the missing-`pre` Llama-3 rescue: the exact Meta-Llama-3-8B GGUF
    // that omits tokenizer.ggml.pre must tokenize BYTE-IDENTICALLY to a known-good
    // pre=llama-bpe Llama-3 GGUF. Ignored by default (needs the multi-GB models); run
    // locally with both env vars set:
    //   CAMELID_LLAMA3_MISSING_PRE_GGUF, CAMELID_LLAMA3_REFERENCE_GGUF
    #[test]
    #[ignore = "needs real Llama-3 GGUFs; set CAMELID_LLAMA3_MISSING_PRE_GGUF and CAMELID_LLAMA3_REFERENCE_GGUF"]
    fn missing_pre_llama3_tokenizes_identically_to_reference() {
        use super::Tokenizer;
        let a = std::env::var("CAMELID_LLAMA3_MISSING_PRE_GGUF")
            .expect("set CAMELID_LLAMA3_MISSING_PRE_GGUF");
        let b = std::env::var("CAMELID_LLAMA3_REFERENCE_GGUF")
            .expect("set CAMELID_LLAMA3_REFERENCE_GGUF");
        let ga = crate::gguf::read_metadata(&a).expect("read missing-pre gguf");
        let gb = crate::gguf::read_metadata(&b).expect("read reference gguf");
        // Guard against a tautological pass: the reference MUST carry an explicit
        // pre=llama-bpe (a genuine oracle), and the file under test MUST actually omit
        // the key (so it drives the recovery branch, not the normal path). Without
        // these, pointing both env vars at missing-pre files would pass trivially.
        assert_eq!(
            gb.metadata_string("tokenizer.ggml.pre"),
            Some("llama-bpe"),
            "CAMELID_LLAMA3_REFERENCE_GGUF must be an explicit pre=llama-bpe oracle"
        );
        assert_eq!(
            ga.metadata_string("tokenizer.ggml.pre"),
            None,
            "CAMELID_LLAMA3_MISSING_PRE_GGUF must actually omit tokenizer.ggml.pre"
        );
        let ta = Tokenizer::from_gguf(&ga)
            .expect("missing-pre Llama-3 gguf must now load (llama-bpe recovered)");
        let tb = Tokenizer::from_gguf(&gb).expect("reference gguf loads");
        // This proves the RECOVERED tokenizer equals an explicit pre=llama-bpe oracle
        // over strings that exercise the split regex (contractions, multi-digit runs,
        // whitespace, unicode, chat markers). The llama-bpe-vs-qwen digit-grouping
        // discrimination itself is covered by qwen2_pretokenizer_splits_digits_singly.
        let battery = [
            "It's a test, don't you think? We'll see.",
            "1234567890 and 42 plus 007 and 100000",
            "The quick brown fox jumps over 13 lazy dogs.",
            "café — naïve — Zürich — 你好 — 🚀",
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>",
            "    leading and    inner   spaces\tand\ttabs",
        ];
        for s in battery {
            let ea = ta.encode(s, false, true).expect("encode missing-pre");
            let eb = tb.encode(s, false, true).expect("encode reference");
            assert_eq!(
                ea, eb,
                "token mismatch for {s:?}:\n  missing-pre: {ea:?}\n  reference:   {eb:?}"
            );
        }
    }

    #[test]
    fn chat_control_markers_are_stripped_but_think_tags_are_kept() {
        // Phi-3 / ChatML <|...|> markers are turn scaffolding → strippable.
        assert!(is_chat_control_marker(&tok(
            "<|end|>",
            TokenKind::UserDefined
        )));
        assert!(is_chat_control_marker(&tok(
            "<|assistant|>",
            TokenKind::UserDefined
        )));
        assert!(is_chat_control_marker(&tok(
            "<|im_end|>",
            TokenKind::UserDefined
        )));
        // Qwen3 reasoning tags are content (and <...>, not <|...|>) → preserved.
        assert!(!is_chat_control_marker(&tok(
            "<think>",
            TokenKind::UserDefined
        )));
        assert!(!is_chat_control_marker(&tok(
            "</think>",
            TokenKind::UserDefined
        )));
        // Normal/content tokens are never markers regardless of shape.
        assert!(!is_chat_control_marker(&tok("<|end|>", TokenKind::Normal)));
        assert!(!is_chat_control_marker(&tok(
            "hello",
            TokenKind::UserDefined
        )));
    }

    #[test]
    fn bpe_registry_uses_ranked_heap_priority_for_merges() {
        let registry = BpeRegistry::from_merges(vec![
            "a b".to_string(),
            "ab c".to_string(),
            "c d".to_string(),
            "abc d".to_string(),
        ]);

        assert_eq!(registry.len(), 4);
        assert_eq!(
            registry.merge_symbols(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ]),
            vec!["abcd".to_string()]
        );
    }

    #[test]
    fn bpe_registry_prefers_lowest_rank_over_leftmost_pair() {
        let registry = BpeRegistry::from_merges(vec![
            "b c".to_string(),
            "bc d".to_string(),
            "a b".to_string(),
        ]);

        assert_eq!(
            registry.merge_symbols(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ]),
            vec!["a".to_string(), "bcd".to_string()]
        );
    }

    #[test]
    fn llama_bpe_pretokenizer_matches_core_llama3_shapes() {
        assert_eq!(bpe_pretokenize("hello world"), vec!["hello", " world"]);
        assert_eq!(bpe_pretokenize("it's"), vec!["it", "'s"]);
        assert_eq!(bpe_pretokenize("WE'LL"), vec!["WE", "'LL"]);
        assert_eq!(bpe_pretokenize("1234"), vec!["123", "4"]);
        assert_eq!(bpe_pretokenize("  hello"), vec![" ", " hello"]);
        assert_eq!(bpe_pretokenize(" !\n\n"), vec![" !\n\n"]);
        assert_eq!(bpe_pretokenize("foo...bar"), vec!["foo", "...", "bar"]);
        assert_eq!(bpe_pretokenize("hi\n  "), vec!["hi", "\n", "  "]);
    }

    #[test]
    fn llama_bpe_pretokenizer_matches_llama3_regex_edge_cases() {
        let cases = [
            ("!hello", vec!["!hello"]),
            ("\thello", vec!["\thello"]),
            ("  \thello", vec!["  ", "\thello"]),
            ("don't", vec!["don", "'t"]),
            ("can'T", vec!["can", "'T"]),
            ("abc12345def", vec!["abc", "123", "45", "def"]),
            ("café déjà", vec!["café", " déjà"]),
            ("!!!\r\nnext", vec!["!!!\r\n", "next"]),
            ("line\r\n  next", vec!["line", "\r\n", " ", " next"]),
            ("tabs\t\tword", vec!["tabs", "\t", "\tword"]),
            ("\t!!!", vec!["\t", "!!!"]),
            ("  !!!", vec![" ", " !!!"]),
            ("hello🙂world", vec!["hello", "🙂world"]),
            ("1२٣4", vec!["1२٣", "4"]),
            ("   ", vec!["   "]),
            ("\r\nhello", vec!["\r\n", "hello"]),
        ];

        for (input, expected) in cases {
            assert_eq!(bpe_pretokenize(input), expected, "input {input:?}");
        }
    }

    #[test]
    fn gpt4o_pretokenizer_matches_pinned_llama_cpp_collapsed_regex() {
        #[derive(serde::Deserialize)]
        struct Case {
            id: String,
            input: String,
            expected_segments: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        struct Oracle {
            commit: String,
        }
        #[derive(serde::Deserialize)]
        struct Pack {
            pack_id: String,
            oracle: Oracle,
            cases: Vec<Case>,
        }

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/qa/prompt-packs/phi4-mini-gpt4o-tokenizer-oracle-v1.json"
        );
        let raw = std::fs::read_to_string(path).expect("read Phi-4 tokenizer oracle pack");
        let pack: Pack = serde_json::from_str(&raw).expect("parse Phi-4 tokenizer oracle pack");
        assert_eq!(pack.pack_id, "phi4-mini-gpt4o-tokenizer-oracle-v1");
        assert_eq!(
            pack.oracle.commit,
            "acd79d603cb2e1c84c0886137b80f1ad649b6857"
        );

        for case in pack.cases {
            assert_eq!(
                bpe_pretokenize_gpt4o(&case.input, BpePreTokenizer::Gpt4o.digit_group_max()),
                case.expected_segments,
                "case {} input {:?}",
                case.id,
                case.input
            );
        }
    }

    // Pinned llama.cpp `acd79d603` (build 9632) oracle vectors for the exact
    // Phi-4-mini Q4_K_M GGUF. Opt-in because the artifact is multi-gigabyte.
    #[test]
    #[ignore = "set CAMELID_PHI4_MINI_GGUF to the pinned Phi-4-mini Q4_K_M GGUF"]
    fn phi4_mini_gpt4o_tokenizer_matches_pinned_oracle() {
        use super::Tokenizer;

        let path = std::env::var("CAMELID_PHI4_MINI_GGUF")
            .expect("set CAMELID_PHI4_MINI_GGUF to the pinned Phi-4-mini Q4_K_M GGUF");
        let gguf = crate::gguf::read_metadata(&path).expect("read Phi-4-mini GGUF metadata");
        assert_eq!(gguf.architecture(), Some("phi3"));
        assert_eq!(gguf.model_name(), Some("Phi 4 Mini Instruct"));
        assert_eq!(gguf.metadata_string("tokenizer.ggml.pre"), Some("gpt-4o"));
        assert!(is_exact_phi4_mini_q4km(&gguf));
        let tokenizer = Tokenizer::from_gguf(&gguf).expect("load Phi-4-mini tokenizer");
        assert!(
            tokenizer.special.eog.contains(&200_020),
            "<|end|> must be an EOG token for Phi-4 chat turns"
        );

        #[derive(serde::Deserialize)]
        struct Case {
            id: String,
            input: String,
            input_utf8_hex: String,
            expected_token_ids: Vec<u32>,
        }
        #[derive(serde::Deserialize)]
        struct Oracle {
            command: String,
            commit: String,
        }
        #[derive(serde::Deserialize)]
        struct Pack {
            oracle: Oracle,
            cases: Vec<Case>,
        }

        let oracle_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/qa/prompt-packs/phi4-mini-gpt4o-tokenizer-oracle-v1.json"
        );
        let raw = std::fs::read_to_string(oracle_path).expect("read Phi-4 tokenizer oracle pack");
        let pack: Pack = serde_json::from_str(&raw).expect("parse Phi-4 tokenizer oracle pack");
        assert_eq!(
            pack.oracle.commit,
            "acd79d603cb2e1c84c0886137b80f1ad649b6857"
        );
        assert!(pack.oracle.command.contains("llama-tokenize"));

        for case in pack.cases {
            let captured_hex = case
                .input
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(
                captured_hex, case.input_utf8_hex,
                "case {} UTF-8 bytes",
                case.id
            );
            assert_eq!(
                tokenizer
                    .encode(&case.input, false, true)
                    .expect("encode Phi-4-mini input"),
                case.expected_token_ids,
                "pinned oracle token mismatch for case {} input {:?}",
                case.id,
                case.input
            );
        }
    }

    #[test]
    fn qwen2_pretokenizer_splits_digits_singly_but_matches_llama3_otherwise() {
        // The ONLY difference between the qwen2 (digit cap 1) and llama-bpe
        // (digit cap 3) pre-tokenizers is digit grouping: qwen2 emits each digit
        // as its own piece (`\p{N}`), llama-bpe groups runs of up to three
        // (`\p{N}{1,3}`). Verified against llama.cpp llama-vocab.cpp
        // LLAMA_VOCAB_PRE_TYPE_QWEN2 vs LLAMA_VOCAB_PRE_TYPE_LLAMA3.
        const QWEN2: usize = 1;
        const LLAMA3: usize = 3;

        // Digits split one-at-a-time under qwen2 …
        assert_eq!(
            bpe_pretokenize_with("1234", QWEN2, false),
            vec!["1", "2", "3", "4"]
        );
        assert_eq!(
            bpe_pretokenize_with("abc12345def", QWEN2, false),
            vec!["abc", "1", "2", "3", "4", "5", "def"]
        );
        // … while the rest of the grammar is byte-for-byte identical to llama-bpe.
        for input in [
            "hello world",
            "it's",
            "WE'LL",
            "  hello",
            "foo...bar",
            "café déjà",
            "line\r\n  next",
            "hello🙂world",
        ] {
            assert_eq!(
                bpe_pretokenize_with(input, QWEN2, false),
                bpe_pretokenize_with(input, LLAMA3, false),
                "non-digit input {input:?} must tokenize identically under both dialects"
            );
        }
    }

    #[test]
    fn qwen35_pretokenizer_folds_combining_marks_into_letter_runs() {
        // qwen35's regex letter branch is `[\p{L}\p{M}]+` and its punctuation
        // class excludes `\p{M}` (llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35).
        // Byte-exactness vs the oracle is held by the ITEM1 tokenizer gate
        // (qa/ornith/constrained-vram/RECEIPT_ITEM1_tokenizer.json); these lock
        // the split behavior at the unit level.
        const QWEN35: usize = 1;

        // NFD: base letter + U+0301 combining acute stays one letter run.
        assert_eq!(
            bpe_pretokenize_with("cafe\u{301} bar", QWEN35, true),
            vec!["cafe\u{301}", " bar"]
        );
        // Without folding (qwen2), the bare accent splits off as punctuation.
        assert_eq!(
            bpe_pretokenize_with("cafe\u{301} bar", QWEN35, false),
            vec!["cafe", "\u{301}", " bar"]
        );
        // Devanagari virama (U+094D, Mn but NOT Other_Alphabetic — invisible to
        // `char::is_alphabetic`) must not break a cluster: नमस्ते is one run.
        assert_eq!(
            bpe_pretokenize_with("\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947}", QWEN35, true),
            vec!["\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947}"]
        );
        // A mark with no preceding letter still starts a letter-class run.
        assert_eq!(
            bpe_pretokenize_with("\u{301}x", QWEN35, true),
            vec!["\u{301}x"]
        );
        // Punctuation runs stop at a mark (punctuation class excludes \p{M}).
        assert_eq!(
            bpe_pretokenize_with("!!\u{301}!!", QWEN35, true),
            vec!["!!", "\u{301}", "!!"]
        );
    }

    #[test]
    fn is_mark_matches_unicode_m_category_samples() {
        assert!(is_mark('\u{301}')); // combining acute (Mn)
        assert!(is_mark('\u{94D}')); // Devanagari virama (Mn)
        assert!(is_mark('\u{93E}')); // Devanagari matra (Mc)
        assert!(is_mark('\u{20DD}')); // enclosing circle (Me)
        assert!(!is_mark('a'));
        assert!(!is_mark('!'));
        assert!(!is_mark(' '));
        assert!(!is_mark('\u{4E2D}')); // CJK letter
    }
}
