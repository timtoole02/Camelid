//! Isolated, fail-closed admission probe for the official Gemma 4 26B-A4B
//! Q4_0-QAT MTP assistant experiment.
//!
//! This test deliberately does not participate in production model admission.
//! It proves the model-family and tokenizer preconditions before assistant
//! weights are downloaded or any MTP proposal reaches CLAIRE's verifier.
//!
//! The assistant artifacts must be the small metadata/tokenizer files from:
//!
//! `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant`
//! revision `9537141506fe8875b3ed45b264af13580cb29166`.
//!
//! Run explicitly with:
//!
//! ```text
//! CAMELID_GEMMA4_MTP_SOURCE_GGUF=/path/to/full/gemma-4-26B_q4_0-it.gguf \
//! CAMELID_GEMMA4_MTP_RUNTIME_GGUF=/path/to/sparse/gemma-4-26B_q4_0-it.gguf \
//! CAMELID_GEMMA4_MTP_CGHOST=/path/to/gemma-4-26B_q4_0-it.cghost \
//! CAMELID_GEMMA4_MTP_ASSISTANT_DIR=/path/to/assistant-small-files \
//! cargo test --test gemma4_mtp_pair_gate -- --ignored --nocapture
//! ```

use std::{
    collections::BTreeMap,
    env,
    fmt::Write as _,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use camelid::{
    gguf,
    ghost::GhostFile,
    model::{Gemma4Binding, LlamaModelConfig},
    tokenizer::Tokenizer,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const DEFAULT_SOURCE_TARGET: &str = "/Volumes/Untitled/models/gemma-4-26B_q4_0-it.gguf";
const DEFAULT_RUNTIME_TARGET: &str = "/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf";
const DEFAULT_CGHOST: &str = "/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost";
const DEFAULT_ASSISTANT_DIR: &str = "/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant";

const ASSISTANT_REPOSITORY: &str = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant";
const ASSISTANT_REVISION: &str = "9537141506fe8875b3ed45b264af13580cb29166";
const ASSISTANT_TOKENIZER_SHA256: &str =
    "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c";
const ASSISTANT_CONFIG_SHA256: &str =
    "23d2bc4a8920f24c23653ff6871437bbd95e52527bf50007aaad05b0b6cab510";
const ASSISTANT_TOKENIZER_CONFIG_SHA256: &str =
    "01f2ff1c21ef2e722891380323edcaecd9c86a776aeb9b40148e2f35e3cee4d3";
const ASSISTANT_MODEL_SIZE: u64 = 839_427_840;
const ASSISTANT_MODEL_SHA256: &str =
    "c082cc581c3ec90d70285c1a41c81544ff56cbc96650f16c900a280940655801";

// Initial official QAT target revision. A local file with the same byte length
// is not sufficient evidence: the full content hash is part of this isolated
// experiment's provenance gate.
const OFFICIAL_TARGET_REPOSITORY: &str = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf";
const OFFICIAL_TARGET_REVISION: &str = "dfc00409adc70be497fee9c90bfe76b3ee130f2e";
const OFFICIAL_TARGET_SIZE: u64 = 14_439_361_440;
const OFFICIAL_TARGET_SHA256: &str =
    "4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5";

const SHARED_VOCAB_SIZE: usize = 262_144;
const TARGET_HIDDEN_SIZE: u32 = 2_816;
const TARGET_LAYERS: u32 = 30;
const TARGET_EXPERTS: u32 = 128;
const TARGET_EXPERTS_PER_TOKEN: u32 = 8;
const TARGET_SHARED_KV_LAYERS: u32 = 0;
const ASSISTANT_SHARED_KV_LAYERS: u32 = 4;

// Both official configs declare the complete 262,144-entry vocabulary, and all
// six multimodal boundary markers are already inside that shared ID space.
// Consequently there are no understood target-only suffix entries to permit.
const ALLOWED_TARGET_ONLY_EXTRAS: &[(u32, &str)] = &[];

const SPECIAL_SENTINELS: &[(u32, &str)] = &[
    (0, "<pad>"),
    (1, "<eos>"),
    (2, "<bos>"),
    (3, "<unk>"),
    (4, "<mask>"),
    (46, "<|tool>"),
    (47, "<tool|>"),
    (48, "<|tool_call>"),
    (49, "<tool_call|>"),
    (50, "<|tool_response>"),
    (51, "<tool_response|>"),
    (52, "<|\"|>"),
    (98, "<|think|>"),
    (100, "<|channel>"),
    (101, "<channel|>"),
    (105, "<|turn>"),
    (106, "<turn|>"),
    (255_999, "<|image>"),
    (256_000, "<|audio>"),
    (258_880, "<|image|>"),
    (258_881, "<|audio|>"),
    (258_882, "<image|>"),
    (258_883, "<audio|>"),
];

const TEXT_PROBES: &[&str] = &[
    "Hello, Camelid!",
    " leading space",
    "café 日本語\nline 2",
    "<|turn>assistant<|channel>analysis<channel|>",
];

#[derive(Default)]
struct GateReport {
    blockers: Vec<String>,
    observations: Vec<String>,
}

impl GateReport {
    fn require(&mut self, condition: bool, blocker: impl FnOnce() -> String) {
        if !condition {
            self.blockers.push(blocker());
        }
    }

    fn blocker(&mut self, message: impl Into<String>) {
        self.blockers.push(message.into());
    }

    fn observe(&mut self, message: impl Into<String>) {
        self.observations.push(message.into());
    }

    fn finish(self) {
        for observation in &self.observations {
            println!("MTP_PAIR_GATE OBSERVE {observation}");
        }
        if self.blockers.is_empty() {
            println!("MTP_PAIR_GATE PASS");
            return;
        }

        let mut message = format!("MTP_PAIR_GATE FAIL ({} blockers)", self.blockers.len());
        for blocker in self.blockers {
            let _ = write!(message, "\n- {blocker}");
        }
        panic!("{message}");
    }
}

fn env_path(name: &str, fallback: &str) -> PathBuf {
    env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(fallback))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_file_prefix(path: &Path, len: u64) -> Result<String, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file.take(len));
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    let mut consumed = 0u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        consumed += read as u64;
    }
    if consumed != len {
        return Err(format!(
            "read only {consumed} of {len} requested bytes from {}",
            path.display()
        ));
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn json_at<'a>(root: &'a Value, pointer: &str) -> Result<&'a Value, String> {
    root.pointer(pointer)
        .ok_or_else(|| format!("assistant config is missing {pointer}"))
}

fn json_u32(root: &Value, pointer: &str) -> Result<u32, String> {
    let value = json_at(root, pointer)?
        .as_u64()
        .ok_or_else(|| format!("assistant config {pointer} is not an unsigned integer"))?;
    u32::try_from(value).map_err(|_| format!("assistant config {pointer} is larger than u32"))
}

/// Turn the HF vocabulary into a dense ID-indexed table. JSON string escapes
/// have already been decoded to Unicode scalar values by `tokenizers`; the
/// canonical piece is then its exact UTF-8 byte sequence. We intentionally do
/// not apply NFC/NFKC, whitespace, or SentencePiece-marker normalization: those
/// transformations can collapse distinct tokens and would make this safety
/// gate weaker.
fn dense_hf_vocabulary(
    tokenizer: &tokenizers::Tokenizer,
    report: &mut GateReport,
) -> Vec<Option<String>> {
    let vocab = tokenizer.get_vocab(true);
    let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
    let mut dense: Vec<Option<String>> = vec![None; max_id.saturating_add(1)];
    for (piece, id) in vocab {
        let slot = &mut dense[id as usize];
        if let Some(previous) = slot {
            if previous.as_bytes() != piece.as_bytes() {
                report.blocker(format!(
                    "assistant tokenizer has duplicate id {id}: {previous:?} versus {piece:?}"
                ));
            }
        } else {
            *slot = Some(piece);
        }
    }
    let holes: Vec<_> = dense
        .iter()
        .enumerate()
        .filter_map(|(id, piece)| piece.is_none().then_some(id as u32))
        .collect();
    if !holes.is_empty() {
        report.blocker(format!(
            "assistant tokenizer ID table has holes at {}",
            format_id_runs(&holes)
        ));
    }
    dense
}

fn format_id_runs(ids: &[u32]) -> String {
    if ids.is_empty() {
        return "[]".to_string();
    }
    let mut out = String::from("[");
    let mut start = ids[0];
    let mut end = ids[0];
    for &id in &ids[1..] {
        if id == end.saturating_add(1) {
            end = id;
            continue;
        }
        push_id_run(&mut out, start, end);
        out.push_str(", ");
        start = id;
        end = id;
    }
    push_id_run(&mut out, start, end);
    out.push(']');
    out
}

fn push_id_run(out: &mut String, start: u32, end: u32) {
    if start == end {
        let _ = write!(out, "{start}");
    } else {
        let _ = write!(out, "{start}..={end}");
    }
}

fn compare_full_vocabulary(
    target_pieces: &[String],
    assistant_pieces: &[Option<String>],
    report: &mut GateReport,
) {
    report.require(assistant_pieces.len() == SHARED_VOCAB_SIZE, || {
        format!(
            "assistant vocabulary has {} IDs, expected {SHARED_VOCAB_SIZE}",
            assistant_pieces.len()
        )
    });

    let shared_len = target_pieces.len().min(assistant_pieces.len());
    let mut mismatch_ids = Vec::new();
    let mut mismatch_details = Vec::new();
    for id in 0..shared_len {
        let Some(assistant_piece) = &assistant_pieces[id] else {
            mismatch_ids.push(id as u32);
            continue;
        };
        // Exact UTF-8 comparison is the normalized representation described in
        // dense_hf_vocabulary; `String` equality is equivalent but the byte form
        // makes the non-NFC behavior explicit.
        if target_pieces[id].as_bytes() != assistant_piece.as_bytes() {
            mismatch_ids.push(id as u32);
            if mismatch_details.len() < 64 {
                mismatch_details.push(format!(
                    "id {id}: target={:?}, assistant={assistant_piece:?}",
                    target_pieces[id]
                ));
            }
        }
    }

    if !mismatch_ids.is_empty() {
        report.blocker(format!(
            "normalized target/assistant ID->piece mismatches at {}; first details: {}",
            format_id_runs(&mismatch_ids),
            mismatch_details.join("; ")
        ));
    }

    if target_pieces.len() < assistant_pieces.len() {
        let missing: Vec<u32> = (target_pieces.len()..assistant_pieces.len())
            .map(|id| id as u32)
            .collect();
        report.blocker(format!(
            "target vocabulary is missing assistant IDs {}",
            format_id_runs(&missing)
        ));
    } else if target_pieces.len() > assistant_pieces.len() {
        let allowed: BTreeMap<u32, &str> = ALLOWED_TARGET_ONLY_EXTRAS.iter().copied().collect();
        let mut rejected = Vec::new();
        for (id, piece) in target_pieces
            .iter()
            .enumerate()
            .skip(assistant_pieces.len())
        {
            if allowed.get(&(id as u32)).copied() != Some(piece.as_str()) {
                rejected.push(id as u32);
            }
        }
        if !rejected.is_empty() {
            report.blocker(format!(
                "unrecognized target-only vocabulary IDs {} (no target-only suffix is permitted for this official pair)",
                format_id_runs(&rejected)
            ));
        }
    }

    report.observe(format!(
        "vocabulary compared: target={} assistant={} mismatches={}",
        target_pieces.len(),
        assistant_pieces.len(),
        mismatch_ids.len()
    ));
}

fn verify_specials_and_probes(
    target: &Tokenizer,
    assistant: &tokenizers::Tokenizer,
    report: &mut GateReport,
) {
    for &(expected_id, piece) in SPECIAL_SENTINELS {
        let target_piece = target
            .tokens
            .get(expected_id as usize)
            .map(|token| token.text.as_str());
        report.require(target_piece == Some(piece), || {
            format!("target special ID {expected_id} is {target_piece:?}, expected {piece:?}")
        });

        let assistant_piece = assistant.id_to_token(expected_id);
        report.require(assistant_piece.as_deref() == Some(piece), || {
            format!("assistant special ID {expected_id} is {assistant_piece:?}, expected {piece:?}")
        });

        match target.encode(piece, false, true) {
            Ok(ids) => report.require(ids == [expected_id], || {
                format!(
                    "target special encode {piece:?} produced {ids:?}, expected [{expected_id}]"
                )
            }),
            Err(error) => {
                report.blocker(format!("target special encode {piece:?} failed: {error}"))
            }
        }
        match assistant.encode(piece, false) {
            Ok(encoding) => {
                let ids = encoding.get_ids();
                report.require(ids == [expected_id], || {
                    format!(
                        "assistant special encode {piece:?} produced {ids:?}, expected [{expected_id}]"
                    )
                });
            }
            Err(error) => report.blocker(format!(
                "assistant special encode {piece:?} failed: {error}"
            )),
        }

        match target.decode(&[expected_id], false) {
            Ok(decoded) => report.require(decoded == piece, || {
                format!(
                    "target special decode [{expected_id}] produced {decoded:?}, expected {piece:?}"
                )
            }),
            Err(error) => report.blocker(format!(
                "target special decode [{expected_id}] failed: {error}"
            )),
        }
        match assistant.decode(&[expected_id], false) {
            Ok(decoded) => report.require(decoded == piece, || {
                format!(
                    "assistant special decode [{expected_id}] produced {decoded:?}, expected {piece:?}"
                )
            }),
            Err(error) => report.blocker(format!(
                "assistant special decode [{expected_id}] failed: {error}"
            )),
        }
    }

    for &probe in TEXT_PROBES {
        let target_ids = match target.encode(probe, false, true) {
            Ok(ids) => ids,
            Err(error) => {
                report.blocker(format!("target probe encode {probe:?} failed: {error}"));
                continue;
            }
        };
        let assistant_encoding = match assistant.encode(probe, false) {
            Ok(encoding) => encoding,
            Err(error) => {
                report.blocker(format!("assistant probe encode {probe:?} failed: {error}"));
                continue;
            }
        };
        let assistant_ids = assistant_encoding.get_ids();
        report.require(target_ids == assistant_ids, || {
            format!(
                "probe tokenization mismatch for {probe:?}: target={target_ids:?}, assistant={assistant_ids:?}"
            )
        });

        match (
            target.decode(&target_ids, false),
            assistant.decode(assistant_ids, false),
        ) {
            (Ok(target_text), Ok(assistant_text)) => {
                report.require(target_text == assistant_text, || {
                    format!(
                        "probe decode mismatch for {probe:?}: target={target_text:?}, assistant={assistant_text:?}"
                    )
                });
            }
            (target_result, assistant_result) => report.blocker(format!(
                "probe decode failed for {probe:?}: target={target_result:?}, assistant={assistant_result:?}"
            )),
        }
    }
}

/// Cheap first-stage falsification: reads only GGUF metadata plus the pinned
/// 32 MB assistant tokenizer. It deliberately makes no artifact-provenance
/// claim; the full gate below must still pass before an assistant is admitted.
#[test]
#[ignore = "diagnostic tokenizer comparator; the full provenance gate remains authoritative"]
fn official_qat_mtp_tokenizer_tables_and_sentinels_match() {
    let target_path = env_path("CAMELID_GEMMA4_MTP_SOURCE_GGUF", DEFAULT_SOURCE_TARGET);
    let assistant_dir = env_path("CAMELID_GEMMA4_MTP_ASSISTANT_DIR", DEFAULT_ASSISTANT_DIR);
    let tokenizer_path = assistant_dir.join("tokenizer.json");
    let mut report = GateReport::default();

    match sha256_file(&tokenizer_path) {
        Ok(actual) => report.require(actual == ASSISTANT_TOKENIZER_SHA256, || {
            format!(
                "assistant tokenizer.json sha256={actual}, expected {ASSISTANT_TOKENIZER_SHA256} ({ASSISTANT_REPOSITORY}@{ASSISTANT_REVISION})"
            )
        }),
        Err(error) => report.blocker(format!("could not hash assistant tokenizer: {error}")),
    }

    let target_file = match gguf::read_metadata(&target_path) {
        Ok(file) => file,
        Err(error) => {
            report.blocker(format!("target GGUF metadata parse failed: {error}"));
            report.finish();
            return;
        }
    };
    let target_pieces = match target_file.metadata_array_strings("tokenizer.ggml.tokens") {
        Ok(tokens) => tokens,
        Err(error) => {
            report.blocker(format!("target token table is unavailable: {error}"));
            report.finish();
            return;
        }
    };
    let target_tokenizer = match Tokenizer::from_gguf(&target_file) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            report.blocker(format!("target tokenizer construction failed: {error}"));
            report.finish();
            return;
        }
    };
    let assistant_tokenizer = match tokenizers::Tokenizer::from_file(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            report.blocker(format!("assistant tokenizer construction failed: {error}"));
            report.finish();
            return;
        }
    };
    let assistant_pieces = dense_hf_vocabulary(&assistant_tokenizer, &mut report);
    compare_full_vocabulary(&target_pieces, &assistant_pieces, &mut report);
    verify_specials_and_probes(&target_tokenizer, &assistant_tokenizer, &mut report);
    report.finish();
}

#[test]
#[ignore = "requires exact full Gemma 4 26B QAT source, its sparse+cghost runtime pair, and pinned official assistant tokenizer/config files"]
fn official_qat_mtp_assistant_pair_is_exact_and_tokenizer_identical() {
    let target_path = env_path("CAMELID_GEMMA4_MTP_SOURCE_GGUF", DEFAULT_SOURCE_TARGET);
    let runtime_path = env_path("CAMELID_GEMMA4_MTP_RUNTIME_GGUF", DEFAULT_RUNTIME_TARGET);
    let cghost_path = env_path("CAMELID_GEMMA4_MTP_CGHOST", DEFAULT_CGHOST);
    let assistant_dir = env_path("CAMELID_GEMMA4_MTP_ASSISTANT_DIR", DEFAULT_ASSISTANT_DIR);
    let tokenizer_path = assistant_dir.join("tokenizer.json");
    let config_path = assistant_dir.join("config.json");
    let tokenizer_config_path = assistant_dir.join("tokenizer_config.json");
    let assistant_model_path = assistant_dir.join("model.safetensors");
    let mut report = GateReport::default();

    let target_size = std::fs::metadata(&target_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    report.require(target_size == OFFICIAL_TARGET_SIZE, || {
        format!(
            "target size is {target_size}, expected official {OFFICIAL_TARGET_SIZE} ({OFFICIAL_TARGET_REPOSITORY}@{OFFICIAL_TARGET_REVISION})"
        )
    });

    // The repository's digest cache is keyed by canonical path, length, ns
    // mtime, device and inode. A changed source misses and is honestly hashed;
    // repeated experiment runs do not re-stream 14.4 GB from the T7.
    match camelid::receipt::sha256_file_hex_cached(&target_path) {
        Ok(sha256) => {
            report.observe(format!("target sha256={sha256}"));
            report.require(sha256 == OFFICIAL_TARGET_SHA256, || {
                format!(
                    "full source target sha256={sha256}, expected official {OFFICIAL_TARGET_SHA256} ({OFFICIAL_TARGET_REPOSITORY}@{OFFICIAL_TARGET_REVISION})"
                )
            });
        }
        Err(error) => report.blocker(format!("could not hash full source target: {error}")),
    }

    for (path, expected, label) in [
        (
            &tokenizer_path,
            ASSISTANT_TOKENIZER_SHA256,
            "tokenizer.json",
        ),
        (&config_path, ASSISTANT_CONFIG_SHA256, "config.json"),
        (
            &tokenizer_config_path,
            ASSISTANT_TOKENIZER_CONFIG_SHA256,
            "tokenizer_config.json",
        ),
    ] {
        match sha256_file(path) {
            Ok(actual) => report.require(actual == expected, || {
                format!(
                    "assistant {label} sha256={actual}, expected {expected} ({ASSISTANT_REPOSITORY}@{ASSISTANT_REVISION})"
                )
            }),
            Err(error) => report.blocker(format!("could not hash assistant {label}: {error}")),
        }
    }
    match camelid::receipt::sha256_file_hex_cached(&assistant_model_path) {
        Ok(actual) => report.require(actual == ASSISTANT_MODEL_SHA256, || {
            format!(
                "assistant model.safetensors sha256={actual}, expected {ASSISTANT_MODEL_SHA256} ({ASSISTANT_REPOSITORY}@{ASSISTANT_REVISION})"
            )
        }),
        Err(error) => report.blocker(format!(
            "could not hash assistant model.safetensors: {error}"
        )),
    }
    let assistant_model_size = std::fs::metadata(&assistant_model_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    report.require(assistant_model_size == ASSISTANT_MODEL_SIZE, || {
        format!(
            "assistant model.safetensors size={assistant_model_size}, expected {ASSISTANT_MODEL_SIZE} ({ASSISTANT_REPOSITORY}@{ASSISTANT_REVISION})"
        )
    });
    report.observe(format!(
        "assistant model bytes={assistant_model_size} (pinned BF16 QAT artifact)"
    ));

    let target_file = match gguf::read_metadata(&target_path) {
        Ok(file) => file,
        Err(error) => {
            report.blocker(format!("target GGUF metadata parse failed: {error}"));
            report.finish();
            return;
        }
    };
    report.require(target_file.architecture() == Some("gemma4"), || {
        format!(
            "target architecture is {:?}, expected gemma4",
            target_file.architecture()
        )
    });
    for (key, expected) in [
        ("gemma4.embedding_length", TARGET_HIDDEN_SIZE),
        ("gemma4.block_count", TARGET_LAYERS),
        ("gemma4.expert_count", TARGET_EXPERTS),
        ("gemma4.expert_used_count", TARGET_EXPERTS_PER_TOKEN),
        ("gemma4.attention.shared_kv_layers", TARGET_SHARED_KV_LAYERS),
    ] {
        let actual = target_file.metadata_u32(key);
        report.require(actual == Some(expected), || {
            format!("target {key} is {actual:?}, expected {expected}")
        });
    }
    report.require(
        target_file.metadata_string("tokenizer.ggml.model") == Some("gemma4"),
        || {
            format!(
                "target tokenizer.ggml.model is {:?}, expected gemma4",
                target_file.metadata_string("tokenizer.ggml.model")
            )
        },
    );

    // The runtime uses a sparse common-core GGUF plus .cghost, not the full T7
    // file. Prove the sparse header is byte-identical to the hash-pinned source,
    // then use the existing sampled identity to bind every common tensor and
    // every expert record to both paths. This is stronger than accepting the
    // sparse file merely because its logical length matches the official GGUF.
    let runtime_file = match gguf::read_metadata(&runtime_path) {
        Ok(file) => file,
        Err(error) => {
            report.blocker(format!(
                "runtime sparse GGUF metadata parse failed: {error}"
            ));
            report.finish();
            return;
        }
    };
    report.require(
        runtime_file.data_start_offset == target_file.data_start_offset,
        || {
            format!(
                "runtime/source data offsets differ: runtime={} source={}",
                runtime_file.data_start_offset, target_file.data_start_offset
            )
        },
    );
    if runtime_file.data_start_offset == target_file.data_start_offset {
        match (
            sha256_file_prefix(&target_path, target_file.data_start_offset),
            sha256_file_prefix(&runtime_path, runtime_file.data_start_offset),
        ) {
            (Ok(source_header), Ok(runtime_header)) => {
                report.observe(format!("source/runtime GGUF header sha256={source_header}"));
                report.require(source_header == runtime_header, || {
                    format!(
                        "runtime sparse GGUF header sha256={runtime_header}, source header sha256={source_header}"
                    )
                });
            }
            (source_result, runtime_result) => report.blocker(format!(
                "could not hash source/runtime GGUF headers: source={source_result:?}, runtime={runtime_result:?}"
            )),
        }
    }
    report.require(runtime_file.tensors == target_file.tensors, || {
        "runtime sparse GGUF tensor directory differs from the official full source".to_string()
    });

    let source_config = LlamaModelConfig::from_gguf(&target_file);
    let runtime_config = LlamaModelConfig::from_gguf(&runtime_file);
    match (source_config, runtime_config) {
        (Ok(source_config), Ok(runtime_config)) => {
            report.require(source_config == runtime_config, || {
                "runtime sparse GGUF model config differs from the official full source".to_string()
            });
            match source_config.gemma4.as_ref() {
                Some(gemma4) => {
                    report.require(gemma4.is_sliding_layer(28), || {
                        "target layer 28 must be the terminal sliding-attention KV source"
                            .to_string()
                    });
                    report.require(!gemma4.is_sliding_layer(29), || {
                        "target layer 29 must be the terminal full-attention KV source".to_string()
                    });
                    report.observe(
                        "shared-KV source map: assistant sliding layers -> target 28; assistant full layer -> target 29",
                    );
                }
                None => report.blocker("target config is missing Gemma 4 metadata"),
            }
            match (
                Gemma4Binding::bind(&target_file, &source_config),
                Gemma4Binding::bind(&runtime_file, &runtime_config),
                GhostFile::open(&cghost_path),
            ) {
                (Ok(source_binding), Ok(runtime_binding), Ok(ghost)) => {
                    report.require(ghost.has_sampled_source_identity(), || {
                        "runtime .cghost has no sampled source identity".to_string()
                    });
                    match ghost.validate_moe_source_identity(
                        &target_path,
                        &source_binding,
                        TARGET_EXPERTS as usize,
                    ) {
                        Ok(()) => report.observe(
                            ".cghost sampled identity matches hash-pinned full source target",
                        ),
                        Err(error) => report.blocker(format!(
                            ".cghost does not derive from the hash-pinned full source target: {error}"
                        )),
                    }
                    match ghost.validate_moe_source_identity(
                        &runtime_path,
                        &runtime_binding,
                        TARGET_EXPERTS as usize,
                    ) {
                        Ok(()) => report.observe(
                            ".cghost sampled identity matches sparse runtime target",
                        ),
                        Err(error) => report.blocker(format!(
                            ".cghost does not pair with the sparse runtime target: {error}"
                        )),
                    }
                }
                (source_binding, runtime_binding, ghost) => report.blocker(format!(
                    "could not establish Ghost runtime provenance: source_binding={:?}, runtime_binding={:?}, cghost={:?}",
                    source_binding.err(),
                    runtime_binding.err(),
                    ghost.err()
                )),
            }
        }
        (source_config, runtime_config) => report.blocker(format!(
            "could not parse source/runtime Gemma 4 configs: source={:?}, runtime={:?}",
            source_config.err(),
            runtime_config.err()
        )),
    }

    let runtime_pieces = match runtime_file.metadata_array_strings("tokenizer.ggml.tokens") {
        Ok(tokens) => tokens,
        Err(error) => {
            report.blocker(format!(
                "runtime sparse target token table is unavailable: {error}"
            ));
            Vec::new()
        }
    };
    let source_pieces_for_runtime = target_file
        .metadata_array_strings("tokenizer.ggml.tokens")
        .unwrap_or_default();
    report.require(runtime_pieces == source_pieces_for_runtime, || {
        "runtime sparse target token table differs from the hash-pinned full source".to_string()
    });

    let config: Value = match std::fs::read(&config_path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))
    {
        Ok(value) => value,
        Err(error) => {
            report.blocker(format!("assistant config parse failed: {error}"));
            report.finish();
            return;
        }
    };
    for (pointer, expected) in [
        ("/backbone_hidden_size", TARGET_HIDDEN_SIZE),
        ("/text_config/vocab_size", SHARED_VOCAB_SIZE as u32),
        (
            "/text_config/num_kv_shared_layers",
            ASSISTANT_SHARED_KV_LAYERS,
        ),
        ("/text_config/bos_token_id", 2),
        ("/text_config/eos_token_id", 1),
        ("/text_config/pad_token_id", 0),
    ] {
        match json_u32(&config, pointer) {
            Ok(actual) => report.require(actual == expected, || {
                format!("assistant config {pointer} is {actual}, expected {expected}")
            }),
            Err(error) => report.blocker(error),
        }
    }
    report.require(
        config.pointer("/model_type").and_then(Value::as_str) == Some("gemma4_assistant"),
        || "assistant config model_type is not gemma4_assistant".to_string(),
    );
    report.require(
        config
            .pointer("/text_config/model_type")
            .and_then(Value::as_str)
            == Some("gemma4_text"),
        || "assistant text config model_type is not gemma4_text".to_string(),
    );
    report.require(
        config.pointer("/dtype").and_then(Value::as_str) == Some("bfloat16"),
        || "assistant config dtype is not bfloat16 QAT checkpoint storage".to_string(),
    );
    let assistant_layer_types: Vec<&str> = config
        .pointer("/text_config/layer_types")
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    report.require(
        assistant_layer_types
            == [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        || {
            format!(
                "assistant layer types are {assistant_layer_types:?}, expected three sliding layers then one full layer"
            )
        },
    );
    for (pointer, expected) in [
        ("/boi_token_id", 255_999),
        ("/boa_token_id", 256_000),
        ("/image_token_id", 258_880),
        ("/audio_token_id", 258_881),
        ("/eoi_token_id", 258_882),
        ("/eoa_token_id", 258_883),
    ] {
        match json_u32(&config, pointer) {
            Ok(actual) => report.require(actual == expected, || {
                format!("assistant config {pointer} is {actual}, expected {expected}")
            }),
            Err(error) => report.blocker(error),
        }
    }
    let root_eos: Vec<u32> = config
        .pointer("/eos_token_id")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|value| u32::try_from(value).ok())
                .collect()
        })
        .unwrap_or_default();
    report.require(root_eos == [1, 106], || {
        format!("assistant root eos_token_id is {root_eos:?}, expected [1, 106] (<eos>, <turn|>)")
    });

    let target_pieces = match target_file.metadata_array_strings("tokenizer.ggml.tokens") {
        Ok(tokens) => tokens,
        Err(error) => {
            report.blocker(format!("target token table is unavailable: {error}"));
            report.finish();
            return;
        }
    };
    let target_tokenizer = match Tokenizer::from_gguf(&target_file) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            report.blocker(format!("target tokenizer construction failed: {error}"));
            report.finish();
            return;
        }
    };
    let assistant_tokenizer = match tokenizers::Tokenizer::from_file(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            report.blocker(format!("assistant tokenizer construction failed: {error}"));
            report.finish();
            return;
        }
    };
    let assistant_pieces = dense_hf_vocabulary(&assistant_tokenizer, &mut report);
    compare_full_vocabulary(&target_pieces, &assistant_pieces, &mut report);

    report.require(target_tokenizer.special.pad == Some(0), || {
        format!(
            "target pad token is {:?}, expected Some(0)",
            target_tokenizer.special.pad
        )
    });
    report.require(target_tokenizer.special.eos == Some(1), || {
        format!(
            "target eos token is {:?}, expected Some(1)",
            target_tokenizer.special.eos
        )
    });
    report.require(target_tokenizer.special.bos == Some(2), || {
        format!(
            "target bos token is {:?}, expected Some(2)",
            target_tokenizer.special.bos
        )
    });
    report.require(target_tokenizer.special.unk == Some(3), || {
        format!(
            "target unknown token is {:?}, expected Some(3)",
            target_tokenizer.special.unk
        )
    });
    report.require(target_tokenizer.special.mask == Some(4), || {
        format!(
            "target mask token is {:?}, expected Some(4)",
            target_tokenizer.special.mask
        )
    });

    verify_specials_and_probes(&target_tokenizer, &assistant_tokenizer, &mut report);
    report.finish();
}
