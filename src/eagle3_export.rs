//! Durable on-disk interchange for exact-Q4 EAGLE-3 teacher features.
//!
//! The format intentionally uses raw little-endian arrays plus JSON manifests. NumPy/MLX can
//! memory-map these files directly, while the Rust exporter avoids a Python/NPZ dependency and
//! can hash every payload independently. See `docs/EAGLE3_Q4_FEATURE_FORMAT.md`.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::eagle3::{DRAFT_VOCAB_SIZE, HIDDEN_SIZE, TARGET_LAYER_INPUT_IDS, TARGET_VOCAB_SIZE};
use crate::error::{BackendError, Result};

pub const SCHEMA_ID: &str = "camelid-eagle3-q4-features-v1";
pub const POSITIONAL_CONTRACT: &str = "eagle3-aux-p-next-token-teacher-p1-v1";
pub const AUX_WIDTH: usize = HIDDEN_SIZE * TARGET_LAYER_INPUT_IDS.len();
pub const INVALID_TOKEN_ID: u32 = u32::MAX;

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::RuntimeShapeMismatch(message.into())
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShiftedTeacherRows {
    pub target_argmax: Vec<u32>,
    pub teacher_draft_logits: Vec<f32>,
    pub teacher_logsumexp: Vec<f32>,
}

/// Shift capture-row teacher outputs into the EAGLE base-row contract. Base row `P`
/// consumes target features at `P` and token `P+1`, so its teacher distribution is the
/// target output produced after capture row `P+1`. The final base row has no such capture.
pub fn shift_teacher_rows(
    capture_argmax: &[u32],
    capture_draft_logits: &[f32],
    capture_logsumexp: &[f32],
) -> Result<ShiftedTeacherRows> {
    let rows = capture_argmax.len();
    if rows < 2
        || capture_draft_logits.len() != rows * DRAFT_VOCAB_SIZE
        || capture_logsumexp.len() != rows
    {
        return Err(invalid(format!(
            "cannot shift EAGLE teacher rows: argmax={}, logits={}, logsumexp={}",
            rows,
            capture_draft_logits.len(),
            capture_logsumexp.len()
        )));
    }
    let mut target_argmax = vec![INVALID_TOKEN_ID; rows];
    target_argmax[..rows - 1].copy_from_slice(&capture_argmax[1..]);
    let mut teacher_draft_logits = vec![0.0f32; rows * DRAFT_VOCAB_SIZE];
    teacher_draft_logits[..(rows - 1) * DRAFT_VOCAB_SIZE]
        .copy_from_slice(&capture_draft_logits[DRAFT_VOCAB_SIZE..]);
    let mut teacher_logsumexp = vec![f32::NAN; rows];
    teacher_logsumexp[..rows - 1].copy_from_slice(&capture_logsumexp[1..]);
    Ok(ShiftedTeacherRows {
        target_argmax,
        teacher_draft_logits,
        teacher_logsumexp,
    })
}

pub fn shifted_corpus_labels(input_ids: &[u32]) -> Result<Vec<u32>> {
    if input_ids.len() < 2 {
        return Err(invalid(
            "EAGLE corpus labels require at least two input ids",
        ));
    }
    let mut labels = vec![INVALID_TOKEN_ID; input_ids.len()];
    if input_ids.len() > 2 {
        labels[..input_ids.len() - 2].copy_from_slice(&input_ids[2..]);
    }
    Ok(labels)
}

/// Convert the canonical token-position corpus mask into the EAGLE base-row mask. Raw row `Q`
/// marks whether token `Q` belongs to the trainable assistant span; base row `P`
/// learns the teacher distribution that predicts token `P+2`, so it consumes raw row `P+2`.
pub fn shift_loss_mask(raw_loss_mask: &[u8]) -> Result<Vec<u8>> {
    if raw_loss_mask.len() < 2 || raw_loss_mask.iter().any(|&value| value > 1) {
        return Err(invalid(
            "EAGLE raw loss mask requires at least two binary entries",
        ));
    }
    let mut shifted = vec![0u8; raw_loss_mask.len()];
    let supervised_rows = raw_loss_mask.len() - 2;
    shifted[..supervised_rows].copy_from_slice(&raw_loss_mask[2..]);
    Ok(shifted)
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Eagle3FeatureInput {
    #[serde(default)]
    pub id: Option<String>,
    pub input_ids: Vec<u32>,
    pub loss_mask: Vec<u8>,
}

impl Eagle3FeatureInput {
    pub fn validate(&self) -> Result<()> {
        if self.input_ids.len() < 2 {
            return Err(invalid(format!(
                "EAGLE feature input needs at least 2 tokens for the P/P+1 pairing, got {}",
                self.input_ids.len()
            )));
        }
        if self.loss_mask.len() != self.input_ids.len() {
            return Err(invalid(format!(
                "EAGLE feature input has {} token ids but {} loss-mask entries",
                self.input_ids.len(),
                self.loss_mask.len()
            )));
        }
        if let Some((index, value)) = self
            .loss_mask
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value > 1)
        {
            return Err(invalid(format!(
                "EAGLE loss_mask[{index}] is {value}; only 0 or 1 is valid"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Eagle3FeatureSample {
    pub id: String,
    pub input_ids: Vec<u32>,
    pub labels: Vec<u32>,
    pub target_argmax: Vec<u32>,
    pub loss_mask: Vec<u8>,
    pub aux_layer_inputs: Vec<f32>,
    pub hidden_state: Vec<f32>,
    pub input_embedding: Vec<f32>,
    pub next_token_embedding: Vec<f32>,
    pub teacher_draft_logits: Vec<f32>,
    pub teacher_logsumexp: Vec<f32>,
    pub bootstrap_rows_masked: usize,
}

impl Eagle3FeatureSample {
    pub fn validate(&self) -> Result<()> {
        let rows = self.input_ids.len();
        let require_rows = |name: &str, actual: usize| -> Result<()> {
            if actual != rows {
                return Err(invalid(format!(
                    "EAGLE sample {} has {actual} {name} rows, expected {rows}",
                    self.id
                )));
            }
            Ok(())
        };
        require_rows("label", self.labels.len())?;
        require_rows("target-argmax", self.target_argmax.len())?;
        require_rows("loss-mask", self.loss_mask.len())?;
        if self.aux_layer_inputs.len() != rows * AUX_WIDTH {
            return Err(invalid(format!(
                "EAGLE sample {} aux stream has {} values, expected {}",
                self.id,
                self.aux_layer_inputs.len(),
                rows * AUX_WIDTH
            )));
        }
        for (name, values) in [
            ("hidden_state", &self.hidden_state),
            ("input_embedding", &self.input_embedding),
            ("next_token_embedding", &self.next_token_embedding),
        ] {
            if values.len() != rows * HIDDEN_SIZE {
                return Err(invalid(format!(
                    "EAGLE sample {} {name} has {} values, expected {}",
                    self.id,
                    values.len(),
                    rows * HIDDEN_SIZE
                )));
            }
        }
        if self.teacher_draft_logits.len() != rows * DRAFT_VOCAB_SIZE {
            return Err(invalid(format!(
                "EAGLE sample {} teacher draft logits have {} values, expected {}",
                self.id,
                self.teacher_draft_logits.len(),
                rows * DRAFT_VOCAB_SIZE
            )));
        }
        require_rows("teacher-logsumexp", self.teacher_logsumexp.len())?;
        if self.bootstrap_rows_masked != 0
            || self.bootstrap_rows_masked > rows
            || self.loss_mask[..self.bootstrap_rows_masked]
                .iter()
                .any(|&value| value != 0)
        {
            return Err(invalid(format!(
                "EAGLE sample {} must not contain missing bootstrap rows (got {})",
                self.id, self.bootstrap_rows_masked
            )));
        }
        if rows < 2
            || self.labels[rows - 2..]
                .iter()
                .any(|&label| label != INVALID_TOKEN_ID)
            || self.loss_mask[rows - 2..].iter().any(|&mask| mask != 0)
            || self.target_argmax.last().copied() != Some(INVALID_TOKEN_ID)
        {
            return Err(invalid(format!(
                "EAGLE sample {} must have two trailing invalid corpus labels, one trailing invalid teacher argmax, and two trailing zero loss-mask sentinels",
                self.id
            )));
        }
        for row in 0..rows.saturating_sub(1) {
            let next = &self.next_token_embedding[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE];
            let shifted = &self.input_embedding[(row + 1) * HIDDEN_SIZE..(row + 2) * HIDDEN_SIZE];
            if next
                .iter()
                .zip(shifted)
                .any(|(left, right)| left.to_bits() != right.to_bits())
            {
                return Err(invalid(format!(
                    "EAGLE sample {} next-token embedding row {row} does not exactly equal input-embedding row {}",
                    self.id,
                    row + 1
                )));
            }
        }
        if self.next_token_embedding[(rows - 1) * HIDDEN_SIZE..]
            .iter()
            .any(|value| value.to_bits() != 0.0f32.to_bits())
        {
            return Err(invalid(format!(
                "EAGLE sample {} final next-token embedding row must be positive zero",
                self.id
            )));
        }
        if self.teacher_draft_logits[(rows - 1) * DRAFT_VOCAB_SIZE..]
            .iter()
            .any(|value| value.to_bits() != 0.0f32.to_bits())
            || !self.teacher_logsumexp[rows - 1].is_nan()
        {
            return Err(invalid(format!(
                "EAGLE sample {} final shifted-teacher row must be zero logits with a NaN logsumexp sentinel",
                self.id
            )));
        }
        if self.teacher_draft_logits[..(rows - 1) * DRAFT_VOCAB_SIZE]
            .iter()
            .any(|value| !value.is_finite())
            || self.teacher_logsumexp[..rows - 1]
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(invalid(format!(
                "EAGLE sample {} contains a non-finite non-sentinel teacher value",
                self.id
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Eagle3ArrayRecord {
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Eagle3SampleMetadata {
    pub schema: String,
    pub positional_contract: String,
    pub id: String,
    pub length: usize,
    pub bootstrap_rows_masked: usize,
    pub draft_vocab_size: usize,
    pub draft_mapping_sha256: String,
    pub arrays: Vec<Eagle3ArrayRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Eagle3ManifestSample {
    pub id: String,
    pub path: String,
    pub length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Eagle3FeatureManifest {
    pub schema: String,
    pub positional_contract: String,
    pub endianness: String,
    pub target_model: String,
    pub target_model_sha256: String,
    pub target_quantization: String,
    pub eagle3_checkpoint_sha256: String,
    pub hidden_size: usize,
    pub layer_input_ids: [usize; 3],
    pub auxiliary_width: usize,
    pub target_vocab_size: usize,
    pub draft_vocab_size: usize,
    pub draft_mapping_sha256: String,
    pub draft_mapping: Eagle3ArrayRecord,
    pub samples: Vec<Eagle3ManifestSample>,
}

#[derive(Debug)]
pub struct Eagle3FeatureDatasetWriter {
    root: PathBuf,
    manifest: Eagle3FeatureManifest,
}

impl Eagle3FeatureDatasetWriter {
    pub fn create(
        root: impl Into<PathBuf>,
        target_model: String,
        target_model_sha256: String,
        target_quantization: String,
        eagle3_checkpoint_sha256: String,
        draft_to_target: &[u32],
    ) -> Result<Self> {
        if draft_to_target.len() != DRAFT_VOCAB_SIZE
            || draft_to_target
                .iter()
                .any(|&token| token as usize >= TARGET_VOCAB_SIZE)
            || draft_to_target.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(format!(
                "EAGLE draft mapping must contain {DRAFT_VOCAB_SIZE} strictly increasing target ids in 0..{TARGET_VOCAB_SIZE}"
            )));
        }
        let root = root.into();
        fs::create_dir(&root).map_err(|error| {
            invalid(format!(
                "creating EAGLE feature directory {}: {error}",
                root.display()
            ))
        })?;
        fs::create_dir(root.join("samples")).map_err(|error| {
            invalid(format!(
                "creating EAGLE sample directory {}: {error}",
                root.join("samples").display()
            ))
        })?;
        let draft_mapping = write_u32_array(
            &root,
            "draft_to_target.u32le",
            &[DRAFT_VOCAB_SIZE],
            draft_to_target,
        )?;
        Ok(Self {
            root,
            manifest: Eagle3FeatureManifest {
                schema: SCHEMA_ID.to_string(),
                positional_contract: POSITIONAL_CONTRACT.to_string(),
                endianness: "little".to_string(),
                target_model,
                target_model_sha256,
                target_quantization,
                eagle3_checkpoint_sha256,
                hidden_size: HIDDEN_SIZE,
                layer_input_ids: TARGET_LAYER_INPUT_IDS,
                auxiliary_width: AUX_WIDTH,
                target_vocab_size: TARGET_VOCAB_SIZE,
                draft_vocab_size: DRAFT_VOCAB_SIZE,
                draft_mapping_sha256: draft_mapping.sha256.clone(),
                draft_mapping,
                samples: Vec::new(),
            },
        })
    }

    pub fn push(&mut self, sample: &Eagle3FeatureSample) -> Result<()> {
        sample.validate()?;
        let ordinal = self.manifest.samples.len();
        let relative = format!("samples/{ordinal:08}");
        let sample_dir = self.root.join(&relative);
        fs::create_dir(&sample_dir).map_err(|error| {
            invalid(format!(
                "creating EAGLE sample directory {}: {error}",
                sample_dir.display()
            ))
        })?;
        let rows = sample.input_ids.len();
        let mut arrays = Vec::with_capacity(10);
        arrays.push(write_u32_array(
            &sample_dir,
            "input_ids.u32le",
            &[rows],
            &sample.input_ids,
        )?);
        arrays.push(write_u32_array(
            &sample_dir,
            "labels.u32le",
            &[rows],
            &sample.labels,
        )?);
        arrays.push(write_u32_array(
            &sample_dir,
            "target_argmax.u32le",
            &[rows],
            &sample.target_argmax,
        )?);
        arrays.push(write_u8_array(
            &sample_dir,
            "loss_mask.u8",
            &[rows],
            &sample.loss_mask,
        )?);
        arrays.push(write_bf16_array(
            &sample_dir,
            "aux_layer_inputs.bf16le",
            &[rows, AUX_WIDTH],
            &sample.aux_layer_inputs,
        )?);
        arrays.push(write_bf16_array(
            &sample_dir,
            "hidden_state.bf16le",
            &[rows, HIDDEN_SIZE],
            &sample.hidden_state,
        )?);
        arrays.push(write_bf16_array(
            &sample_dir,
            "input_embedding.bf16le",
            &[rows, HIDDEN_SIZE],
            &sample.input_embedding,
        )?);
        arrays.push(write_bf16_array(
            &sample_dir,
            "next_token_embedding.bf16le",
            &[rows, HIDDEN_SIZE],
            &sample.next_token_embedding,
        )?);
        arrays.push(write_bf16_array(
            &sample_dir,
            "teacher_draft_logits.bf16le",
            &[rows, DRAFT_VOCAB_SIZE],
            &sample.teacher_draft_logits,
        )?);
        arrays.push(write_f32_array(
            &sample_dir,
            "teacher_logsumexp.f32le",
            &[rows],
            &sample.teacher_logsumexp,
        )?);
        let metadata = Eagle3SampleMetadata {
            schema: SCHEMA_ID.to_string(),
            positional_contract: POSITIONAL_CONTRACT.to_string(),
            id: sample.id.clone(),
            length: rows,
            bootstrap_rows_masked: sample.bootstrap_rows_masked,
            draft_vocab_size: DRAFT_VOCAB_SIZE,
            draft_mapping_sha256: self.manifest.draft_mapping_sha256.clone(),
            arrays,
        };
        write_json_atomic(&sample_dir.join("meta.json"), &metadata)?;
        self.manifest.samples.push(Eagle3ManifestSample {
            id: sample.id.clone(),
            path: relative,
            length: rows,
        });
        // Publish only after every payload and its per-sample metadata are complete. A trainer
        // may tail manifest.json while this process continues exporting later records.
        write_json_atomic(&self.root.join("manifest.json"), &self.manifest)?;
        Ok(())
    }

    pub fn finish(self) -> Result<Eagle3FeatureManifest> {
        write_json_atomic(&self.root.join("manifest.json"), &self.manifest)?;
        Ok(self.manifest)
    }
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    let file = File::create(&temporary)
        .map_err(|error| invalid(format!("creating {}: {error}", temporary.display())))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|error| invalid(format!("serializing {}: {error}", temporary.display())))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .map_err(|error| invalid(format!("writing {}: {error}", temporary.display())))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|error| invalid(format!("syncing {}: {error}", temporary.display())))?;
    fs::rename(&temporary, path).map_err(|error| {
        invalid(format!(
            "publishing {} as {}: {error}",
            temporary.display(),
            path.display()
        ))
    })
}

fn hash_record(path: &Path, dtype: &str, shape: &[usize]) -> Result<Eagle3ArrayRecord> {
    let sha256 = crate::receipt::sha256_file_hex(path).map_err(|error| {
        invalid(format!(
            "hashing EAGLE feature payload {}: {error}",
            path.display()
        ))
    })?;
    Ok(Eagle3ArrayRecord {
        file: path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid(format!("non-UTF-8 EAGLE payload path {}", path.display())))?
            .to_string(),
        dtype: dtype.to_string(),
        shape: shape.to_vec(),
        sha256,
    })
}

fn write_u32_array(
    root: &Path,
    file: &str,
    shape: &[usize],
    values: &[u32],
) -> Result<Eagle3ArrayRecord> {
    let path = root.join(file);
    let mut writer = BufWriter::new(
        File::create(&path)
            .map_err(|error| invalid(format!("creating {}: {error}", path.display())))?,
    );
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| invalid(format!("writing {}: {error}", path.display())))?;
    }
    writer
        .flush()
        .map_err(|error| invalid(format!("writing {}: {error}", path.display())))?;
    hash_record(&path, "uint32", shape)
}

fn write_u8_array(
    root: &Path,
    file: &str,
    shape: &[usize],
    values: &[u8],
) -> Result<Eagle3ArrayRecord> {
    let path = root.join(file);
    fs::write(&path, values)
        .map_err(|error| invalid(format!("writing {}: {error}", path.display())))?;
    hash_record(&path, "uint8", shape)
}

fn write_f32_array(
    root: &Path,
    file: &str,
    shape: &[usize],
    values: &[f32],
) -> Result<Eagle3ArrayRecord> {
    let path = root.join(file);
    let mut writer = BufWriter::new(
        File::create(&path)
            .map_err(|error| invalid(format!("creating {}: {error}", path.display())))?,
    );
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| invalid(format!("writing {}: {error}", path.display())))?;
    }
    writer
        .flush()
        .map_err(|error| invalid(format!("writing {}: {error}", path.display())))?;
    hash_record(&path, "float32", shape)
}

fn write_bf16_array(
    root: &Path,
    file: &str,
    shape: &[usize],
    values: &[f32],
) -> Result<Eagle3ArrayRecord> {
    let path = root.join(file);
    let mut writer = BufWriter::new(
        File::create(&path)
            .map_err(|error| invalid(format!("creating {}: {error}", path.display())))?,
    );
    for value in values {
        writer
            .write_all(&f32_to_bf16_rne(*value).to_le_bytes())
            .map_err(|error| invalid(format!("writing {}: {error}", path.display())))?;
    }
    writer
        .flush()
        .map_err(|error| invalid(format!("writing {}: {error}", path.display())))?;
    hash_record(&path, "bfloat16", shape)
}

/// IEEE-754 round-to-nearest-even conversion used by PyTorch/MLX BF16 casts.
pub fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let round_bias = 0x7fff_u32 + ((bits >> 16) & 1);
    bits.wrapping_add(round_bias).wrapping_shr(16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_sample() -> Eagle3FeatureSample {
        let rows = 4;
        let mut input_embedding = Vec::with_capacity(rows * HIDDEN_SIZE);
        for row in 0..rows {
            input_embedding.extend(std::iter::repeat_n(row as f32 + 1.0, HIDDEN_SIZE));
        }
        let mut next_token_embedding = input_embedding[HIDDEN_SIZE..].to_vec();
        next_token_embedding.resize(rows * HIDDEN_SIZE, 0.0);
        let mut teacher_draft_logits = vec![0.25; rows * DRAFT_VOCAB_SIZE];
        teacher_draft_logits[(rows - 1) * DRAFT_VOCAB_SIZE..].fill(0.0);
        Eagle3FeatureSample {
            id: "fixture".to_string(),
            input_ids: vec![1, 2, 3, 4],
            labels: vec![3, 4, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
            target_argmax: vec![5, 6, 7, INVALID_TOKEN_ID],
            loss_mask: vec![0, 0, 0, 0],
            aux_layer_inputs: vec![0.5; rows * AUX_WIDTH],
            hidden_state: vec![1.0; rows * HIDDEN_SIZE],
            input_embedding,
            next_token_embedding,
            teacher_draft_logits,
            teacher_logsumexp: vec![1.0, 2.0, 3.0, f32::NAN],
            bootstrap_rows_masked: 0,
        }
    }

    #[test]
    fn bf16_uses_round_to_nearest_even() {
        assert_eq!(f32_to_bf16_rne(1.0), 0x3f80);
        assert_eq!(f32_to_bf16_rne(-2.0), 0xc000);
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn sample_validation_pins_shapes_and_masked_bootstrap() {
        let mut sample = fixture_sample();
        sample.validate().unwrap();
        sample.aux_layer_inputs.pop();
        assert!(sample.validate().is_err());
        sample = fixture_sample();
        sample.bootstrap_rows_masked = 1;
        assert!(sample.validate().is_err());
        sample = fixture_sample();
        sample.teacher_draft_logits[0] = f32::NAN;
        assert!(sample.validate().is_err());
    }

    #[test]
    fn positional_contract_shifts_teacher_by_one_capture_row() {
        let capture_argmax = vec![10, 11, 12, 13];
        let mut capture_logits = Vec::new();
        for row in 0..4 {
            capture_logits.extend(std::iter::repeat_n(row as f32, DRAFT_VOCAB_SIZE));
        }
        let shifted = shift_teacher_rows(
            &capture_argmax,
            &capture_logits,
            &[100.0, 101.0, 102.0, 103.0],
        )
        .unwrap();
        assert_eq!(shifted.target_argmax, [11, 12, 13, INVALID_TOKEN_ID]);
        assert_eq!(
            shifted
                .teacher_draft_logits
                .chunks_exact(DRAFT_VOCAB_SIZE)
                .map(|row| row[0])
                .collect::<Vec<_>>(),
            [1.0, 2.0, 3.0, 0.0]
        );
        assert_eq!(&shifted.teacher_logsumexp[..3], &[101.0, 102.0, 103.0]);
        assert!(shifted.teacher_logsumexp[3].is_nan());
        assert_eq!(
            shifted_corpus_labels(&[1, 2, 3, 4]).unwrap(),
            [3, 4, INVALID_TOKEN_ID, INVALID_TOKEN_ID]
        );
    }

    #[test]
    fn loss_mask_shifts_token_positions_to_teacher_rows_and_keeps_two_sentinels() {
        assert_eq!(shift_loss_mask(&[0, 0, 1, 1]).unwrap(), [1, 1, 0, 0]);

        let raw_final_token_may_be_trainable = Eagle3FeatureInput {
            id: Some("assistant-through-final-token".to_string()),
            input_ids: vec![1, 2, 3],
            loss_mask: vec![0, 1, 1],
        };
        raw_final_token_may_be_trainable.validate().unwrap();
        assert_eq!(shift_loss_mask(&[0, 1, 1]).unwrap(), [1, 0, 0]);
    }

    #[test]
    fn writer_emits_manifest_and_hashed_raw_arrays() {
        let root = std::env::temp_dir().join(format!(
            "camelid-eagle-export-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        let mut writer = Eagle3FeatureDatasetWriter::create(
            &root,
            "target.gguf".to_string(),
            "00".repeat(32),
            "Q4_K_M".to_string(),
            "11".repeat(32),
            &(0..DRAFT_VOCAB_SIZE as u32).collect::<Vec<_>>(),
        )
        .unwrap();
        writer.push(&fixture_sample()).unwrap();
        let manifest = writer.finish().unwrap();
        assert_eq!(manifest.schema, SCHEMA_ID);
        assert_eq!(manifest.positional_contract, POSITIONAL_CONTRACT);
        assert_eq!(manifest.layer_input_ids, [2, 14, 25]);
        assert_eq!(manifest.draft_vocab_size, DRAFT_VOCAB_SIZE);
        assert_eq!(manifest.samples[0].length, 4);
        let metadata: Eagle3SampleMetadata = serde_json::from_slice(
            &std::fs::read(root.join("samples/00000000/meta.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata.arrays.len(), 10);
        assert_eq!(metadata.positional_contract, POSITIONAL_CONTRACT);
        assert_eq!(metadata.draft_mapping_sha256, manifest.draft_mapping_sha256);
        let aux = metadata
            .arrays
            .iter()
            .find(|array| array.file == "aux_layer_inputs.bf16le")
            .unwrap();
        assert_eq!(aux.shape, [4, AUX_WIDTH]);
        assert_eq!(
            std::fs::metadata(root.join("samples/00000000/aux_layer_inputs.bf16le"))
                .unwrap()
                .len(),
            (4 * AUX_WIDTH * 2) as u64
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
