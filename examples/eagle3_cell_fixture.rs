//! Emit a synthetic, target-free Camelid Metal EAGLE-3 cell parity fixture.
//!
//! The fixture exercises an authoritative causal prefix followed by recurrent
//! TTT-style cells at the final prefix row. It never loads a target GGUF or any
//! prompt. `tools/eagle3_mlx/cell_parity.py` consumes the resulting directory.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use camelid::eagle3::{Eagle3DraftModel, HIDDEN_SIZE};
use camelid::metal::{
    Eagle3MetalOutput, Eagle3MetalState, Eagle3MetalWeights, EAGLE3_AUX_WIDTH, EAGLE3_DRAFT_VOCAB,
    EAGLE3_TOP_K_CANDIDATES,
};
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};

const SCHEMA: &str = "camelid-eagle3-cell-parity-v1";
const GENERATOR: &str = "splitmix64-f32-v1";

#[derive(Debug, Parser)]
#[command(about = "Write a strict MLX-vs-Camelid EAGLE cell parity fixture")]
struct Args {
    /// Exact 15-tensor EAGLE-3 checkpoint directory.
    #[arg(long)]
    eagle3: PathBuf,

    /// New output directory. Existing paths are refused.
    #[arg(long)]
    output: PathBuf,

    /// Number of authoritative base rows.
    #[arg(long, default_value_t = 4)]
    rows: usize,

    /// Total TTT depths, including the authoritative depth-zero pass.
    #[arg(long, default_value_t = 3)]
    depths: usize,

    /// Deterministic synthetic-input seed recorded in the manifest.
    #[arg(long, default_value_t = 0xC4_3E_11_D3)]
    seed: u64,
}

#[derive(Debug, Serialize)]
struct ArrayRecord {
    file: String,
    dtype: String,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct Manifest {
    schema: &'static str,
    generator: &'static str,
    seed: u64,
    rows: usize,
    depths: usize,
    hidden_size: usize,
    auxiliary_width: usize,
    draft_vocab_size: usize,
    top_k: usize,
    rope_theta: f32,
    sliding_window: Option<usize>,
    checkpoint_weights_sha256: String,
    checkpoint_config_sha256: String,
    fixture_binary_sha256: String,
    camelid_version: &'static str,
    camelid_lane: &'static str,
    arrays: Vec<ArrayRecord>,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut stream =
        fs::File::open(path).with_context(|| format!("opening {} for SHA-256", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = stream
            .read(&mut buffer)
            .with_context(|| format!("hashing {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_bytes(
    root: &Path,
    file: &str,
    dtype: &str,
    shape: Vec<usize>,
    bytes: &[u8],
) -> Result<ArrayRecord> {
    let path = root.join(file);
    let mut stream = fs::File::create(&path)
        .with_context(|| format!("creating parity payload {}", path.display()))?;
    stream
        .write_all(bytes)
        .with_context(|| format!("writing parity payload {}", path.display()))?;
    stream
        .sync_all()
        .with_context(|| format!("syncing parity payload {}", path.display()))?;
    Ok(ArrayRecord {
        file: file.to_string(),
        dtype: dtype.to_string(),
        shape,
        sha256: sha256_bytes(bytes),
    })
}

fn write_f32(root: &Path, file: &str, shape: Vec<usize>, values: &[f32]) -> Result<ArrayRecord> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_bytes(root, file, "float32", shape, &bytes)
}

fn write_u32(root: &Path, file: &str, shape: Vec<usize>, values: &[u32]) -> Result<ArrayRecord> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_bytes(root, file, "uint32", shape, &bytes)
}

fn splitmix_value(seed: u64, index: usize, scale: f32) -> f32 {
    let mut value = seed.wrapping_add((index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    let unit = ((value >> 40) as u32) as f32 * (1.0 / 16_777_216.0);
    (unit * 2.0 - 1.0) * scale
}

fn require_dense_runtime_environment() -> Result<()> {
    for name in [
        "CAMELID_EAGLE3_LM_HEAD_ROWS",
        "CAMELID_EAGLE3_LM_HEAD_Q8",
        "CAMELID_EAGLE3_BODY_Q8",
    ] {
        if std::env::var_os(name).is_some() {
            bail!("strict cell parity requires {name} to be unset");
        }
    }
    Ok(())
}

fn metal_state(model: &Eagle3DraftModel, max_positions: usize) -> Result<Eagle3MetalState> {
    let matrices = &model.matrices;
    let norms = &model.norms;
    Eagle3MetalState::new(
        Eagle3MetalWeights {
            fc_bf16: &matrices.feature_fusion.bytes,
            q_proj_bf16: &matrices.attention_q.bytes,
            k_proj_bf16: &matrices.attention_k.bytes,
            v_proj_bf16: &matrices.attention_v.bytes,
            o_proj_bf16: &matrices.attention_o.bytes,
            gate_proj_bf16: &matrices.mlp_gate.bytes,
            up_proj_bf16: &matrices.mlp_up.bytes,
            down_proj_bf16: &matrices.mlp_down.bytes,
            lm_head_bf16: &matrices.lm_head.bytes,
            input_layernorm: &norms.input,
            hidden_norm: &norms.hidden,
            post_attention_layernorm: &norms.post_attention,
            output_norm: &norms.output,
            d2t_offsets: &model.d2t_offsets,
            rope_theta: model.config.rope_theta,
            sliding_window: model.config.sliding_window,
        },
        max_positions,
    )
    .map_err(anyhow::Error::msg)
}

fn record_ranking(
    output: &Eagle3MetalOutput,
    selected: &mut Vec<u32>,
    draft_ids: &mut Vec<u32>,
    target_ids: &mut Vec<u32>,
    logits: &mut Vec<f32>,
) -> Result<()> {
    if output.evaluated_vocab_rows != EAGLE3_DRAFT_VOCAB
        || output.top_candidates.len() != EAGLE3_TOP_K_CANDIDATES
    {
        bail!(
            "strict fixture requires all {EAGLE3_DRAFT_VOCAB} logits and exactly \
             {EAGLE3_TOP_K_CANDIDATES} retained candidates, got {}/{}",
            output.evaluated_vocab_rows,
            output.top_candidates.len()
        );
    }
    selected.push(output.draft_token);
    for candidate in &output.top_candidates {
        draft_ids.push(candidate.draft_token);
        target_ids.push(candidate.target_token);
        logits.push(candidate.logit);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.rows < 2 || !(2..=7).contains(&args.depths) {
        bail!("parity fixture requires rows>=2 and depths in 2..=7");
    }
    require_dense_runtime_environment()?;
    if args.output.exists() {
        bail!("refusing existing parity output {}", args.output.display());
    }
    let model = Eagle3DraftModel::load(&args.eagle3)?;
    let logical_positions = args.rows + args.depths - 1;
    if model
        .config
        .sliding_window
        .is_some_and(|window| logical_positions > window)
    {
        bail!(
            "fixture logical span {logical_positions} exceeds checkpoint window {:?}",
            model.config.sliding_window
        );
    }

    let aux: Vec<f32> = (0..args.rows * EAGLE3_AUX_WIDTH)
        .map(|index| splitmix_value(args.seed ^ 0xA0A0_A0A0, index, 0.25))
        .collect();
    let embeddings: Vec<f32> = (0..args.depths * args.rows * HIDDEN_SIZE)
        .map(|index| splitmix_value(args.seed ^ 0xE1E1_E1E1, index, 0.25))
        .collect();

    let mut head = metal_state(&model, logical_positions)?;
    let fused = head.fuse_features(&aux).map_err(anyhow::Error::msg)?;
    let depth0_embeddings = &embeddings[..args.rows * HIDDEN_SIZE];
    let depth0 = head
        .forward_batch(depth0_embeddings, &fused, 0)
        .map_err(anyhow::Error::msg)?;
    let mut depth0_hidden = Vec::with_capacity(args.rows * HIDDEN_SIZE);
    for output in &depth0 {
        depth0_hidden.extend_from_slice(&output.raw_hidden);
    }

    let mut selected = Vec::with_capacity(args.depths);
    let mut top_draft_ids = Vec::with_capacity(args.depths * EAGLE3_TOP_K_CANDIDATES);
    let mut top_target_ids = Vec::with_capacity(args.depths * EAGLE3_TOP_K_CANDIDATES);
    let mut top_logits = Vec::with_capacity(args.depths * EAGLE3_TOP_K_CANDIDATES);
    let mut previous = depth0
        .last()
        .context("authoritative fixture produced no output")?
        .raw_hidden
        .clone();
    record_ranking(
        depth0.last().expect("checked non-empty"),
        &mut selected,
        &mut top_draft_ids,
        &mut top_target_ids,
        &mut top_logits,
    )?;

    let mut recurrent_hidden = Vec::with_capacity((args.depths - 1) * HIDDEN_SIZE);
    for depth in 1..args.depths {
        let row = depth * args.rows + (args.rows - 1);
        let start = row * HIDDEN_SIZE;
        let output = head
            .forward_token(
                &embeddings[start..start + HIDDEN_SIZE],
                &previous,
                args.rows - 1 + depth,
            )
            .map_err(anyhow::Error::msg)?;
        recurrent_hidden.extend_from_slice(&output.raw_hidden);
        record_ranking(
            &output,
            &mut selected,
            &mut top_draft_ids,
            &mut top_target_ids,
            &mut top_logits,
        )?;
        previous = output.raw_hidden;
    }

    let checkpoint_weights_sha256 = sha256_file(&args.eagle3.join("model.safetensors"))?;
    let checkpoint_config_sha256 = sha256_file(&args.eagle3.join("config.json"))?;
    let fixture_binary_sha256 =
        sha256_file(&std::env::current_exe().context("resolving fixture executable")?)?;

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating parity parent {}", parent.display()))?;
    }
    fs::create_dir(&args.output)
        .with_context(|| format!("creating parity output {}", args.output.display()))?;

    let mut arrays = vec![
        write_f32(
            &args.output,
            "input_aux.f32le",
            vec![args.rows, EAGLE3_AUX_WIDTH],
            &aux,
        )?,
        write_f32(
            &args.output,
            "input_embeddings.f32le",
            vec![args.depths, args.rows, HIDDEN_SIZE],
            &embeddings,
        )?,
        write_f32(
            &args.output,
            "camelid_fused.f32le",
            vec![args.rows, HIDDEN_SIZE],
            &fused,
        )?,
        write_f32(
            &args.output,
            "camelid_depth0_hidden.f32le",
            vec![args.rows, HIDDEN_SIZE],
            &depth0_hidden,
        )?,
        write_f32(
            &args.output,
            "camelid_recurrent_last_hidden.f32le",
            vec![args.depths - 1, HIDDEN_SIZE],
            &recurrent_hidden,
        )?,
        write_u32(
            &args.output,
            "camelid_selected_draft_ids.u32le",
            vec![args.depths],
            &selected,
        )?,
        write_u32(
            &args.output,
            "camelid_top_draft_ids.u32le",
            vec![args.depths, EAGLE3_TOP_K_CANDIDATES],
            &top_draft_ids,
        )?,
        write_u32(
            &args.output,
            "camelid_top_target_ids.u32le",
            vec![args.depths, EAGLE3_TOP_K_CANDIDATES],
            &top_target_ids,
        )?,
    ];
    arrays.push(write_f32(
        &args.output,
        "camelid_top_logits.f32le",
        vec![args.depths, EAGLE3_TOP_K_CANDIDATES],
        &top_logits,
    )?);

    let manifest = Manifest {
        schema: SCHEMA,
        generator: GENERATOR,
        seed: args.seed,
        rows: args.rows,
        depths: args.depths,
        hidden_size: HIDDEN_SIZE,
        auxiliary_width: EAGLE3_AUX_WIDTH,
        draft_vocab_size: EAGLE3_DRAFT_VOCAB,
        top_k: EAGLE3_TOP_K_CANDIDATES,
        rope_theta: model.config.rope_theta,
        sliding_window: model.config.sliding_window,
        checkpoint_weights_sha256,
        checkpoint_config_sha256,
        fixture_binary_sha256,
        camelid_version: env!("CARGO_PKG_VERSION"),
        camelid_lane: "dense-bf16-weights-f32-activations-f16-kv",
        arrays,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    fs::write(
        args.output.join("manifest.json"),
        [&manifest_bytes[..], b"\n"].concat(),
    )?;
    println!(
        "wrote {} rows x {} depths synthetic EAGLE cell fixture to {}",
        args.rows,
        args.depths,
        args.output.display()
    );
    Ok(())
}
