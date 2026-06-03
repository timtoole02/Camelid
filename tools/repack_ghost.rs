//! repack-ghost: restructure a GGUF model into a `.cghost` layer-streaming container.
//!
//! Ghost (layer-streaming) mode executes a model one transformer block at a time, holding
//! only a tiny working window in RAM. GGUF scatters a block's tensors across the file; this
//! tool writes every block's tensors contiguously (one sequential read per block) at the
//! SOURCE quantization — v1 is a pure re-layout, so the streamed path can be parity-gated
//! byte-for-byte against the resident path.

use std::path::PathBuf;

use clap::Parser;

use camelid::gguf::read_metadata;
use camelid::ghost::write_cghost;
use camelid::model::{LlamaModelConfig, LlamaTensorBinding};
use camelid::tensor::TensorStore;

#[derive(Parser)]
#[command(
    name = "repack-ghost",
    about = "Repack a GGUF into a layer-contiguous .cghost streaming container"
)]
struct Args {
    /// Source GGUF model
    model: PathBuf,
    /// Output path (default: <model>.cghost)
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let out = args.out.unwrap_or_else(|| {
        let mut p = args.model.clone();
        p.set_extension("cghost");
        p
    });

    println!(
        "[repack-ghost] reading GGUF metadata from {:?}...",
        args.model
    );
    let gguf = read_metadata(&args.model)?;
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&args.model, &gguf);
    let source = args
        .model
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    println!(
        "[repack-ghost] repacking {} transformer blocks -> {:?}",
        binding.layers.len(),
        out
    );
    let index = write_cghost(&store, &binding, &source, &out)?;

    let total: u64 = index.groups.iter().map(|g| g.span().1).sum();
    let max_layer = index
        .groups
        .iter()
        .filter(|g| g.id.starts_with("blk."))
        .map(|g| g.span().1)
        .max()
        .unwrap_or(0);
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!(
        "[repack-ghost] done: {} groups, {:.2} GiB payload, largest block group {:.1} MiB \
         (the streaming window per layer), tied_output={}",
        index.groups.len(),
        gib(total),
        mib(max_layer),
        index.tied_output
    );
    Ok(())
}
