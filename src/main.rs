use std::{
    io::Write,
    net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_set_qos_class_self_np(
        qos_class: u32,
        relative_priority: std::os::raw::c_int,
    ) -> std::os::raw::c_int;
}

use camelid::{
    api,
    cluster::{
        recv_activation_packet, recv_token_feedback, send_activation_packet, send_token_feedback,
    },
    gguf::{read_metadata, GgufTensorType},
    ghost::{GhostFile, GhostPrefetcher},
    inference::{
        LlamaInferenceSession, LlamaLayerWeights, LlamaLoadedWeights, LlamaSampler,
        Q8ResidencyReport, SamplingConfig,
    },
    metal::detect_metal_device,
    model::{LlamaModelConfig, LlamaTensorBinding},
    tensor::{CpuTensor, Q8_0TensorBlocks, TensorStore},
    tokenizer::Tokenizer,
};
use clap::{Parser, Subcommand};
use rayon::ThreadPoolBuilder;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(name = "camelid", about = "Rust-native local GGUF inference backend")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the local HTTP API server.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8181", env = "CAMELID_ADDR")]
        addr: SocketAddr,
        /// Load a GGUF model at startup and auto-select the safest validated execution plan.
        #[arg(long, env = "CAMELID_MODEL")]
        model: Option<PathBuf>,
        /// Override Rayon worker threads for the inference server.
        #[arg(long, env = "CAMELID_THREADS")]
        threads: Option<usize>,
        /// Override the linear-output parallelization threshold used by hot-path CPU kernels.
        #[arg(long, env = "CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS")]
        parallel_linear_min_outputs: Option<usize>,
        /// Override the minimum matrix size before macOS Accelerate BLAS is used.
        ///
        /// On macOS, Camelid defaults to using Accelerate only for larger dense linear rows.
        #[arg(long, env = "CAMELID_APPLE_ACCELERATE_MIN_ELEMENTS")]
        apple_accelerate_min_elements: Option<usize>,
        /// Enable the experimental Metal dense linear-row path on macOS.
        #[arg(long, env = "CAMELID_METAL_LINEAR", default_value_t = false)]
        metal_linear: bool,
        /// Enable the experimental Metal Q8_0 encoded row-dot path on macOS.
        #[arg(long, env = "CAMELID_METAL_Q8", default_value_t = false)]
        metal_q8: bool,
        /// Log the current acceleration/runtime discovery state at startup.
        #[arg(long, default_value_t = true)]
        log_acceleration: bool,
    },
    /// Start the distributed HTTP API server or TCP Worker.
    ServeDistributed {
        /// Mode to run: coordinator or worker
        #[arg(long, default_value = "coordinator")]
        role: String,
        /// Address to listen on (worker TCP listener or coordinator HTTP server)
        #[arg(long, default_value = "127.0.0.1:8181")]
        addr: SocketAddr,
        /// Address of the worker TCP listener (required for coordinator)
        #[arg(long)]
        worker_addr: Option<String>,
        /// Partition range of layers to evaluate on this node (e.g. 0..16 or 16..32)
        #[arg(long)]
        layer_range: String,
        /// Load a GGUF model at startup
        #[arg(long, env = "CAMELID_MODEL")]
        model: PathBuf,
        /// Override Rayon worker threads
        #[arg(long, env = "CAMELID_THREADS")]
        threads: Option<usize>,
    },
    /// Benchmark raw TCP latency and bandwidth between Coordinator and Worker.
    BenchNetwork {
        /// Mode to run: coordinator or worker
        #[arg(long, default_value = "coordinator")]
        role: String,
        /// Address to bind to or connect to
        #[arg(long, default_value = "127.0.0.1:8182")]
        addr: String,
        /// Number of round-trips to perform for latency test
        #[arg(long, default_value_t = 1000)]
        ping_count: usize,
        /// Payload size in bytes for the latency test (default: 16KB hidden state size)
        #[arg(long, default_value_t = 16384)]
        payload_size: usize,
        /// Amount of megabytes to stream for throughput testing (default: 100 MB)
        #[arg(long, default_value_t = 100)]
        bandwidth_mb: usize,
    },
    /// Inspect GGUF metadata and tensor descriptors.
    Inspect { path: PathBuf },
    /// Dump focused tensor descriptor, raw block, and f32 dequantization diagnostics.
    TensorDump {
        path: PathBuf,
        /// Tensor name to dump. Repeat to override the TinyLlama parity default set.
        #[arg(long = "tensor")]
        tensors: Vec<String>,
        /// Number of decoded f32 values to include from tensor start and max-absolute window.
        #[arg(long, default_value_t = 8)]
        window: usize,
        /// Row index to sample for each 2D tensor using the dump's runtime shape.
        #[arg(long = "row")]
        rows: Vec<usize>,
        /// Token id to sample as a logical token-major row for embedding-shaped tensors.
        #[arg(long = "token")]
        tokens: Vec<usize>,
        /// LLaMA layer index whose Q/K/V/O and FFN tensors should be included in the dump.
        #[arg(long = "layer")]
        layers: Vec<usize>,
    },
    /// Run a deterministic release-mode microbenchmark for dense matmul/FFN hot loops.
    BenchDenseHotloops {
        /// LLaMA hidden width for the synthetic single-row input.
        #[arg(long, default_value_t = 2048)]
        hidden: usize,
        /// LLaMA feed-forward width for synthetic gate/up/down projections.
        #[arg(long, default_value_t = 5632)]
        ffn: usize,
        /// Measured iterations after warmup.
        #[arg(long, default_value_t = 20)]
        repeats: usize,
        /// Unreported warmup iterations.
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Override Rayon worker threads for this benchmark. Defaults to RAYON_NUM_THREADS/Rayon.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Load one GGUF Q8_0 tensor as retained blocks and benchmark bounded row dequantization/dot rows.
    BenchQ8Blocks {
        /// GGUF model path.
        path: PathBuf,
        /// Q8_0 tensor name to load as block-only data.
        #[arg(long, default_value = "blk.0.ffn_gate.weight")]
        tensor: String,
        /// Reinterpret a rank-2 tensor by swapping its logical rows/cols before benchmarking.
        ///
        /// This mirrors Camelid's guarded rectangular linear/output-projection layout path for
        /// tensors whose GGUF descriptor dimensions are stored token/input-major but the lazy
        /// Q8 hot path consumes contiguous logical output rows.
        #[arg(long)]
        swap_rank2_shape: bool,
        /// Row index to dequantize. Repeat for multiple rows.
        #[arg(long = "row")]
        rows: Vec<usize>,
        /// Measured iterations after warmup.
        #[arg(long, default_value_t = 20)]
        repeats: usize,
        /// Unreported warmup iterations.
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Also benchmark the lazy all-row Q8_0 dot helper that returns a dense f32 output vector.
        #[arg(long)]
        all_rows_dot: bool,
        /// Also benchmark the rank-2 single-input-row Q8_0 lazy-linear adapter shape.
        #[arg(long)]
        single_input_row_dot: bool,
    },
    /// Start a distributed pipeline worker node.
    DistributeWorker {
        /// GGUF model path.
        path: PathBuf,
        /// Listen address for incoming master/worker connection.
        #[arg(long, default_value = "0.0.0.0:5005")]
        addr: SocketAddr,
        /// Target forward address (next node in the pipeline).
        #[arg(long)]
        forward_addr: Option<SocketAddr>,
        /// Range of layers to own and execute, e.g., "16..32" or "24..56".
        #[arg(long)]
        layers: String,
        /// Master address to send token feedback to when we are the final node.
        #[arg(long)]
        master_addr: Option<SocketAddr>,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
        /// EXPERIMENTAL ghost mesh: stream this node's layer shard per token from a
        /// `.cghost` file (double-buffered) instead of holding it resident. Only the
        /// embedding/output ends stay in RAM; the shard's disk window overlaps the other
        /// node's compute.
        #[arg(long)]
        cghost: Option<PathBuf>,
    },
    /// Start a distributed pipeline master node.
    DistributeMaster {
        /// GGUF model path.
        path: PathBuf,
        /// Worker address to send activation streams to.
        #[arg(long)]
        worker_addr: SocketAddr,
        /// Range of layers to own and execute, e.g., "0..16" or "0..24".
        #[arg(long)]
        layers: String,
        /// Listen address for token feedback or final results from the last node in the pipeline.
        #[arg(long, default_value = "0.0.0.0:5006")]
        addr: SocketAddr,
        /// Prompt to execute.
        #[arg(long, default_value = "Write a quick Rust hello-world function:")]
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
        /// EXPERIMENTAL ghost mesh: stream this node's layer shard per token from a
        /// `.cghost` file (double-buffered) instead of holding it resident. Only the
        /// embedding/output ends stay in RAM; the shard's disk window overlaps the other
        /// node's compute.
        #[arg(long)]
        cghost: Option<PathBuf>,
    },
    /// Single-node generation microbenchmark. Loads a GGUF model once, generates
    /// from a prompt, and emits one JSON metrics object per measured iteration
    /// (load/prefill/TTFT/decode timings, decode tok/s, peak RSS). For runtime
    /// comparison harnesses.
    BenchGenerate {
        /// GGUF model path.
        model: PathBuf,
        /// Read the prompt from this UTF-8 file. Takes precedence over --prompt.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Inline prompt text (used when --prompt-file is absent).
        #[arg(long)]
        prompt: Option<String>,
        /// Maximum tokens to generate per iteration.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy/argmax, deterministic).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Number of measured iterations (one JSON object per iteration).
        #[arg(long, default_value_t = 1)]
        iterations: usize,
        /// Run one unmeasured warmup generation before the measured iterations.
        #[arg(long, default_value_t = false)]
        warmup: bool,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
        /// Accepted for compatibility; JSON is always emitted to stdout.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// EXPERIMENTAL ghost (layer-streaming) mode: execute a model one transformer block at
    /// a time, streaming each block's weights from a layer-contiguous `.cghost` file
    /// (see the `repack-ghost` tool) and holding only a one-layer working window plus the
    /// embedding/output ends in RAM. Trades throughput for a strict memory ceiling.
    /// Synchronous v1: each layer's read blocks the forward (no prefetch yet).
    GhostRun {
        /// GGUF model path (metadata, tokenizer, and resident embedding/output ends).
        model: PathBuf,
        /// Layer-contiguous .cghost file produced by `repack-ghost` from the same model.
        #[arg(long)]
        cghost: PathBuf,
        /// Prompt to execute (greedy decode).
        #[arg(long, default_value = "Write a quick Rust hello-world function:")]
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
        /// Disable the double-buffered prefetch worker and read each layer synchronously
        /// on the critical path (the v1 behavior; useful for A/B comparison).
        #[arg(long, default_value_t = false)]
        sync_stream: bool,
        /// Strict memory ceiling mode: bypass the OS page cache for `.cghost` reads
        /// (F_NOCACHE) so streamed pages never accumulate. Leave off when the model fits
        /// in RAM — the cache is a free win there.
        #[arg(long, default_value_t = false)]
        evict_page_cache: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    match Cli::parse().command {
        Command::Serve {
            addr,
            model,
            threads,
            parallel_linear_min_outputs,
            apple_accelerate_min_elements,
            metal_linear,
            metal_q8,
            log_acceleration,
        } => {
            configure_rayon_threads(threads)?;
            apply_runtime_tuning_env(
                parallel_linear_min_outputs,
                apple_accelerate_min_elements,
                metal_linear,
                metal_q8,
            );
            if log_acceleration {
                log_acceleration_state();
            }
            #[cfg(target_os = "macos")]
            unsafe {
                pthread_set_qos_class_self_np(0x09, 0); // QOS_CLASS_BACKGROUND (forces network I/O onto E-cores)
            }
            api::serve(addr, threads, model).await?
        }
        Command::ServeDistributed {
            role,
            addr,
            worker_addr,
            layer_range,
            model,
            threads,
        } => {
            configure_rayon_threads(threads)?;

            let parts: Vec<&str> = layer_range.split("..").collect();
            anyhow::ensure!(
                parts.len() == 2,
                "Layer range must be in format START..END (e.g. 0..16)"
            );
            let layer_start = parts[0].parse::<usize>()?;
            let layer_end = parts[1].parse::<usize>()?;
            anyhow::ensure!(
                layer_start < layer_end,
                "layer_start must be less than layer_end"
            );

            let _ = camelid::distributed::DISTRIBUTED_RANGE.set((layer_start, layer_end));

            if role == "coordinator" {
                let worker_addr_str = worker_addr.ok_or_else(|| {
                    anyhow::anyhow!("--worker-addr is required in coordinator mode")
                })?;

                tracing::info!(worker_addr = %worker_addr_str, "Coordinator connecting to worker");
                let client = camelid::distributed::DistributedClient::connect(&worker_addr_str)?;
                camelid::distributed::DISTRIBUTED_CLIENT
                    .set(client)
                    .map_err(|_| anyhow::anyhow!("Failed to set global distributed client lock"))?;
                tracing::info!("Coordinator connected to worker successfully");

                #[cfg(target_os = "macos")]
                unsafe {
                    pthread_set_qos_class_self_np(0x09, 0); // QOS_CLASS_BACKGROUND (forces network I/O onto E-cores)
                }
                api::serve(addr, threads, Some(model)).await?
            } else if role == "worker" {
                let gguf = camelid::gguf::read_metadata(&model)?;
                let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
                let binding = camelid::model::LlamaTensorBinding::bind(&gguf, &config)?;
                let store = camelid::tensor::TensorStore::open(&model, &gguf);

                tracing::info!(
                    "Worker loading partitioned weights (layers {}..{})",
                    layer_start,
                    layer_end
                );
                let weights = camelid::inference::LlamaLoadedWeights::load_distributed(
                    &store,
                    &binding,
                    layer_start,
                    layer_end,
                    false,
                    false,
                )?;

                tracing::info!("Worker weights loaded successfully. Initializing session.");
                let session = camelid::inference::LlamaInferenceSession::new(config, weights)?;

                let addr_str = addr.to_string();
                #[cfg(target_os = "macos")]
                unsafe {
                    pthread_set_qos_class_self_np(0x09, 0); // QOS_CLASS_BACKGROUND (forces network I/O onto E-cores)
                }
                camelid::distributed::run_worker_loop(&addr_str, session)?;
            } else {
                anyhow::bail!("Invalid role: {role}. Must be 'coordinator' or 'worker'");
            }
        }
        Command::BenchNetwork {
            role,
            addr,
            ping_count,
            payload_size,
            bandwidth_mb,
        } => {
            if role == "coordinator" {
                camelid::distributed::run_network_benchmark_coordinator(
                    &addr,
                    ping_count,
                    payload_size,
                    bandwidth_mb,
                )?;
            } else if role == "worker" {
                camelid::distributed::run_network_benchmark_worker(&addr)?;
            } else {
                anyhow::bail!("Invalid role: {role}. Must be 'coordinator' or 'worker'");
            }
        }
        Command::Inspect { path } => {
            let gguf = read_metadata(path)?;
            println!("{}", serde_json::to_string_pretty(&gguf)?);
        }
        Command::TensorDump {
            path,
            tensors,
            window,
            rows,
            tokens,
            layers,
        } => {
            let gguf = read_metadata(&path)?;
            let store = TensorStore::open(&path, &gguf);
            let names = tensor_dump_names(tensors, layers);
            let mut dumps = Vec::with_capacity(names.len());
            for name in names {
                dumps.push(dump_tensor(&store, &name, window, &rows, &tokens)?);
            }
            let dump = TensorDumpFile {
                path: path.display().to_string(),
                tensors: dumps,
            };
            println!("{}", serde_json::to_string_pretty(&dump)?);
        }
        Command::BenchDenseHotloops {
            hidden,
            ffn,
            repeats,
            warmup,
            threads,
        } => {
            configure_rayon_threads(threads)?;
            let report = bench_dense_hotloops(hidden, ffn, repeats, warmup)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::BenchQ8Blocks {
            path,
            tensor,
            rows,
            repeats,
            warmup,
            swap_rank2_shape,
            all_rows_dot,
            single_input_row_dot,
        } => {
            let report = bench_q8_blocks(Q8BlockBenchOptions {
                path: &path,
                tensor_name: &tensor,
                rows,
                repeats,
                warmup,
                swap_rank2_shape,
                all_rows_dot,
                single_input_row_dot,
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::DistributeWorker {
            path,
            addr,
            forward_addr,
            layers,
            master_addr,
            threads,
            cghost,
        } => {
            run_distribute_worker(
                path,
                addr,
                forward_addr,
                layers,
                master_addr,
                threads,
                cghost,
            )
            .await?;
        }
        Command::DistributeMaster {
            path,
            worker_addr,
            layers,
            addr,
            prompt,
            max_tokens,
            threads,
            cghost,
        } => {
            run_distribute_master(
                path,
                worker_addr,
                layers,
                addr,
                prompt,
                max_tokens,
                threads,
                cghost,
            )
            .await?;
        }
        Command::BenchGenerate {
            model,
            prompt_file,
            prompt,
            max_tokens,
            temperature,
            iterations,
            warmup,
            threads,
            json: _,
        } => {
            run_bench_generate(
                model,
                prompt_file,
                prompt,
                max_tokens,
                temperature,
                iterations,
                warmup,
                threads,
            )?;
        }
        Command::GhostRun {
            model,
            cghost,
            prompt,
            max_tokens,
            threads,
            sync_stream,
            evict_page_cache,
        } => {
            run_ghost(
                model,
                cghost,
                prompt,
                max_tokens,
                threads,
                sync_stream,
                evict_page_cache,
            )?;
        }
    }
    Ok(())
}

/// How ghost mode gets each layer's weights off disk. `range` is the node's pipeline shard
/// (the whole model on a single node); streaming cycles over it chunk after chunk.
struct GhostStreamer {
    range: std::ops::Range<usize>,
    kind: GhostStreamerKind,
}

enum GhostStreamerKind {
    /// v1: the read + decode happens on the critical path, before each layer's forward.
    Sync { ghost: Arc<GhostFile>, buf: Vec<u8> },
    /// v2 double-buffered: a background worker reads + decodes layer N+1 while layer N's
    /// forward runs; the reported time is only the STALL waiting for the handoff. The
    /// rendezvous handoff bounds the weight working set to two layer windows.
    Prefetched { prefetcher: GhostPrefetcher },
}

impl GhostStreamer {
    fn new_sync(ghost: Arc<GhostFile>, range: std::ops::Range<usize>) -> Self {
        Self {
            range,
            kind: GhostStreamerKind::Sync {
                buf: Vec::with_capacity(ghost.max_layer_span() as usize),
                ghost,
            },
        }
    }

    fn new_prefetched(ghost: Arc<GhostFile>, range: std::ops::Range<usize>) -> Self {
        Self {
            range,
            kind: GhostStreamerKind::Prefetched {
                prefetcher: GhostPrefetcher::spawn(ghost),
            },
        }
    }

    /// Queue the first chunk's layer reads (prefetched mode; no-op for sync).
    fn prime(&self) -> anyhow::Result<()> {
        if let GhostStreamerKind::Prefetched { prefetcher } = &self.kind {
            for layer_idx in self.range.clone() {
                prefetcher.request(layer_idx)?;
            }
        }
        Ok(())
    }

    /// Produce layer `layer_idx`'s decoded weights: (weights, bytes streamed, blocked µs).
    /// On the chunk's last layer the prefetched mode queues the ENTIRE next chunk first, so
    /// the worker is already rewinding to the shard's first layer for the next token while
    /// this layer's forward runs — on a mesh node that disk window overlaps the OTHER
    /// node's compute and the network hops. The trailing chunk queued after the final token
    /// is never consumed — the worker reads at most one extra layer, blocks on the
    /// rendezvous, and is released by Drop.
    fn fetch(
        &mut self,
        layer_idx: usize,
        last_in_chunk: bool,
    ) -> anyhow::Result<(LlamaLayerWeights, u64, u128)> {
        let range = self.range.clone();
        match &mut self.kind {
            GhostStreamerKind::Sync { ghost, buf } => {
                let started = Instant::now();
                let (layer, span) = ghost.read_layer(layer_idx, buf)?;
                Ok((layer, span, started.elapsed().as_micros()))
            }
            GhostStreamerKind::Prefetched { prefetcher } => {
                if last_in_chunk {
                    for next_idx in range {
                        prefetcher.request(next_idx)?;
                    }
                }
                let started = Instant::now();
                let prefetched = prefetcher.next()?;
                anyhow::ensure!(
                    prefetched.layer_idx == layer_idx,
                    "prefetcher returned layer {} but layer {layer_idx} was expected",
                    prefetched.layer_idx
                );
                Ok((
                    prefetched.weights,
                    prefetched.bytes,
                    started.elapsed().as_micros(),
                ))
            }
        }
    }
}

/// Build the ghost-mesh streaming context for a pipeline node: open the node's `.cghost`
/// shard, spawn the double-buffered prefetcher over the node's layer range, and prime the
/// first chunk. Returns None when the node runs the resident path. While this node waits on
/// the network (the other node computing), its prefetch worker is already streaming the
/// next token's layers — the disk window overlaps the peer's compute.
fn make_ghost_node_ctx(
    session: &LlamaInferenceSession,
    cghost: Option<&std::path::Path>,
    layer_range: std::ops::Range<usize>,
) -> anyhow::Result<Option<(GhostStreamer, LlamaLayerWeights)>> {
    let Some(path) = cghost else { return Ok(None) };
    let ghost = Arc::new(GhostFile::open(path)?);
    let n_layers = session.weights.layers.len();
    anyhow::ensure!(
        ghost.index.block_count == n_layers,
        ".cghost block_count {} does not match model block_count {n_layers}",
        ghost.index.block_count
    );
    let placeholder = session.weights.layers[0].clone();
    let streamer = GhostStreamer::new_prefetched(Arc::clone(&ghost), layer_range.clone());
    streamer.prime()?;
    println!(
        "[ghost] mesh node streams layers {:?} from {:?} ({:.1} MiB window, double-buffered)",
        layer_range,
        path,
        ghost.max_layer_span() as f64 / (1024.0 * 1024.0),
    );
    Ok(Some((streamer, placeholder)))
}

/// Ghost mode: run every transformer layer of one chunk (prefill or a single decoded
/// token), streaming each layer's weights from the `.cghost` file and dropping them right
/// after the layer's forward — the weight working window is one layer (sync) or two
/// (prefetched). Returns the chunk's output hidden state plus (bytes streamed, time blocked
/// on streaming, forward time).
fn ghost_stream_layers(
    session: &mut LlamaInferenceSession,
    streamer: &mut GhostStreamer,
    placeholder: &LlamaLayerWeights,
    hidden: CpuTensor,
    pos: usize,
    seq_len: usize,
    log_layers: bool,
) -> anyhow::Result<(CpuTensor, u64, u128, u128)> {
    let range = streamer.range.clone();
    let mut hidden = hidden;
    let mut bytes_total = 0u64;
    let mut wait_us_total = 0u128;
    let mut forward_us_total = 0u128;
    for layer_idx in range.clone() {
        let (layer, span, wait_us) = streamer.fetch(layer_idx, layer_idx + 1 == range.end)?;
        Arc::make_mut(&mut session.weights).layers[layer_idx] = layer;
        let forward_started = Instant::now();
        hidden = session.ghost_forward_one_layer(&hidden, layer_idx, pos, seq_len)?;
        let forward_us = forward_started.elapsed().as_micros();
        // Drop the streamed weights immediately; the window never accumulates.
        Arc::make_mut(&mut session.weights).layers[layer_idx] = placeholder.clone();
        bytes_total += span;
        wait_us_total += wait_us;
        forward_us_total += forward_us;
        if log_layers {
            eprintln!(
                "[ghost] layer {layer_idx:>3}: wait {:7.1} ms ({:6.1} MiB)  forward {:7.1} ms",
                wait_us as f64 / 1000.0,
                span as f64 / (1024.0 * 1024.0),
                forward_us as f64 / 1000.0,
            );
        }
    }
    session.ghost_advance_position(seq_len);
    Ok((hidden, bytes_total, wait_us_total, forward_us_total))
}

/// EXPERIMENTAL ghost (layer-streaming) mode: greedy generation with the model executed one
/// transformer block at a time from a `.cghost` file. RAM holds the embedding/output ends +
/// KV cache + the streaming window (one layer sync, two prefetched); everything else stays
/// on disk.
fn run_ghost(
    model: PathBuf,
    cghost: PathBuf,
    prompt: String,
    max_tokens: usize,
    threads: Option<usize>,
    sync_stream: bool,
    evict_page_cache: bool,
) -> anyhow::Result<()> {
    configure_rayon_threads(threads)?;
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("[ghost] loading GGUF metadata from {:?}...", model);
    let gguf = read_metadata(&model)?;
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
    let binding = camelid::model::LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let ghost = Arc::new(GhostFile::open_with_options(&cghost, evict_page_cache)?);
    let n_layers = config.block_count as usize;
    anyhow::ensure!(
        ghost.index.block_count == n_layers,
        ".cghost block_count {} does not match model block_count {n_layers}",
        ghost.index.block_count
    );

    // Resident ends only (embedding + output projection); every transformer layer is a
    // placeholder that ghost_stream_layers swaps real weights into, one at a time.
    let load_started = Instant::now();
    let weights = LlamaLoadedWeights::load_distributed(&store, &binding, 0, 0, true, true)?;
    let mut session = LlamaInferenceSession::new(config.clone(), Arc::new(weights))?;
    let placeholder = session.weights.layers[0].clone();
    let mut streamer = if sync_stream {
        GhostStreamer::new_sync(Arc::clone(&ghost), 0..n_layers)
    } else {
        GhostStreamer::new_prefetched(Arc::clone(&ghost), 0..n_layers)
    };
    println!(
        "[ghost] resident ends loaded in {:.1}s; {} layers x {:.1} MiB max streaming window \
         ({}, page cache {}); footprint {:.2} GiB",
        load_started.elapsed().as_secs_f64(),
        n_layers,
        ghost.max_layer_span() as f64 / (1024.0 * 1024.0),
        if sync_stream {
            "sync"
        } else {
            "double-buffered prefetch"
        },
        if evict_page_cache { "evicted" } else { "on" },
        gib(phys_footprint_bytes()),
    );

    let token_ids = tokenizer.encode(&prompt, true, false)?;
    println!("[ghost] prompt tokens: {:?}", token_ids);
    let mut pos = 0usize;

    let prefill_started = Instant::now();
    streamer.prime()?;
    let hidden = session
        .weights
        .token_embedding
        .embedding_lookup(&token_ids, "token_embedding_ghost")?;
    let (mut hidden, bytes, wait_us, forward_us) = ghost_stream_layers(
        &mut session,
        &mut streamer,
        &placeholder,
        hidden,
        pos,
        token_ids.len(),
        true,
    )?;
    pos += token_ids.len();
    println!(
        "[ghost] prefill: {:.1}s ({:.2} GiB streamed, blocked {:.1}s, forward {:.1}s); \
         footprint {:.2} GiB",
        prefill_started.elapsed().as_secs_f64(),
        gib(bytes),
        wait_us as f64 / 1_000_000.0,
        forward_us as f64 / 1_000_000.0,
        gib(phys_footprint_bytes()),
    );

    let mut generated: Vec<u32> = Vec::new();
    let mut decode_us_total: u128 = 0;
    for step in 0..max_tokens {
        let logits = session.forward_final_norm_and_logits(&hidden)?;
        let vocab = logits.dim(1)?;
        let rows = logits.dim(0)?;
        let last_row_start = (rows - 1) * vocab;
        let last_row = CpuTensor::from_f32(
            "ghost_last_logits",
            vec![1, vocab],
            logits.data[last_row_start..last_row_start + vocab].to_vec(),
        )?;
        let token = LlamaSampler::Greedy.sample(&last_row)?;
        generated.push(token);
        print!("{}", tokenizer.decode(&[token], true)?);
        std::io::stdout().flush()?;
        if tokenizer.special.eos == Some(token) || tokenizer.special.eot == Some(token) {
            break;
        }
        if step + 1 == max_tokens {
            break;
        }
        let token_started = Instant::now();
        let embedding = session
            .weights
            .token_embedding
            .embedding_lookup(&[token], "token_embedding_ghost")?;
        let (next_hidden, bytes, wait_us, forward_us) = ghost_stream_layers(
            &mut session,
            &mut streamer,
            &placeholder,
            embedding,
            pos,
            1,
            false,
        )?;
        hidden = next_hidden;
        pos += 1;
        let token_us = token_started.elapsed().as_micros();
        decode_us_total += token_us;
        eprintln!(
            "[ghost] token {:>3}: {:6.0} ms ({:.2} GiB streamed, blocked {:5.0} ms, forward \
             {:5.0} ms)",
            step + 1,
            token_us as f64 / 1000.0,
            gib(bytes),
            wait_us as f64 / 1000.0,
            forward_us as f64 / 1000.0,
        );
    }
    println!();

    let streamed_tokens = generated.len().saturating_sub(1);
    if streamed_tokens > 0 {
        println!(
            "[ghost] decode: {} tokens in {:.1}s = {:.3} tok/s",
            streamed_tokens,
            decode_us_total as f64 / 1_000_000.0,
            streamed_tokens as f64 / (decode_us_total as f64 / 1_000_000.0),
        );
    }
    println!(
        "[ghost] final footprint {:.2} GiB, peak RSS {:.2} GiB",
        gib(phys_footprint_bytes()),
        gib(peak_rss_bytes()),
    );
    Ok(())
}

/// One JSON metrics record per measured generation iteration (stdout, JSONL).
#[derive(Serialize)]
struct BenchGenerateRecord {
    runtime: &'static str,
    commit: String,
    model: String,
    quantization: String,
    iteration: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    load_ms: f64,
    prefill_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
    tokens_per_second: f64,
    peak_memory_bytes: u64,
    output_text: String,
    output_token_ids: Vec<u32>,
}

struct GenerationRun {
    generated: Vec<u32>,
    prefill_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
}

/// One full single-node generation with a fresh KV cache (weights are reused).
fn generate_run(
    config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    sampler: &LlamaSampler,
    max_tokens: usize,
) -> anyhow::Result<GenerationRun> {
    let mut session = LlamaInferenceSession::new(config.clone(), weights.clone())?;
    let mut history: Vec<u32> = prompt_tokens.to_vec();
    let mut input: Vec<u32> = prompt_tokens.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    // Prefill + first token: this whole span is time-to-first-token.
    let ttft_start = Instant::now();
    let step = session.generate_next_token_with_history_diagnostics(
        &input,
        sampler.clone(),
        &history,
        false,
    )?;
    let ttft_ms = ttft_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_ms = step.prefill_timings.total as f64 / 1000.0; // microseconds -> ms
    let first = step.next_token_id;
    generated.push(first);
    history.push(first);
    let mut finished = tokenizer.special.eog.contains(&first);
    input.clear();
    input.push(first);

    // Decode the remaining tokens (pure decode throughput).
    let decode_start = Instant::now();
    while !finished && generated.len() < max_tokens {
        let step = session.generate_next_token_with_history_diagnostics(
            &input,
            sampler.clone(),
            &history,
            false,
        )?;
        let next = step.next_token_id;
        generated.push(next);
        history.push(next);
        finished = tokenizer.special.eog.contains(&next);
        input.clear();
        input.push(next);
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    Ok(GenerationRun {
        generated,
        prefill_ms,
        ttft_ms,
        decode_ms,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_bench_generate(
    model: PathBuf,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    iterations: usize,
    warmup: bool,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    anyhow::ensure!(iterations >= 1, "--iterations must be at least 1");
    configure_rayon_threads(threads)?;

    let prompt_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };

    // Load the model once; this cost is measured separately from generation.
    let load_start = Instant::now();
    let gguf = read_metadata(&model)?;
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let prompt_token_ids = tokenizer.encode(&prompt_text, true, false)?;
    let prompt_tokens = prompt_token_ids.len();
    anyhow::ensure!(prompt_tokens >= 1, "prompt encoded to zero tokens");

    let sampler = if temperature <= 0.0 {
        LlamaSampler::Greedy
    } else {
        LlamaSampler::Sampling(SamplingConfig {
            temperature,
            ..Default::default()
        })
    };

    let commit = std::env::var("CAMELID_COMMIT").unwrap_or_else(|_| "unknown".to_string());
    let quantization = infer_quantization(&model);
    let model_label = model.display().to_string();

    if warmup {
        eprintln!("[bench-generate] warmup iteration (unmeasured)...");
        let _ = generate_run(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            &sampler,
            max_tokens,
        )?;
    }

    let stdout = std::io::stdout();
    for iteration in 0..iterations {
        let run = generate_run(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            &sampler,
            max_tokens,
        )?;
        let generated_tokens = run.generated.len();
        let decode_tokens = generated_tokens.saturating_sub(1);
        let tokens_per_second = if run.decode_ms > 0.0 && decode_tokens > 0 {
            decode_tokens as f64 / (run.decode_ms / 1000.0)
        } else {
            0.0
        };
        let output_text = tokenizer.decode(&run.generated, true).unwrap_or_default();
        let record = BenchGenerateRecord {
            runtime: "camelid",
            commit: commit.clone(),
            model: model_label.clone(),
            quantization: quantization.clone(),
            iteration,
            prompt_tokens,
            generated_tokens,
            load_ms,
            prefill_ms: run.prefill_ms,
            ttft_ms: run.ttft_ms,
            decode_ms: run.decode_ms,
            tokens_per_second,
            peak_memory_bytes: peak_rss_bytes(),
            output_text,
            output_token_ids: run.generated,
        };
        {
            let mut handle = stdout.lock();
            writeln!(handle, "{}", serde_json::to_string(&record)?)?;
            handle.flush()?;
        }
        eprintln!(
            "[bench-generate] iter {} | prompt {} tok | gen {} tok | ttft {:.1} ms | decode {:.1} ms | {:.2} tok/s | peak {:.2} GB",
            iteration,
            prompt_tokens,
            generated_tokens,
            record.ttft_ms,
            record.decode_ms,
            record.tokens_per_second,
            record.peak_memory_bytes as f64 / 1.073_741_824e9,
        );
    }
    Ok(())
}

/// Best-effort quantization label from the GGUF filename.
fn infer_quantization(path: &std::path::Path) -> String {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_uppercase();
    for q in [
        "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q2_K",
        "BF16", "F16", "F32",
    ] {
        if name.contains(q) {
            return q.to_string();
        }
    }
    "unknown".to_string()
}

/// Hard residency gate for pipeline nodes: every owned Q8_0 linear must hold plain
/// RAM-resident blocks, and the process memory footprint must account for them. Panics with
/// a per-tensor trace otherwise — a node is NEVER allowed to silently fall back to streaming
/// weights from disk per token (~100x slower decode, and it disqualifies the GPU-resident path).
fn assert_q8_0_weight_residency(weights: &LlamaLoadedWeights, node: &str) {
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let report: Q8ResidencyReport = weights.q8_0_residency_report();
    if !report.violations.is_empty() {
        eprintln!("[{node}] Q8_0 residency violations:");
        for violation in &report.violations {
            eprintln!("  - {violation}");
        }
        panic!(
            "[{node}] {} Q8_0 tensor(s) are NOT RAM-resident plain blocks; refusing to run",
            report.violations.len()
        );
    }
    // The retained blocks must show up in this process's physical footprint. The threshold
    // derives from the node's actual owned shard (a fixed floor would false-fail small
    // models and sharded splits); 90% slack covers allocator/OS accounting noise. A node
    // that silently fell back to disk streaming sits at a few hundred MB and misses this by
    // a wide margin. Footprint (not RSS) is the metric: macOS compresses untouched pages
    // under memory pressure, which drops them out of RSS while they are still materialized.
    let footprint = phys_footprint_bytes();
    let min_footprint = report.resident_block_bytes / 10 * 9;
    if footprint < min_footprint {
        panic!(
            "[{node}] memory footprint {:.2} GiB < required {:.2} GiB for {} retained Q8_0 \
             tensors ({:.2} GiB of blocks) — weights did not actually materialize in RAM",
            gib(footprint),
            gib(min_footprint),
            report.resident_tensors,
            gib(report.resident_block_bytes)
        );
    }
    println!(
        "[{node}] Q8_0 residency OK: {} tensors, {:.2} GiB retained blocks, footprint {:.2} GiB",
        report.resident_tensors,
        gib(report.resident_block_bytes),
        gib(footprint)
    );
}

/// Current physical memory footprint of this process in bytes — the metric Activity Monitor
/// and `/usr/bin/time -l`'s "memory footprint" report. Unlike RSS it includes pages the OS
/// compressed under memory pressure, so freshly-materialized weights are counted even on a
/// loaded machine. Falls back to peak RSS where unavailable.
fn phys_footprint_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        let ret = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as libc::c_int,
                libc::RUSAGE_INFO_V2,
                &mut info as *mut libc::rusage_info_v2 as *mut libc::rusage_info_t,
            )
        };
        if ret == 0 && info.ri_phys_footprint > 0 {
            return info.ri_phys_footprint;
        }
    }
    peak_rss_bytes()
}

/// Peak resident set size of this process. macOS reports bytes; Linux kilobytes.
fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret != 0 {
        return 0;
    }
    let max = usage.ru_maxrss.max(0) as u64;
    #[cfg(target_os = "macos")]
    {
        max
    }
    #[cfg(not(target_os = "macos"))]
    {
        max * 1024
    }
}

fn connect_with_retry(addr: SocketAddr) -> TcpStream {
    println!("Connecting to downstream {}...", addr);
    let start = Instant::now();
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => {
                stream.set_nodelay(true).unwrap();
                println!("Connected successfully to {}!", addr);
                return stream;
            }
            Err(e) => {
                // Pipeline nodes bind their sockets only after loading their weight
                // shard, which can take minutes for large models (especially when one
                // node streams from slower storage). Keep retrying well past that.
                if start.elapsed().as_secs() > 600 {
                    panic!("Failed to connect to {} after 600 seconds: {}", addr, e);
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
}

fn accept_connection(listener: &TcpListener) -> TcpStream {
    let (stream, client_addr) = listener.accept().unwrap();
    stream.set_nodelay(true).unwrap();
    println!("Accepted connection from upstream/client: {}", client_addr);
    stream
}

fn parse_layers_range(layers_str: &str) -> anyhow::Result<std::ops::Range<usize>> {
    let parts: Vec<&str> = layers_str.split("..").collect();
    if parts.len() != 2 {
        return Err(anyhow::anyhow!(
            "Invalid layers range format: {}",
            layers_str
        ));
    }
    let start = parts[0].parse::<usize>()?;
    let end = parts[1].parse::<usize>()?;
    Ok(start..end)
}

#[allow(clippy::too_many_arguments)]
async fn run_distribute_worker(
    path: PathBuf,
    addr: SocketAddr,
    forward_addr: Option<SocketAddr>,
    layers: String,
    master_addr: Option<SocketAddr>,
    threads: Option<usize>,
    cghost: Option<PathBuf>,
) -> anyhow::Result<()> {
    configure_rayon_threads(threads)?;

    println!("Loading GGUF metadata from {:?}...", path);
    let gguf = read_metadata(&path)?;
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
    let binding = camelid::model::LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&path, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf).ok();

    let layer_range = parse_layers_range(&layers)?;
    println!("Initializing worker session for layers {:?}", layer_range);

    let weights = Arc::new(if cghost.is_some() {
        // Ghost mesh: only the output ends stay resident (this is the LAST node when it has
        // no forward_addr); the layer shard streams from the .cghost per token.
        LlamaLoadedWeights::load_distributed(&store, &binding, 0, 0, false, true)?
    } else {
        LlamaLoadedWeights::load(&store, &binding, Some(layer_range.clone()))?
    });
    let mut session = LlamaInferenceSession::new(config.clone(), weights)?;
    assert_q8_0_weight_residency(&session.weights, "dist-worker");
    let mut ghost_ctx = make_ghost_node_ctx(&session, cghost.as_deref(), layer_range.clone())?;

    let listener = TcpListener::bind(addr)?;
    println!("Worker listening on {}...", addr);

    let mut downstream_stream = if let Some(faddr) = forward_addr {
        Some(connect_with_retry(faddr))
    } else {
        master_addr.map(connect_with_retry)
    };

    let mut client_stream = accept_connection(&listener);

    println!("Cluster worker execution loop active!");
    let trace = std::env::var_os("CAMELID_DISTRIBUTED_TRACE").is_some();
    let mut activations = Vec::new();

    loop {
        let idle_started = Instant::now();
        let header = match recv_activation_packet(&mut client_stream, &mut activations) {
            Ok(h) => h,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    println!("Upstream connection closed. Exiting worker loop.");
                    break;
                }
                return Err(e.into());
            }
        };

        let hidden_dim = config.embedding_length as usize;
        if activations.is_empty() || activations.len() % hidden_dim != 0 {
            return Err(anyhow::anyhow!(
                "Invalid activation packet size: {}",
                activations.len()
            ));
        }
        let rows = activations.len() / hidden_dim;
        let idle_us = idle_started.elapsed().as_micros();
        let hidden =
            CpuTensor::from_f32("activations", vec![rows, hidden_dim], activations.clone())?;

        let forward_started = Instant::now();
        let out_hidden = if let Some((streamer, placeholder)) = ghost_ctx.as_mut() {
            let (out, _bytes, _wait_us, _forward_us) = ghost_stream_layers(
                &mut session,
                streamer,
                placeholder,
                hidden,
                header.pos as usize,
                header.seq_len as usize,
                false,
            )?;
            out
        } else {
            session.forward_layer_range_from_hidden(
                &hidden,
                header.pos as usize,
                header.seq_len as usize,
            )?
        };
        let forward_us = forward_started.elapsed().as_micros();
        let tail_started = Instant::now();

        if let Some(ref mut ds) = downstream_stream {
            if forward_addr.is_some() {
                send_activation_packet(ds, header.pos, header.seq_len, &out_hidden.data)?;
            } else {
                let logits = session.forward_final_norm_and_logits(&out_hidden)?;
                let vocab_size = logits.dim(1)?;
                let last_row_start = (header.seq_len as usize - 1) * vocab_size;
                let last_row_data =
                    logits.data[last_row_start..last_row_start + vocab_size].to_vec();
                let last_row_logits =
                    CpuTensor::from_f32("last_row_logits", vec![1, vocab_size], last_row_data)?;
                let token_id = LlamaSampler::Greedy.sample(&last_row_logits)?;

                let is_finished = tokenizer.as_ref().is_some_and(|tok| {
                    tok.special.eos == Some(token_id) || tok.special.eot == Some(token_id)
                });

                send_token_feedback(ds, token_id, is_finished)?;
            }
        }
        if trace {
            eprintln!(
                "[dist-worker] pos={} rows={} idle={}us forward={}us logits_send={}us",
                header.pos,
                rows,
                idle_us,
                forward_us,
                tail_started.elapsed().as_micros()
            );
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_distribute_master(
    path: PathBuf,
    worker_addr: SocketAddr,
    layers: String,
    addr: SocketAddr,
    prompt: String,
    max_tokens: usize,
    threads: Option<usize>,
    cghost: Option<PathBuf>,
) -> anyhow::Result<()> {
    configure_rayon_threads(threads)?;

    println!("Loading GGUF metadata from {:?}...", path);
    let gguf = read_metadata(&path)?;
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
    let binding = camelid::model::LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&path, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let layer_range = parse_layers_range(&layers)?;
    println!("Initializing master session for layers {:?}", layer_range);

    let weights = Arc::new(if cghost.is_some() {
        // Ghost mesh: only the token embedding stays resident (the master is the FIRST
        // node); the layer shard streams from the .cghost per token.
        LlamaLoadedWeights::load_distributed(&store, &binding, 0, 0, true, false)?
    } else {
        LlamaLoadedWeights::load(&store, &binding, Some(layer_range.clone()))?
    });
    let mut session = LlamaInferenceSession::new(config.clone(), weights)?;
    assert_q8_0_weight_residency(&session.weights, "dist-master");
    let mut ghost_ctx = make_ghost_node_ctx(&session, cghost.as_deref(), layer_range.clone())?;

    let listener = TcpListener::bind(addr)?;
    println!("Master listening for feedback on {}...", addr);

    let mut downstream_stream = connect_with_retry(worker_addr);
    let mut feedback_stream = accept_connection(&listener);

    println!("Tokenizing prompt: {:?}", prompt);
    let token_ids = tokenizer.encode(&prompt, true, false)?;
    println!("Encoded prompt: {:?}", token_ids);

    let mut pos = 0usize;
    let mut seq_len = token_ids.len();

    let hidden = session
        .weights
        .token_embedding
        .embedding_lookup(&token_ids, "token_embedding_prefill")?;
    let out_hidden = if let Some((streamer, placeholder)) = ghost_ctx.as_mut() {
        ghost_stream_layers(
            &mut session,
            streamer,
            placeholder,
            hidden,
            pos,
            seq_len,
            false,
        )?
        .0
    } else {
        session.forward_layer_range_from_hidden(&hidden, pos, seq_len)?
    };

    send_activation_packet(
        &mut downstream_stream,
        pos as u32,
        seq_len as u32,
        &out_hidden.data,
    )?;

    let feedback = recv_token_feedback(&mut feedback_stream)?;
    let mut current_token = feedback.token_id;
    let mut is_finished = feedback.is_finished;

    print!("{}", tokenizer.decode(&[current_token], true)?);
    std::io::stdout().flush()?;

    pos += seq_len;
    seq_len = 1;

    let trace = std::env::var_os("CAMELID_DISTRIBUTED_TRACE").is_some();
    let decode_start = Instant::now();
    let mut generated = 1;
    while !is_finished && generated < max_tokens {
        let compute_started = Instant::now();
        let hidden = session
            .weights
            .token_embedding
            .embedding_lookup(&[current_token], "token_embedding")?;
        let out_hidden = if let Some((streamer, placeholder)) = ghost_ctx.as_mut() {
            ghost_stream_layers(
                &mut session,
                streamer,
                placeholder,
                hidden,
                pos,
                seq_len,
                false,
            )?
            .0
        } else {
            session.forward_layer_range_from_hidden(&hidden, pos, seq_len)?
        };
        let compute_us = compute_started.elapsed().as_micros();
        let send_started = Instant::now();
        send_activation_packet(
            &mut downstream_stream,
            pos as u32,
            seq_len as u32,
            &out_hidden.data,
        )?;
        let send_us = send_started.elapsed().as_micros();
        let wait_started = Instant::now();
        let feedback = recv_token_feedback(&mut feedback_stream)?;
        if trace {
            eprintln!(
                "[dist-master] pos={pos} compute={compute_us}us send={send_us}us wait={}us",
                wait_started.elapsed().as_micros()
            );
        }
        current_token = feedback.token_id;
        is_finished = feedback.is_finished;

        print!("{}", tokenizer.decode(&[current_token], true)?);
        std::io::stdout().flush()?;

        pos += 1;
        generated += 1;
    }
    println!();

    let decode_secs = decode_start.elapsed().as_secs_f64();
    let decode_tokens = generated.saturating_sub(1);
    if decode_tokens > 0 && decode_secs > 0.0 {
        println!(
            "[distributed] decode: {} tokens in {:.2}s = {:.2} tok/s",
            decode_tokens,
            decode_secs,
            decode_tokens as f64 / decode_secs
        );
    }

    Ok(())
}

fn tensor_dump_names(tensors: Vec<String>, layers: Vec<usize>) -> Vec<String> {
    let mut names = if tensors.is_empty() {
        default_tensor_dump_names()
    } else {
        tensors
    };

    for layer in layers {
        names.extend(layer_tensor_dump_names(layer));
    }
    dedup_preserving_order(names)
}

fn default_tensor_dump_names() -> Vec<String> {
    let mut names = vec!["token_embd.weight".to_string(), "output.weight".to_string()];
    names.extend(layer_tensor_dump_names(0));
    names
}

fn layer_tensor_dump_names(layer: usize) -> Vec<String> {
    [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]
    .into_iter()
    .map(|suffix| format!("blk.{layer}.{suffix}"))
    .collect()
}

fn dedup_preserving_order(names: Vec<String>) -> Vec<String> {
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

#[derive(Debug, Serialize)]
struct DenseHotloopBenchReport {
    hidden: usize,
    ffn: usize,
    repeats: usize,
    warmup: usize,
    rayon_threads: usize,
    checksum: f32,
    avg_ms: DenseHotloopBenchTimings,
    min_ms: DenseHotloopBenchTimings,
    max_ms: DenseHotloopBenchTimings,
}

#[derive(Debug, Serialize, Clone, Copy)]
struct DenseHotloopBenchTimings {
    gate: f64,
    up: f64,
    activation: f64,
    down: f64,
    total: f64,
}

#[derive(Debug, Serialize)]
struct Q8BlockBenchDeterminismReport {
    execution: &'static str,
    parallel_kernel_default: bool,
    serial_vs_parallel_delta_target: f32,
    serial_vs_parallel_delta_fail_threshold: f32,
}

#[derive(Debug, Serialize)]
struct Q8BlockBenchReport {
    path: String,
    tensor: String,
    shape: Vec<usize>,
    storage_shape: Vec<usize>,
    logical_shape: Vec<usize>,
    swap_rank2_shape: bool,
    tensor_n_bytes: u64,
    tensor_mib: f64,
    element_count: usize,
    block_count: usize,
    f32_materialized_mib: f64,
    retained_q8_payload_mib: f64,
    dot_input_f32_mib: f64,
    all_rows_output_f32_mib: Option<f64>,
    single_input_row_output_f32_mib: Option<f64>,
    determinism: Q8BlockBenchDeterminismReport,
    rows: Vec<usize>,
    row_len: usize,
    repeats: usize,
    warmup: usize,
    metadata_load_ms: f64,
    block_load_ms: f64,
    checksum: f32,
    avg_dequant_ms: f64,
    min_dequant_ms: f64,
    max_dequant_ms: f64,
    dot_checksum: f32,
    avg_dot_ms: f64,
    min_dot_ms: f64,
    max_dot_ms: f64,
    all_rows_dot: bool,
    all_rows_dot_checksum: Option<f32>,
    avg_all_rows_dot_ms: Option<f64>,
    min_all_rows_dot_ms: Option<f64>,
    max_all_rows_dot_ms: Option<f64>,
    single_input_row_dot: bool,
    single_input_row_dot_checksum: Option<f32>,
    avg_single_input_row_dot_ms: Option<f64>,
    min_single_input_row_dot_ms: Option<f64>,
    max_single_input_row_dot_ms: Option<f64>,
    dot_input_pattern: &'static str,
    notes: Vec<&'static str>,
}

struct Q8BlockBenchOptions<'a> {
    path: &'a PathBuf,
    tensor_name: &'a str,
    rows: Vec<usize>,
    repeats: usize,
    warmup: usize,
    swap_rank2_shape: bool,
    all_rows_dot: bool,
    single_input_row_dot: bool,
}

fn bench_q8_blocks(options: Q8BlockBenchOptions<'_>) -> anyhow::Result<Q8BlockBenchReport> {
    let Q8BlockBenchOptions {
        path,
        tensor_name,
        rows,
        repeats,
        warmup,
        swap_rank2_shape,
        all_rows_dot,
        single_input_row_dot,
    } = options;

    anyhow::ensure!(repeats > 0, "--repeats must be greater than zero");

    let started = Instant::now();
    let gguf = read_metadata(path)?;
    let metadata_load_ms = elapsed_ms(started);
    let store = TensorStore::open(path, &gguf);
    let desc = store.descriptor(tensor_name)?.clone();

    anyhow::ensure!(
        desc.tensor_type == GgufTensorType::Q8_0,
        "tensor {tensor_name} has storage type {:?}; bench-q8-blocks requires Q8_0",
        desc.tensor_type
    );

    let started = Instant::now();
    let mut tensor = store.load_q8_0_blocks(tensor_name)?;
    let block_load_ms = elapsed_ms(started);
    let storage_shape = tensor.shape.dims.clone();
    anyhow::ensure!(
        tensor.shape.dims.len() == 2,
        "bench-q8-blocks expects a rank-2 tensor, got {:?}",
        tensor.shape.dims
    );
    if swap_rank2_shape {
        tensor.shape.dims.swap(0, 1);
    }
    let row_count = tensor.shape.dims[0];
    let row_len = tensor.shape.dims[1];
    let rows = if rows.is_empty() { vec![0] } else { rows };
    for row in &rows {
        anyhow::ensure!(
            *row < row_count,
            "row {row} out of range for tensor {tensor_name} with {row_count} rows"
        );
    }

    let dot_input = bench_values(row_len, 0.00019);
    let single_input = if single_input_row_dot {
        Some(CpuTensor::from_f32(
            "bench_single_input",
            vec![1, row_len],
            dot_input.clone(),
        )?)
    } else {
        None
    };

    for _ in 0..warmup {
        let _ = dequantize_q8_rows_once(&tensor, &rows)?;
        let _ = dot_q8_rows_once(&tensor, &rows, &dot_input)?;
        if all_rows_dot {
            let _ = dot_q8_all_rows_once(&tensor, &dot_input)?;
        }
        if let Some(input) = &single_input {
            let _ = dot_q8_single_input_row_once(&tensor, input)?;
        }
    }

    let mut checksum = 0.0;
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        checksum += dequantize_q8_rows_once(&tensor, &rows)?;
        timings.push(elapsed_ms(started));
    }

    let mut dot_checksum = 0.0;
    let mut dot_timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        dot_checksum += dot_q8_rows_once(&tensor, &rows, &dot_input)?;
        dot_timings.push(elapsed_ms(started));
    }

    let (all_rows_dot_checksum, all_rows_dot_timings) = if all_rows_dot {
        let mut all_rows_checksum = 0.0;
        let mut timings = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let started = Instant::now();
            all_rows_checksum += dot_q8_all_rows_once(&tensor, &dot_input)?;
            timings.push(elapsed_ms(started));
        }
        (Some(all_rows_checksum), Some(timings))
    } else {
        (None, None)
    };

    let (single_input_row_dot_checksum, single_input_row_dot_timings) =
        if let Some(input) = &single_input {
            let mut single_input_checksum = 0.0;
            let mut timings = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let started = Instant::now();
                single_input_checksum += dot_q8_single_input_row_once(&tensor, input)?;
                timings.push(elapsed_ms(started));
            }
            (Some(single_input_checksum), Some(timings))
        } else {
            (None, None)
        };

    let element_count = tensor.element_count()?;
    let dot_input_f32_mib =
        bytes_to_mib(dot_input.len() as f64 * std::mem::size_of::<f32>() as f64);
    let output_vector_mib = bytes_to_mib(row_count as f64 * std::mem::size_of::<f32>() as f64);
    let all_rows_output_f32_mib = all_rows_dot.then_some(output_vector_mib);
    let single_input_row_output_f32_mib = single_input_row_dot.then_some(output_vector_mib);
    Ok(Q8BlockBenchReport {
        path: path.display().to_string(),
        tensor: tensor_name.to_string(),
        shape: tensor.shape.dims.clone(),
        storage_shape,
        logical_shape: tensor.shape.dims.clone(),
        swap_rank2_shape,
        tensor_n_bytes: desc.n_bytes,
        tensor_mib: bytes_to_mib(desc.n_bytes as f64),
        element_count,
        block_count: tensor.blocks.len(),
        f32_materialized_mib: bytes_to_mib(tensor.byte_size_if_f32_materialized()? as f64),
        retained_q8_payload_mib: bytes_to_mib(desc.n_bytes as f64),
        dot_input_f32_mib,
        all_rows_output_f32_mib,
        single_input_row_output_f32_mib,
        determinism: Q8BlockBenchDeterminismReport {
            execution: "serial_only_q8_0_block_rows",
            parallel_kernel_default: false,
            serial_vs_parallel_delta_target: 0.0,
            serial_vs_parallel_delta_fail_threshold: 1e-7,
        },
        rows,
        row_len,
        repeats,
        warmup,
        metadata_load_ms,
        block_load_ms,
        checksum,
        avg_dequant_ms: average_f64(&timings),
        min_dequant_ms: timings.iter().copied().fold(f64::INFINITY, f64::min),
        max_dequant_ms: timings.iter().copied().fold(0.0, f64::max),
        dot_checksum,
        avg_dot_ms: average_f64(&dot_timings),
        min_dot_ms: dot_timings.iter().copied().fold(f64::INFINITY, f64::min),
        max_dot_ms: dot_timings.iter().copied().fold(0.0, f64::max),
        all_rows_dot,
        all_rows_dot_checksum,
        avg_all_rows_dot_ms: all_rows_dot_timings.as_ref().map(|timings| average_f64(timings)),
        min_all_rows_dot_ms: all_rows_dot_timings
            .as_ref()
            .map(|timings| timings.iter().copied().fold(f64::INFINITY, f64::min)),
        max_all_rows_dot_ms: all_rows_dot_timings
            .as_ref()
            .map(|timings| timings.iter().copied().fold(0.0, f64::max)),
        single_input_row_dot,
        single_input_row_dot_checksum,
        avg_single_input_row_dot_ms: single_input_row_dot_timings
            .as_ref()
            .map(|timings| average_f64(timings)),
        min_single_input_row_dot_ms: single_input_row_dot_timings
            .as_ref()
            .map(|timings| timings.iter().copied().fold(f64::INFINITY, f64::min)),
        max_single_input_row_dot_ms: single_input_row_dot_timings
            .as_ref()
            .map(|timings| timings.iter().copied().fold(0.0, f64::max)),
        dot_input_pattern: "deterministic bench_values(row_len, 0.00019)",
        notes: vec![
            "Loads only the selected Q8_0 tensor payload as retained blocks, not full model f32 weights.",
            "Reports the bounded f32 activation input and optional output-vector sizes so memory pressure evidence distinguishes scratch/output buffers from avoided full f32 weight materialization.",
            "Benchmarks serial bounded row dequantization, row dot products, optional all-row dot output, and optional single-input-row lazy-linear adapter output; this is groundwork evidence for lazy/on-demand Q8_0 execution, not a generation-support claim.",
            "When swap_rank2_shape is true, the benchmark reinterprets rank-2 rows/cols without transposing payload bytes, matching the current guarded runtime layout path for selected rectangular LLaMA tensors.",
            "Determinism fields intentionally record that this bench path is serial-only today; any future parallel Q8 kernel must add serial-vs-parallel evidence targeting zero delta and failing above 1e-7 unless guarded off by default.",
        ],
    })
}

fn dequantize_q8_rows_once(tensor: &Q8_0TensorBlocks, rows: &[usize]) -> anyhow::Result<f32> {
    let mut checksum = 0.0;
    for row in rows {
        let values = tensor.dequantize_row(*row)?;
        checksum += values.iter().copied().sum::<f32>();
    }
    Ok(checksum)
}

fn dot_q8_rows_once(
    tensor: &Q8_0TensorBlocks,
    rows: &[usize],
    input: &[f32],
) -> anyhow::Result<f32> {
    let mut checksum = 0.0;
    for row in rows {
        checksum += tensor.dot_row_f32(*row, input)?;
    }
    Ok(checksum)
}

fn dot_q8_all_rows_once(tensor: &Q8_0TensorBlocks, input: &[f32]) -> anyhow::Result<f32> {
    let output = tensor.dot_all_rows_f32(input, "bench_all_rows_dot")?;
    Ok(output.data.iter().copied().sum::<f32>())
}

fn dot_q8_single_input_row_once(
    tensor: &Q8_0TensorBlocks,
    input: &CpuTensor,
) -> anyhow::Result<f32> {
    let output = tensor.dot_single_input_row_f32(input, "bench_single_input_row_dot")?;
    Ok(output.data.iter().copied().sum::<f32>())
}

fn bytes_to_mib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0)
}

fn average_f64(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn bench_dense_hotloops(
    hidden: usize,
    ffn: usize,
    repeats: usize,
    warmup: usize,
) -> anyhow::Result<DenseHotloopBenchReport> {
    anyhow::ensure!(hidden > 0, "--hidden must be greater than zero");
    anyhow::ensure!(ffn > 0, "--ffn must be greater than zero");
    anyhow::ensure!(repeats > 0, "--repeats must be greater than zero");

    let input = CpuTensor::from_f32("bench_input", vec![1, hidden], bench_values(hidden, 0.001))?;
    let gate = CpuTensor::from_f32(
        "bench_gate",
        vec![hidden, ffn],
        bench_values(hidden * ffn, 0.0003),
    )?;
    let up = CpuTensor::from_f32(
        "bench_up",
        vec![hidden, ffn],
        bench_values(hidden * ffn, 0.0005),
    )?;
    let down = CpuTensor::from_f32(
        "bench_down",
        vec![ffn, hidden],
        bench_values(ffn * hidden, 0.0007),
    )?;

    for _ in 0..warmup {
        let _ = run_dense_hotloop_once(&input, &gate, &up, &down)?;
    }

    let mut checksum = 0.0;
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let measured = run_dense_hotloop_once(&input, &gate, &up, &down)?;
        checksum += measured.checksum;
        timings.push(measured.timings);
    }

    Ok(DenseHotloopBenchReport {
        hidden,
        ffn,
        repeats,
        warmup,
        rayon_threads: rayon::current_num_threads(),
        checksum,
        avg_ms: average_timings(&timings),
        min_ms: min_timings(&timings),
        max_ms: max_timings(&timings),
    })
}

#[derive(Debug)]
struct DenseHotloopMeasurement {
    timings: DenseHotloopBenchTimings,
    checksum: f32,
}

fn run_dense_hotloop_once(
    input: &CpuTensor,
    gate: &CpuTensor,
    up: &CpuTensor,
    down: &CpuTensor,
) -> anyhow::Result<DenseHotloopMeasurement> {
    let total_started = Instant::now();

    let started = Instant::now();
    let gate_out = input.matmul(gate, "bench_gate_out")?;
    let gate_ms = elapsed_ms(started);

    let started = Instant::now();
    let up_out = input.matmul(up, "bench_up_out")?;
    let up_ms = elapsed_ms(started);

    let started = Instant::now();
    let activation = gate_out.silu_mul(&up_out, "bench_activation")?;
    let activation_ms = elapsed_ms(started);

    let started = Instant::now();
    let down_out = activation.matmul(down, "bench_down_out")?;
    let down_ms = elapsed_ms(started);

    Ok(DenseHotloopMeasurement {
        timings: DenseHotloopBenchTimings {
            gate: gate_ms,
            up: up_ms,
            activation: activation_ms,
            down: down_ms,
            total: elapsed_ms(total_started),
        },
        checksum: down_out.data.iter().copied().sum(),
    })
}

fn bench_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| (((idx % 97) as f32) - 48.0) * scale)
        .collect()
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn apply_runtime_tuning_env(
    parallel_linear_min_outputs: Option<usize>,
    apple_accelerate_min_elements: Option<usize>,
    metal_linear: bool,
    metal_q8: bool,
) {
    if let Some(value) = parallel_linear_min_outputs.filter(|value| *value > 0) {
        std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", value.to_string());
    }
    if let Some(value) = apple_accelerate_min_elements.filter(|value| *value > 0) {
        std::env::set_var("CAMELID_APPLE_ACCELERATE_MIN_ELEMENTS", value.to_string());
    }
    if metal_linear {
        std::env::set_var("CAMELID_METAL_LINEAR", "1");
    }
    if metal_q8 {
        std::env::set_var("CAMELID_METAL_Q8", "1");
    }
}

fn log_acceleration_state() {
    let metal = detect_metal_device();
    tracing::info!(
        rayon_threads = rayon::current_num_threads(),
        parallel_linear_min_outputs = std::env::var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS")
            .ok()
            .as_deref()
            .unwrap_or("default"),
        apple_accelerate_min_elements = std::env::var("CAMELID_APPLE_ACCELERATE_MIN_ELEMENTS")
            .ok()
            .as_deref()
            .unwrap_or("default(262144 on macOS)"),
        apple_accelerate = cfg!(target_os = "macos"),
        metal_linear = std::env::var("CAMELID_METAL_LINEAR")
            .ok()
            .as_deref()
            .unwrap_or("off"),
        metal_q8 = std::env::var("CAMELID_METAL_Q8")
            .ok()
            .as_deref()
            .unwrap_or("off"),
        metal_q8_retained = std::env::var("CAMELID_METAL_Q8_RETAINED")
            .ok()
            .as_deref()
            .unwrap_or("off"),
        hybrid_q8_retained = std::env::var("CAMELID_HYBRID_Q8_RETAINED")
            .ok()
            .as_deref()
            .unwrap_or("off"),
        hybrid_q8_gpu_percent = std::env::var("CAMELID_HYBRID_Q8_GPU_PERCENT")
            .ok()
            .as_deref()
            .unwrap_or("default(10)"),
        metal_available = metal.available,
        metal_device = metal.device_name.as_deref().unwrap_or("none"),
        metal_note = metal.note.as_deref().unwrap_or(""),
        "camelid acceleration state"
    );
}

fn configure_rayon_threads(threads: Option<usize>) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    let should_configure = true;
    #[cfg(not(target_os = "macos"))]
    let should_configure = threads.is_some();

    if should_configure {
        let mut builder = ThreadPoolBuilder::new();
        if let Some(t) = threads {
            anyhow::ensure!(t > 0, "--threads must be greater than zero");
            builder = builder.num_threads(t);
        }
        #[cfg(target_os = "macos")]
        {
            builder = builder.start_handler(|_| {
                unsafe {
                    pthread_set_qos_class_self_np(0x21, 0); // QOS_CLASS_USER_INTERACTIVE (forces P-cores)
                }
            });
        }
        builder
            .build_global()
            .map_err(|err| anyhow::anyhow!("failed to configure Rayon thread pool: {err}"))?;
    }
    Ok(())
}

fn average_timings(timings: &[DenseHotloopBenchTimings]) -> DenseHotloopBenchTimings {
    let mut total = DenseHotloopBenchTimings::zero();
    for timing in timings {
        total.add_assign(*timing);
    }
    total.scale(1.0 / timings.len() as f64)
}

fn min_timings(timings: &[DenseHotloopBenchTimings]) -> DenseHotloopBenchTimings {
    timings.iter().copied().fold(
        DenseHotloopBenchTimings::infinity(),
        DenseHotloopBenchTimings::min,
    )
}

fn max_timings(timings: &[DenseHotloopBenchTimings]) -> DenseHotloopBenchTimings {
    timings.iter().copied().fold(
        DenseHotloopBenchTimings::zero(),
        DenseHotloopBenchTimings::max,
    )
}

impl DenseHotloopBenchTimings {
    fn zero() -> Self {
        Self {
            gate: 0.0,
            up: 0.0,
            activation: 0.0,
            down: 0.0,
            total: 0.0,
        }
    }

    fn infinity() -> Self {
        Self {
            gate: f64::INFINITY,
            up: f64::INFINITY,
            activation: f64::INFINITY,
            down: f64::INFINITY,
            total: f64::INFINITY,
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.gate += other.gate;
        self.up += other.up;
        self.activation += other.activation;
        self.down += other.down;
        self.total += other.total;
    }

    fn scale(self, scale: f64) -> Self {
        Self {
            gate: self.gate * scale,
            up: self.up * scale,
            activation: self.activation * scale,
            down: self.down * scale,
            total: self.total * scale,
        }
    }

    fn min(self, other: Self) -> Self {
        Self {
            gate: self.gate.min(other.gate),
            up: self.up.min(other.up),
            activation: self.activation.min(other.activation),
            down: self.down.min(other.down),
            total: self.total.min(other.total),
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            gate: self.gate.max(other.gate),
            up: self.up.max(other.up),
            activation: self.activation.max(other.activation),
            down: self.down.max(other.down),
            total: self.total.max(other.total),
        }
    }
}

#[derive(Debug, Serialize)]
struct TensorDumpFile {
    path: String,
    tensors: Vec<TensorDump>,
}

#[derive(Debug, Serialize)]
struct TensorDump {
    name: String,
    descriptor: TensorDescriptorDump,
    q8_0: Option<Q8Dump>,
    decoded: DecodedTensorDump,
}

#[derive(Debug, Serialize)]
struct TensorDescriptorDump {
    gguf_dimensions: Vec<u64>,
    gguf_dimension_strides: Vec<u64>,
    runtime_shape: Vec<usize>,
    runtime_row_major_strides: Vec<usize>,
    tensor_type: GgufTensorType,
    absolute_offset: u64,
    relative_offset: u64,
    n_bytes: u64,
    element_count: usize,
    block_count: Option<usize>,
    storage_block_size: u64,
    storage_type_size_bytes: u64,
    storage_row_values: u64,
    storage_row_count: u64,
    storage_row_stride_values: u64,
    storage_row_size_bytes: u64,
    storage_row_stride_bytes: u64,
}

#[derive(Debug, Serialize)]
struct Q8Dump {
    block_count: usize,
    scale: NumberStats,
    first_scales: Vec<f32>,
    first_block_quants: Vec<i8>,
    max_abs_scale_block: usize,
    max_abs_scale_block_quants: Vec<i8>,
}

#[derive(Debug, Serialize)]
struct DecodedTensorDump {
    stats: NumberStats,
    first_values: Vec<f32>,
    max_abs_window_start: usize,
    max_abs_window: Vec<f32>,
    rows: Vec<RowDump>,
    logical_token_rows: Vec<LogicalTokenRowDump>,
    descriptor_token_columns: Vec<LogicalTokenRowDump>,
}

#[derive(Debug, Serialize)]
struct RowDump {
    row: usize,
    start: usize,
    len: usize,
    first_values: Vec<f32>,
    max_abs_window_start: usize,
    max_abs_window: Vec<f32>,
    q8_0_blocks: Vec<Q8BlockDump>,
    q8_0_value_checks: Vec<Q8ValueCheckDump>,
}

#[derive(Debug, Serialize)]
struct LogicalTokenRowDump {
    token_id: usize,
    start: usize,
    stride: usize,
    len: usize,
    source_layout: &'static str,
    first_values: Vec<f32>,
    max_abs_window_start: usize,
    max_abs_window: Vec<f32>,
    q8_0_blocks: Vec<Q8BlockDump>,
    q8_0_value_checks: Vec<Q8ValueCheckDump>,
}

#[derive(Debug, Serialize)]
struct Q8BlockDump {
    block: usize,
    value_start: usize,
    scale: f32,
    quant_values: Vec<i8>,
    dequantized_values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct Q8ValueCheckDump {
    element_index: usize,
    block: usize,
    block_offset: usize,
    scale: f32,
    quant_value: i8,
    dequantized: f32,
    decoded: f32,
    absolute_delta: f32,
}

#[derive(Debug, Serialize)]
struct NumberStats {
    min: f32,
    max: f32,
    mean: f64,
    rms: f64,
    max_abs: f32,
    max_abs_index: usize,
}

fn dump_tensor(
    store: &TensorStore,
    name: &str,
    window: usize,
    rows: &[usize],
    tokens: &[usize],
) -> anyhow::Result<TensorDump> {
    let desc = store.descriptor(name)?.clone();
    let tensor = store.load_cpu_f32(name)?;
    let bytes = store.tensor_bytes(name)?;
    let element_count = tensor.shape.element_count()?;
    let block_count = desc.tensor_type.layout().and_then(|(block_size, _)| {
        if block_size > 1 {
            usize::try_from(block_size)
                .ok()
                .map(|size| element_count / size)
        } else {
            None
        }
    });
    let row_dumps = dump_rows(
        &tensor.data,
        &tensor.shape.dims,
        &desc.tensor_type,
        &bytes,
        rows,
        window,
    )?;
    let logical_token_rows = dump_logical_token_rows(
        name,
        &tensor.data,
        &tensor.shape.dims,
        &desc.tensor_type,
        &bytes,
        tokens,
        window,
    )?;
    let descriptor_token_columns = dump_descriptor_token_columns(
        name,
        &tensor.data,
        &tensor.shape.dims,
        &desc.tensor_type,
        &bytes,
        tokens,
        window,
    )?;
    let storage = tensor_storage_layout(&desc.dimensions, desc.tensor_type)?;
    Ok(TensorDump {
        name: name.to_string(),
        descriptor: TensorDescriptorDump {
            gguf_dimension_strides: gguf_dimension_strides(&desc.dimensions),
            gguf_dimensions: desc.dimensions,
            runtime_row_major_strides: row_major_strides(&tensor.shape.dims),
            runtime_shape: tensor.shape.dims.clone(),
            tensor_type: desc.tensor_type,
            absolute_offset: desc.absolute_offset,
            relative_offset: desc.relative_offset,
            n_bytes: desc.n_bytes,
            element_count,
            block_count,
            storage_block_size: storage.block_size,
            storage_type_size_bytes: storage.type_size_bytes,
            storage_row_values: storage.row_values,
            storage_row_count: storage.row_count,
            storage_row_stride_values: storage.row_stride_values,
            storage_row_size_bytes: storage.row_size_bytes,
            storage_row_stride_bytes: storage.row_stride_bytes,
        },
        q8_0: match desc.tensor_type {
            GgufTensorType::Q8_0 => Some(dump_q8_0(&bytes, window)?),
            _ => None,
        },
        decoded: DecodedTensorDump {
            stats: number_stats(&tensor.data),
            first_values: tensor.data.iter().copied().take(window).collect(),
            max_abs_window_start: max_abs_window_start(&tensor.data, window),
            max_abs_window: window_around_max_abs(&tensor.data, window),
            rows: row_dumps,
            logical_token_rows,
            descriptor_token_columns,
        },
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TensorStorageLayoutDump {
    block_size: u64,
    type_size_bytes: u64,
    row_values: u64,
    row_count: u64,
    row_stride_values: u64,
    row_size_bytes: u64,
    row_stride_bytes: u64,
}

fn tensor_storage_layout(
    dimensions: &[u64],
    tensor_type: GgufTensorType,
) -> anyhow::Result<TensorStorageLayoutDump> {
    let (block_size, type_size_bytes) = tensor_type
        .layout()
        .ok_or_else(|| anyhow::anyhow!("unsupported tensor type {tensor_type:?}"))?;
    let row_values = *dimensions.first().unwrap_or(&1);
    if !row_values.is_multiple_of(block_size) {
        anyhow::bail!(
            "first tensor dimension {row_values} is not divisible by block size {block_size}"
        );
    }
    let row_count = dimensions.iter().skip(1).try_fold(1u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| anyhow::anyhow!("tensor storage row-count overflow"))
    })?;
    let row_size_bytes = row_values
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size_bytes))
        .ok_or_else(|| anyhow::anyhow!("tensor storage row-size overflow"))?;
    Ok(TensorStorageLayoutDump {
        block_size,
        type_size_bytes,
        row_values,
        row_count,
        row_stride_values: row_values,
        row_size_bytes,
        row_stride_bytes: row_size_bytes,
    })
}

fn dump_rows(
    values: &[f32],
    shape: &[usize],
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    rows: &[usize],
    window: usize,
) -> anyhow::Result<Vec<RowDump>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    if shape.len() != 2 {
        anyhow::bail!("--row requires 2D tensors, got shape {shape:?}");
    }
    let row_count = shape[0];
    let row_len = shape[1];
    let mut dumps = Vec::with_capacity(rows.len());
    for row in rows {
        if *row >= row_count {
            anyhow::bail!("row {row} out of range for shape {shape:?}");
        }
        let start = row * row_len;
        let slice = &values[start..start + row_len];
        let max_abs_offset = max_abs_window_start(slice, window);
        let q8_value_indices = sampled_q8_indices(start, row_len, 1, max_abs_offset, window);
        dumps.push(RowDump {
            row: *row,
            start,
            len: row_len,
            first_values: slice.iter().copied().take(window).collect(),
            max_abs_window_start: start + max_abs_offset,
            max_abs_window: window_around_max_abs(slice, window),
            q8_0_blocks: dump_q8_0_blocks_for_range(tensor_type, bytes, start, row_len, window)?,
            q8_0_value_checks: dump_q8_0_value_checks(
                tensor_type,
                bytes,
                values,
                q8_value_indices,
            )?,
        });
    }
    Ok(dumps)
}

fn dump_logical_token_rows(
    name: &str,
    values: &[f32],
    shape: &[usize],
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    tokens: &[usize],
    window: usize,
) -> anyhow::Result<Vec<LogicalTokenRowDump>> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    if shape.len() != 2 {
        anyhow::bail!("--token requires 2D tensors, got {name} shape {shape:?}");
    }
    let Some(layout) = logical_token_row_layout(name, shape) else {
        return Ok(Vec::new());
    };
    dump_token_rows_for_layout(values, tensor_type, bytes, tokens, window, layout)
}

fn dump_descriptor_token_columns(
    name: &str,
    values: &[f32],
    shape: &[usize],
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    tokens: &[usize],
    window: usize,
) -> anyhow::Result<Vec<LogicalTokenRowDump>> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let Some(layout) = descriptor_token_column_layout(name, shape) else {
        return Ok(Vec::new());
    };
    dump_token_rows_for_layout(values, tensor_type, bytes, tokens, window, layout)
}

fn dump_token_rows_for_layout(
    values: &[f32],
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    tokens: &[usize],
    window: usize,
    layout: LogicalTokenRowLayout,
) -> anyhow::Result<Vec<LogicalTokenRowDump>> {
    let mut dumps = Vec::with_capacity(tokens.len());
    for token_id in tokens {
        if *token_id >= layout.vocab_size {
            anyhow::bail!(
                "token {token_id} out of range for logical vocab size {}",
                layout.vocab_size
            );
        }
        let start = layout.start_for_token(*token_id);
        let row_values = strided_values(
            values,
            start,
            layout.embedding_width,
            layout.component_stride,
        );
        let max_abs_offset = max_abs_window_start(&row_values, window);
        let q8_value_indices = sampled_q8_indices(
            start,
            layout.embedding_width,
            layout.component_stride,
            max_abs_offset,
            window,
        );
        dumps.push(LogicalTokenRowDump {
            token_id: *token_id,
            start,
            stride: layout.component_stride,
            len: layout.embedding_width,
            source_layout: layout.source_layout,
            first_values: row_values.iter().copied().take(window).collect(),
            max_abs_window_start: start + max_abs_offset * layout.component_stride,
            max_abs_window: row_values
                .iter()
                .copied()
                .skip(max_abs_offset)
                .take(window)
                .collect(),
            q8_0_blocks: dump_q8_0_blocks_for_strided_row(
                tensor_type,
                bytes,
                start,
                layout.embedding_width,
                layout.component_stride,
                max_abs_offset,
                window,
            )?,
            q8_0_value_checks: dump_q8_0_value_checks(
                tensor_type,
                bytes,
                values,
                q8_value_indices,
            )?,
        });
    }
    Ok(dumps)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LogicalTokenRowLayout {
    vocab_size: usize,
    embedding_width: usize,
    token_start_stride: usize,
    component_stride: usize,
    source_layout: &'static str,
}

impl LogicalTokenRowLayout {
    fn start_for_token(self, token_id: usize) -> usize {
        token_id * self.token_start_stride
    }
}

fn logical_token_row_layout(name: &str, shape: &[usize]) -> Option<LogicalTokenRowLayout> {
    match name {
        "token_embd.weight" if shape[0] < shape[1] => Some(LogicalTokenRowLayout {
            vocab_size: shape[1],
            embedding_width: shape[0],
            token_start_stride: shape[0],
            component_stride: 1,
            source_layout: "gguf_token_major_shape_reinterpreted",
        }),
        "token_embd.weight" => Some(LogicalTokenRowLayout {
            vocab_size: shape[0],
            embedding_width: shape[1],
            token_start_stride: shape[1],
            component_stride: 1,
            source_layout: "runtime_token_major",
        }),
        "output.weight" if shape[0] < shape[1] => Some(LogicalTokenRowLayout {
            vocab_size: shape[1],
            embedding_width: shape[0],
            token_start_stride: shape[0],
            component_stride: 1,
            source_layout: "gguf_output_token_major_shape_reinterpreted",
        }),
        "output.weight" => Some(LogicalTokenRowLayout {
            vocab_size: shape[0],
            embedding_width: shape[1],
            token_start_stride: shape[1],
            component_stride: 1,
            source_layout: "token_major_output_row",
        }),
        _ => None,
    }
}

fn descriptor_token_column_layout(name: &str, shape: &[usize]) -> Option<LogicalTokenRowLayout> {
    match name {
        "output.weight" if shape.len() == 2 && shape[0] < shape[1] => Some(LogicalTokenRowLayout {
            vocab_size: shape[1],
            embedding_width: shape[0],
            token_start_stride: 1,
            component_stride: shape[1],
            source_layout: "descriptor_output_column",
        }),
        _ => None,
    }
}

fn strided_values(values: &[f32], start: usize, len: usize, stride: usize) -> Vec<f32> {
    (0..len).map(|idx| values[start + idx * stride]).collect()
}

fn gguf_dimension_strides(dims: &[u64]) -> Vec<u64> {
    let mut stride = 1u64;
    let mut strides = Vec::with_capacity(dims.len());
    for dim in dims {
        strides.push(stride);
        stride = stride.saturating_mul(*dim);
    }
    strides
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1usize; dims.len()];
    let mut stride = 1usize;
    for idx in (0..dims.len()).rev() {
        strides[idx] = stride;
        stride = stride.saturating_mul(dims[idx]);
    }
    strides
}

fn dump_q8_0(bytes: &[u8], window: usize) -> anyhow::Result<Q8Dump> {
    const BLOCK_BYTES: usize = 34;
    if !bytes.len().is_multiple_of(BLOCK_BYTES) {
        anyhow::bail!(
            "q8_0 byte length {} is not divisible by {BLOCK_BYTES}",
            bytes.len()
        );
    }
    let mut scales = Vec::with_capacity(bytes.len() / BLOCK_BYTES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        scales.push(f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]])));
    }
    let max_abs_scale_block = number_stats(&scales).max_abs_index;
    let first_block_quants = block_quants(bytes, 0, window);
    let max_abs_scale_block_quants = block_quants(bytes, max_abs_scale_block, window);
    Ok(Q8Dump {
        block_count: scales.len(),
        scale: number_stats(&scales),
        first_scales: scales.iter().copied().take(window).collect(),
        first_block_quants,
        max_abs_scale_block,
        max_abs_scale_block_quants,
    })
}

fn block_quants(bytes: &[u8], block_idx: usize, window: usize) -> Vec<i8> {
    const BLOCK_BYTES: usize = 34;
    let start = block_idx * BLOCK_BYTES + 2;
    bytes[start..start + 32]
        .iter()
        .copied()
        .map(|value| value as i8)
        .take(window)
        .collect()
}

fn dump_q8_0_blocks_for_range(
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    start: usize,
    len: usize,
    window: usize,
) -> anyhow::Result<Vec<Q8BlockDump>> {
    if *tensor_type != GgufTensorType::Q8_0 || len == 0 {
        return Ok(Vec::new());
    }
    dump_q8_0_blocks(bytes, [start, start + len - 1], window)
}

fn dump_q8_0_blocks_for_strided_row(
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    start: usize,
    len: usize,
    stride: usize,
    max_abs_offset: usize,
    window: usize,
) -> anyhow::Result<Vec<Q8BlockDump>> {
    if *tensor_type != GgufTensorType::Q8_0 || len == 0 {
        return Ok(Vec::new());
    }
    let first_indices = (0..len.min(window)).map(|offset| start + offset * stride);
    let max_window_end = len.min(max_abs_offset.saturating_add(window));
    let max_indices = (max_abs_offset..max_window_end).map(|offset| start + offset * stride);
    dump_q8_0_blocks(bytes, first_indices.chain(max_indices), window)
}

fn dump_q8_0_blocks(
    bytes: &[u8],
    indices: impl IntoIterator<Item = usize>,
    window: usize,
) -> anyhow::Result<Vec<Q8BlockDump>> {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let mut blocks = Vec::new();
    for index in indices {
        let block = index / BLOCK_VALUES;
        if blocks.iter().any(|dump: &Q8BlockDump| dump.block == block) {
            continue;
        }
        let byte_start = block * BLOCK_BYTES;
        if byte_start + BLOCK_BYTES > bytes.len() {
            anyhow::bail!(
                "q8_0 block {block} exceeds tensor byte length {}",
                bytes.len()
            );
        }
        let scale = f16_bits_to_f32(u16::from_le_bytes([
            bytes[byte_start],
            bytes[byte_start + 1],
        ]));
        let quant_values = bytes[byte_start + 2..byte_start + BLOCK_BYTES]
            .iter()
            .copied()
            .map(|value| value as i8)
            .take(window)
            .collect::<Vec<_>>();
        blocks.push(Q8BlockDump {
            block,
            value_start: block * BLOCK_VALUES,
            scale,
            dequantized_values: quant_values
                .iter()
                .map(|value| scale * f32::from(*value))
                .collect(),
            quant_values,
        });
    }
    Ok(blocks)
}

fn sampled_q8_indices(
    start: usize,
    len: usize,
    stride: usize,
    max_abs_offset: usize,
    window: usize,
) -> Vec<usize> {
    if len == 0 || window == 0 {
        return Vec::new();
    }
    let first_indices = (0..len.min(window)).map(|offset| start + offset * stride);
    let max_window_end = len.min(max_abs_offset.saturating_add(window));
    let max_indices = (max_abs_offset..max_window_end).map(|offset| start + offset * stride);
    dedup_usize_preserving_order(first_indices.chain(max_indices).collect())
}

fn dedup_usize_preserving_order(values: Vec<usize>) -> Vec<usize> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if !out.contains(&value) {
            out.push(value);
        }
    }
    out
}

fn dump_q8_0_value_checks(
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    values: &[f32],
    indices: Vec<usize>,
) -> anyhow::Result<Vec<Q8ValueCheckDump>> {
    if *tensor_type != GgufTensorType::Q8_0 {
        return Ok(Vec::new());
    }
    let mut checks = Vec::with_capacity(indices.len());
    for element_index in indices {
        checks.push(q8_0_value_check(bytes, values, element_index)?);
    }
    Ok(checks)
}

fn q8_0_value_check(
    bytes: &[u8],
    values: &[f32],
    element_index: usize,
) -> anyhow::Result<Q8ValueCheckDump> {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 34;
    if element_index >= values.len() {
        anyhow::bail!(
            "q8_0 value index {element_index} exceeds decoded tensor length {}",
            values.len()
        );
    }
    let block = element_index / BLOCK_VALUES;
    let block_offset = element_index % BLOCK_VALUES;
    let byte_start = block * BLOCK_BYTES;
    if byte_start + BLOCK_BYTES > bytes.len() {
        anyhow::bail!(
            "q8_0 block {block} for value index {element_index} exceeds tensor byte length {}",
            bytes.len()
        );
    }
    let scale = f16_bits_to_f32(u16::from_le_bytes([
        bytes[byte_start],
        bytes[byte_start + 1],
    ]));
    let quant_value = bytes[byte_start + 2 + block_offset] as i8;
    let dequantized = scale * f32::from(quant_value);
    let decoded = values[element_index];
    Ok(Q8ValueCheckDump {
        element_index,
        block,
        block_offset,
        scale,
        quant_value,
        dequantized,
        decoded,
        absolute_delta: (decoded - dequantized).abs(),
    })
}

fn number_stats(values: &[f32]) -> NumberStats {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut square_sum = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut max_abs_index = 0usize;
    for (idx, value) in values.iter().copied().enumerate() {
        min = min.min(value);
        max = max.max(value);
        sum += f64::from(value);
        square_sum += f64::from(value) * f64::from(value);
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
            max_abs_index = idx;
        }
    }
    let len = values.len() as f64;
    NumberStats {
        min,
        max,
        mean: sum / len,
        rms: (square_sum / len).sqrt(),
        max_abs,
        max_abs_index,
    }
}

fn max_abs_window_start(values: &[f32], window: usize) -> usize {
    if values.is_empty() || window == 0 {
        return 0;
    }
    let max_idx = number_stats(values).max_abs_index;
    max_idx.min(values.len().saturating_sub(window))
}

fn window_around_max_abs(values: &[f32], window: usize) -> Vec<f32> {
    let start = max_abs_window_start(values, window);
    values.iter().copied().skip(start).take(window).collect()
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = u32::from(bits & 0x03ff);
    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                let exp32 = u32::try_from(e + 127).expect("subnormal f16 exponent in range");
                sign | (exp32 << 23) | (mant << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tensor_dump_tests {
    use super::*;

    #[test]
    fn tensor_dump_layer_selection_extends_defaults_without_duplicates() {
        let names = tensor_dump_names(Vec::new(), vec![0, 2]);

        assert_eq!(names[0], "token_embd.weight");
        assert_eq!(names[1], "output.weight");
        assert!(names.contains(&"blk.0.attn_q.weight".to_string()));
        assert!(names.contains(&"blk.2.attn_q.weight".to_string()));
        assert!(names.contains(&"blk.2.ffn_down.weight".to_string()));
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "blk.0.attn_q.weight")
                .count(),
            1
        );
    }

    #[test]
    fn tensor_dump_layer_selection_extends_explicit_tensors() {
        let names = tensor_dump_names(vec!["output.weight".to_string()], vec![2]);

        assert_eq!(names[0], "output.weight");
        assert!(!names.contains(&"token_embd.weight".to_string()));
        assert_eq!(names[1], "blk.2.attn_q.weight");
        assert_eq!(
            names.last().map(String::as_str),
            Some("blk.2.ffn_down.weight")
        );
    }

    #[test]
    fn logical_token_row_layout_reports_embedding_and_output_strides() {
        assert_eq!(
            logical_token_row_layout("token_embd.weight", &[4, 10]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 4,
                component_stride: 1,
                source_layout: "gguf_token_major_shape_reinterpreted",
            })
        );
        assert_eq!(
            logical_token_row_layout("token_embd.weight", &[10, 4]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 4,
                component_stride: 1,
                source_layout: "runtime_token_major",
            })
        );
        assert_eq!(
            logical_token_row_layout("output.weight", &[4, 10]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 4,
                component_stride: 1,
                source_layout: "gguf_output_token_major_shape_reinterpreted",
            })
        );
        assert_eq!(
            descriptor_token_column_layout("output.weight", &[4, 10]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 1,
                component_stride: 10,
                source_layout: "descriptor_output_column",
            })
        );
        assert_eq!(
            logical_token_row_layout("output.weight", &[10, 4]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 4,
                component_stride: 1,
                source_layout: "token_major_output_row",
            })
        );
    }

    #[test]
    fn tensor_dump_reports_gguf_and_runtime_strides() {
        assert_eq!(gguf_dimension_strides(&[4, 10, 3]), vec![1, 4, 40]);
        assert_eq!(row_major_strides(&[4, 10, 3]), vec![30, 3, 1]);
    }

    #[test]
    fn tensor_dump_reports_q8_0_storage_row_size_and_stride() {
        let storage = tensor_storage_layout(&[2048, 32000], GgufTensorType::Q8_0)
            .expect("q8 output storage layout");

        assert_eq!(storage.block_size, 32);
        assert_eq!(storage.type_size_bytes, 34);
        assert_eq!(storage.row_values, 2048);
        assert_eq!(storage.row_count, 32000);
        assert_eq!(storage.row_stride_values, 2048);
        assert_eq!(storage.row_size_bytes, 2176);
        assert_eq!(storage.row_stride_bytes, 2176);
        assert_eq!(storage.row_size_bytes * storage.row_count, 69_632_000);
    }

    #[test]
    fn dump_logical_token_rows_samples_prompt_embedding_rows() {
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let rows = dump_logical_token_rows(
            "token_embd.weight",
            &values,
            &[3, 4],
            &GgufTensorType::F32,
            &[],
            &[0, 2],
            2,
        )
        .expect("logical token rows");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].token_id, 0);
        assert_eq!(rows[0].start, 0);
        assert_eq!(rows[0].stride, 1);
        assert_eq!(rows[0].len, 3);
        assert_eq!(rows[0].first_values, vec![0.0, 1.0]);
        assert_eq!(rows[1].token_id, 2);
        assert_eq!(rows[1].start, 6);
        assert_eq!(rows[1].first_values, vec![6.0, 7.0]);
        assert!(rows[0].q8_0_blocks.is_empty());
    }

    #[test]
    fn dump_logical_token_rows_samples_output_weight_token_vectors() {
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let rows = dump_logical_token_rows(
            "output.weight",
            &values,
            &[3, 4],
            &GgufTensorType::F32,
            &[],
            &[1],
            3,
        )
        .expect("output token rows");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].token_id, 1);
        assert_eq!(rows[0].start, 3);
        assert_eq!(rows[0].stride, 1);
        assert_eq!(rows[0].len, 3);
        assert_eq!(
            rows[0].source_layout,
            "gguf_output_token_major_shape_reinterpreted"
        );
        assert_eq!(rows[0].first_values, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn dump_descriptor_token_columns_samples_output_weight_descriptor_columns() {
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let rows = dump_descriptor_token_columns(
            "output.weight",
            &values,
            &[3, 4],
            &GgufTensorType::F32,
            &[],
            &[1],
            3,
        )
        .expect("output descriptor token columns");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].token_id, 1);
        assert_eq!(rows[0].start, 1);
        assert_eq!(rows[0].stride, 4);
        assert_eq!(rows[0].len, 3);
        assert_eq!(rows[0].source_layout, "descriptor_output_column");
        assert_eq!(rows[0].first_values, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn dump_rows_reports_q8_value_checks_for_contiguous_rows() {
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        for block in 0..4 {
            bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // scale 1.0
            for offset in 0..32 {
                let quant = block as i8 + offset as i8;
                bytes.push(quant as u8);
                values.push(f32::from(quant));
            }
        }

        let rows = dump_rows(&values, &[2, 64], &GgufTensorType::Q8_0, &bytes, &[1], 2)
            .expect("q8 row dump");

        let row = &rows[0];
        assert_eq!(row.row, 1);
        assert_eq!(row.start, 64);
        assert_eq!(row.first_values, vec![2.0, 3.0]);
        assert_eq!(row.max_abs_window_start, 126);
        assert_eq!(row.max_abs_window, vec![33.0, 34.0]);
        assert_eq!(row.q8_0_value_checks.len(), 4);
        assert_eq!(row.q8_0_value_checks[0].element_index, 64);
        assert_eq!(row.q8_0_value_checks[0].block, 2);
        assert_eq!(row.q8_0_value_checks[0].block_offset, 0);
        assert_eq!(row.q8_0_value_checks[0].quant_value, 2);
        assert_eq!(row.q8_0_value_checks[0].decoded, 2.0);
        assert_eq!(row.q8_0_value_checks[0].absolute_delta, 0.0);
        assert_eq!(row.q8_0_value_checks[3].element_index, 127);
        assert_eq!(row.q8_0_value_checks[3].block, 3);
        assert_eq!(row.q8_0_value_checks[3].block_offset, 31);
        assert_eq!(row.q8_0_value_checks[3].dequantized, 34.0);
    }

    #[test]
    fn dump_logical_token_rows_reports_q8_value_checks_for_token_major_output_rows() {
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        for block in 0..8 {
            bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // scale 1.0
            for offset in 0..32 {
                let quant = block as i8 + offset as i8;
                bytes.push(quant as u8);
                values.push(f32::from(quant));
            }
        }

        let rows = dump_logical_token_rows(
            "output.weight",
            &values,
            &[4, 64],
            &GgufTensorType::Q8_0,
            &bytes,
            &[1],
            2,
        )
        .expect("q8 output token row");

        let row = &rows[0];
        assert_eq!(row.start, 4);
        assert_eq!(row.stride, 1);
        assert_eq!(row.first_values, vec![4.0, 5.0]);
        assert_eq!(row.max_abs_window_start, 6);
        assert_eq!(row.max_abs_window, vec![6.0, 7.0]);
        assert_eq!(row.q8_0_blocks.len(), 1);
        assert_eq!(row.q8_0_blocks[0].block, 0);
        assert_eq!(row.q8_0_blocks[0].value_start, 0);
        assert_eq!(row.q8_0_blocks[0].quant_values, vec![0, 1]);
        assert_eq!(row.q8_0_blocks[0].dequantized_values, vec![0.0, 1.0]);
        assert_eq!(row.q8_0_value_checks.len(), 4);
        assert_eq!(row.q8_0_value_checks[0].element_index, 4);
        assert_eq!(row.q8_0_value_checks[0].block, 0);
        assert_eq!(row.q8_0_value_checks[0].block_offset, 4);
        assert_eq!(row.q8_0_value_checks[0].quant_value, 4);
        assert_eq!(row.q8_0_value_checks[0].dequantized, 4.0);
        assert_eq!(row.q8_0_value_checks[0].decoded, 4.0);
        assert_eq!(row.q8_0_value_checks[0].absolute_delta, 0.0);
        assert_eq!(row.q8_0_value_checks[3].element_index, 7);
        assert_eq!(row.q8_0_value_checks[3].block, 0);
        assert_eq!(row.q8_0_value_checks[3].block_offset, 7);
        assert_eq!(row.q8_0_value_checks[3].quant_value, 7);
    }

    #[test]
    fn dump_descriptor_token_columns_reports_strided_q8_value_checks() {
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        for block in 0..8 {
            bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // scale 1.0
            for offset in 0..32 {
                let quant = block as i8 + offset as i8;
                bytes.push(quant as u8);
                values.push(f32::from(quant));
            }
        }

        let rows = dump_descriptor_token_columns(
            "output.weight",
            &values,
            &[4, 64],
            &GgufTensorType::Q8_0,
            &bytes,
            &[1],
            2,
        )
        .expect("q8 output descriptor token column");

        let row = &rows[0];
        assert_eq!(row.start, 1);
        assert_eq!(row.stride, 64);
        assert_eq!(row.first_values, vec![1.0, 3.0]);
        assert_eq!(row.max_abs_window_start, 129);
        assert_eq!(row.max_abs_window, vec![5.0, 7.0]);
        assert_eq!(row.q8_0_blocks.len(), 4);
        assert_eq!(row.q8_0_blocks[0].block, 0);
        assert_eq!(row.q8_0_blocks[0].value_start, 0);
        assert_eq!(row.q8_0_blocks[0].quant_values, vec![0, 1]);
        assert_eq!(row.q8_0_blocks[0].dequantized_values, vec![0.0, 1.0]);
        assert_eq!(row.q8_0_value_checks.len(), 4);
        assert_eq!(row.q8_0_value_checks[0].element_index, 1);
        assert_eq!(row.q8_0_value_checks[0].block, 0);
        assert_eq!(row.q8_0_value_checks[0].block_offset, 1);
        assert_eq!(row.q8_0_value_checks[0].quant_value, 1);
        assert_eq!(row.q8_0_value_checks[0].dequantized, 1.0);
        assert_eq!(row.q8_0_value_checks[0].decoded, 1.0);
        assert_eq!(row.q8_0_value_checks[0].absolute_delta, 0.0);
        assert_eq!(row.q8_0_value_checks[3].element_index, 193);
        assert_eq!(row.q8_0_value_checks[3].block, 6);
        assert_eq!(row.q8_0_value_checks[3].block_offset, 1);
        assert_eq!(row.q8_0_value_checks[3].quant_value, 7);
    }

    #[test]
    fn dump_logical_token_rows_rejects_out_of_range_tokens() {
        let err = dump_logical_token_rows(
            "token_embd.weight",
            &[0.0; 12],
            &[3, 4],
            &GgufTensorType::F32,
            &[],
            &[4],
            2,
        )
        .expect_err("token should be out of range");
        assert!(err.to_string().contains("token 4 out of range"));
    }
}
