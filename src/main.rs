use std::{net::SocketAddr, path::PathBuf, time::Instant};

use camelid::{
    api,
    gguf::{read_metadata, GgufTensorType},
    tensor::{CpuTensor, Q8_0TensorBlocks, TensorStore},
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
        /// Override Rayon worker threads for API inference. Defaults to RAYON_NUM_THREADS/Rayon.
        #[arg(long, env = "CAMELID_RAYON_THREADS")]
        threads: Option<usize>,
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    match Cli::parse().command {
        Command::Serve { addr, threads } => {
            configure_rayon_threads(threads)?;
            api::serve(addr).await?;
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

fn configure_rayon_threads(threads: Option<usize>) -> anyhow::Result<()> {
    if let Some(threads) = threads {
        anyhow::ensure!(threads > 0, "--threads must be greater than zero");
        ThreadPoolBuilder::new()
            .num_threads(threads)
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
