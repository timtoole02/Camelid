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
    api, chat,
    cluster::{
        recv_activation_packet, recv_token_feedback, send_activation_packet, send_token_feedback,
    },
    gguf::{read_metadata, GgufTensorType},
    ghost::{GhostFile, GhostPipelinePrefetcher, GhostPrefetcher},
    inference::{
        speculative::{
            accepted_draft_prefix, ModelDrafter, NGramDrafter, SpeculativeDrafter,
            DEFAULT_MODEL_DRAFT_TOKENS, DEFAULT_NGRAM_DRAFT_TOKENS,
        },
        LlamaForwardTimings, LlamaInferenceSession, LlamaLayerWeights, LlamaLoadedWeights,
        LlamaSampler, Q8ResidencyReport, SamplingConfig,
    },
    metal::detect_metal_device,
    model::{KvCacheQuantization, LlamaModelConfig, LlamaTensorBinding},
    tensor::{CpuTensor, Q8_0TensorBlocks, TensorStore},
    tokenizer::Tokenizer,
};
use clap::{Parser, Subcommand};
use rayon::ThreadPoolBuilder;
use serde::Serialize;

// Prefer the git describe stamped in by build.rs (e.g. "v0.1.1" or
// "v0.1.1-3-gabcdef-dirty"); fall back to the crate version for builds without
// a git checkout.
const VERSION: &str = match option_env!("CAMELID_GIT_DESCRIBE") {
    Some(describe) => describe,
    None => env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Parser)]
#[command(
    name = "camelid",
    version = VERSION,
    about = "Rust-native local GGUF inference backend"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

/// The action taken when the binary is launched with no subcommand (e.g. a
/// double-click of `camelid.exe`): start the local server and open the chat UI,
/// with the GPU resident-decode path armed automatically. This is what makes the
/// shipped Windows build a single open-and-use app — no terminal, flags, or
/// toggles required.
fn default_launch_command() -> Command {
    Command::Serve {
        addr: "127.0.0.1:8181".parse().expect("valid default serve addr"),
        model: std::env::var_os("CAMELID_MODEL").map(PathBuf::from),
        threads: None,
        parallel_linear_min_outputs: None,
        apple_accelerate_min_elements: None,
        metal_linear: false,
        metal_q8: false,
        log_acceleration: true,
        spec_decode: None,
        spec_draft_model: None,
        spec_draft_tokens: None,
        no_open: false,
        deterministic: false,
        enable_thinking: false,
        models_dir: std::env::var_os("CAMELID_MODELS_DIR").map(PathBuf::from),
        // The double-click launch bypasses clap parsing, so the `env` on the `--gpu`
        // arg does not apply here; read CAMELID_GPU explicitly, like the sibling
        // env-backed fields above. Default (and any unrecognised value) is Auto.
        gpu: GpuMode::from_env(),
        kv_quant: std::env::var("CAMELID_KV_QUANT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_default(),
    }
}

/// Find a GGUF to load when `serve` is started without an explicit `--model`,
/// so the open-and-use launch lands directly in a usable chat. Looks next to the
/// executable (the shipped layout: `camelid.exe` beside a `models/` folder) and
/// in the working directory; returns the first `*.gguf` found.
fn auto_select_model() -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models"));
            dirs.push(parent.to_path_buf());
        }
    }
    dirs.push(PathBuf::from("models"));
    dirs.push(PathBuf::from("."));
    for dir in dirs {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut ggufs: Vec<PathBuf> = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| {
                    path.extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                })
                .collect();
            ggufs.sort();
            if let Some(first) = ggufs.into_iter().next() {
                return Some(first);
            }
        }
    }
    None
}

/// Windows + CUDA: make the NVIDIA runtime DLLs (NVRTC etc.) loadable without the
/// user having to add the CUDA `bin` directory to PATH. Looks in two places, in
/// priority order: (1) the running exe's OWN directory, so a self-contained
/// download that ships the NVRTC redistributable DLLs beside `camelid.exe` runs
/// on the GPU with only the NVIDIA *driver* installed (no CUDA toolkit); and
/// (2) an installed toolkit (via `CUDA_PATH*` or the standard install root). The
/// matching `bin`/exe dirs are prepended to the process PATH before any GPU code
/// runs. No-op if neither is present (the engine then falls back to CPU).
#[cfg(all(windows, feature = "cuda"))]
fn ensure_cuda_runtime_on_path() {
    use std::path::{Path, PathBuf};

    let mut candidates: Vec<PathBuf> = Vec::new();
    // The exe's own directory goes FIRST so a shipped, version-matched NVRTC pair
    // (staged by scripts/package-windows-cuda.ps1) wins over any — possibly
    // mismatched — system-installed toolkit. Windows already searches the exe dir
    // for `LoadLibrary`, but adding it to PATH explicitly is robust against
    // altered DLL search policies and makes the self-contained path intentional.
    // Only add it when NVRTC is actually present, to avoid polluting PATH (e.g. a
    // dev build under target/debug, where the DLLs are not staged).
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        if dir_has_nvrtc(&dir) {
            candidates.push(dir);
        }
    }
    for (key, value) in std::env::vars_os() {
        let key = key.to_string_lossy();
        if key == "CUDA_PATH" || key.starts_with("CUDA_PATH_V") {
            let bin = Path::new(&value).join("bin");
            if bin.is_dir() {
                candidates.push(bin);
            }
        }
    }
    let root = Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut versions: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        versions.sort();
        for version in versions.into_iter().rev() {
            let bin = version.join("bin");
            if bin.is_dir() {
                candidates.push(bin);
            }
        }
    }
    if candidates.is_empty() {
        return;
    }
    let current = std::env::var_os("PATH").unwrap_or_default();
    let current_lower = current.to_string_lossy().to_lowercase();
    let mut prefix = std::ffi::OsString::new();
    for bin in &candidates {
        if !current_lower.contains(&bin.to_string_lossy().to_lowercase()) {
            prefix.push(bin);
            prefix.push(";");
        }
    }
    if prefix.is_empty() {
        return;
    }
    prefix.push(current);
    std::env::set_var("PATH", prefix);
}

/// Whether `dir` contains an NVRTC runtime DLL (`nvrtc64_*.dll`). Used to decide
/// if the exe's own directory carries a shipped, self-contained CUDA runtime.
#[cfg(all(windows, feature = "cuda"))]
fn dir_has_nvrtc(dir: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .file_name()
            .to_string_lossy()
            .to_ascii_lowercase()
            .starts_with("nvrtc64_")
    })
}

#[cfg(not(all(windows, feature = "cuda")))]
fn ensure_cuda_runtime_on_path() {}

/// Stop cudarc's missing-library panics from printing, without silencing anything
/// else.
///
/// cudarc's lazy loaders panic when a CUDA library cannot be dlopen'd. We already
/// catch those and fall back to the CPU path, so on a host with an NVIDIA driver
/// but no CUDA toolkit the fallback is *working as designed* — but the default
/// panic hook still prints the panic plus a `RUST_BACKTRACE` hint on the way
/// through, so a healthy CPU-only start reads like a crash. That population is
/// now large: CUDA is compiled into the default x86_64 Linux build, and NVRTC
/// ships with the toolkit rather than the driver.
///
/// Installed ONCE, and it filters by panic origin rather than swapping the hook
/// around each call: a scoped swap would be process-global for its duration and
/// could swallow an unrelated thread's panic message (model loads run while the
/// server handles other requests). Every non-cudarc panic reaches the previous
/// hook untouched.
fn quiet_cudarc_loader_panics() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let from_cudarc = info
                .location()
                .is_some_and(|loc| loc.file().replace('\\', "/").contains("/cudarc-"));
            if !from_cudarc {
                previous(info);
            }
        }));
    });
}

// Optimus / Enduro hint variables, exported from the exe via build.rs. A laptop's
// hybrid-graphics driver reads these at process start and routes the process to
// the discrete NVIDIA / AMD GPU rather than the integrated Intel one. Without this
// Windows assigns the process to the iGPU by default, so Task Manager shows the
// Intel GPU "busy" even though CUDA compute runs (and can only run) on the NVIDIA
// card — the source of the "it's on Intel" confusion.
#[cfg(windows)]
#[no_mangle]
pub static NvOptimusEnablement: u32 = 1;
#[cfg(windows)]
#[no_mangle]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;

/// Tell Windows to run this executable on the high-performance (discrete NVIDIA)
/// GPU — the same setting as Settings → System → Display → Graphics → set the app
/// to "High performance". Writing it (HKCU, no admin needed) makes Windows and
/// Task Manager attribute the app to the NVIDIA GPU instead of the integrated
/// Intel one. Idempotent and best-effort; failures are ignored.
#[cfg(windows)]
fn pin_to_high_performance_gpu() {
    use std::os::windows::process::CommandExt;
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let exe = exe.to_string_lossy().to_string();
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = std::process::Command::new("reg")
        .args([
            "add",
            r"HKCU\Software\Microsoft\DirectX\UserGpuPreferences",
            "/v",
            &exe,
            "/t",
            "REG_SZ",
            "/d",
            "GpuPreference=2;",
            "/f",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
}

#[cfg(not(windows))]
fn pin_to_high_performance_gpu() {}

/// `camelid gait <action>` — GAIT cache maintenance subcommands.
#[derive(Debug, Subcommand)]
enum AgentAction {
    /// Run one goal to completion with no human present, print the final answer
    /// to stdout, and exit 0 (answered) / 1 (failed or blocked) / 3
    /// (inconclusive: step-capped, aborted, or no longer making progress).
    ///
    /// Progress narrates on stderr so stdout carries only the answer. With no
    /// operator to confirm anything, every approval-gated tool is DENIED unless
    /// --today-is-a-good-day-to-die (alias --yolo) is passed, and that flag is
    /// refused under CAMELID_PRODUCTION.
    Exec {
        /// The goal. Omit to read it from stdin.
        goal: Option<String>,
        /// GGUF to drive. Must be a tool-capable supported row.
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8231")]
        addr: SocketAddr,
        /// Sandbox root for the agent's file tools (default: cwd).
        #[arg(long)]
        workdir: Option<PathBuf>,
        #[arg(long, default_value_t = 25)]
        max_steps: usize,
        #[arg(long, default_value_t = 1024)]
        max_tokens: u32,
        /// Auto-approve write/network tools (exec tools still gated). Refused
        /// under CAMELID_PRODUCTION.
        #[arg(long, default_value_t = false)]
        auto_approve: bool,
        /// UNATTENDED: auto-approve everything including exec tools — today is
        /// a good day to die. Refused under CAMELID_PRODUCTION. (`--yolo` is a
        /// compatible alias.)
        #[arg(
            long = "today-is-a-good-day-to-die",
            visible_alias = "yolo",
            default_value_t = false
        )]
        yolo: bool,
        #[arg(long, default_value_t = false)]
        allow_net: bool,
        #[arg(long, default_value_t = false)]
        allow_fs: bool,
        #[arg(long, default_value_t = false)]
        allow_mcp: bool,
        #[arg(long, default_value = "sandboxed")]
        shell_sandbox: String,
        #[arg(long, default_value_t = 30)]
        shell_timeout: u64,
        #[arg(long)]
        models_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum WorkspaceAction {
    /// Ask one grounded, read-only question. Use --thread to continue a saved conversation.
    Ask {
        /// Local folder that confines all Workspace file tools.
        workspace: PathBuf,
        /// Question or analysis request.
        goal: String,
        /// Resume this durable Workspace conversation.
        #[arg(long)]
        thread: Option<String>,
        #[arg(long, default_value_t = 12)]
        max_steps: usize,
        #[arg(long, default_value_t = 512)]
        max_tokens: u32,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
    },
    /// List durable conversations for a local folder and the active model.
    Threads {
        #[arg(default_value = ".")]
        workspace: PathBuf,
    },
    /// Print a durable conversation transcript.
    Show {
        thread: String,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
    },
    /// Compact a durable conversation, or restore its previous compaction state.
    Compact {
        thread: String,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        #[arg(long, default_value_t = false)]
        undo: bool,
    },
    /// Permanently delete one durable conversation.
    Delete {
        thread: String,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
    },
}

#[cfg(test)]
mod workspace_command_tests {
    use super::*;

    fn on_cli_test_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("workspace-cli-parse-test".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(test)
            .expect("spawn Workspace CLI parse test")
            .join()
            .expect("Workspace CLI parse test panicked");
    }

    #[test]
    fn workspace_ask_parses_durable_resume_and_limits() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "workspace",
                "ask",
                ".",
                "inspect README.md",
                "--thread",
                "workspace-123",
                "--max-steps",
                "8",
                "--max-tokens",
                "256",
            ])
            .unwrap();
            match cli.command {
                Some(Command::Workspace {
                    action:
                        WorkspaceAction::Ask {
                            workspace,
                            goal,
                            thread,
                            max_steps,
                            max_tokens,
                            ..
                        },
                    ..
                }) => {
                    assert_eq!(workspace, PathBuf::from("."));
                    assert_eq!(goal, "inspect README.md");
                    assert_eq!(thread.as_deref(), Some("workspace-123"));
                    assert_eq!(max_steps, 8);
                    assert_eq!(max_tokens, 256);
                }
                other => panic!("expected workspace ask, got {other:?}"),
            }
        });
    }

    #[test]
    fn workspace_threads_uses_current_directory_and_json_is_global() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from(["camelid", "workspace", "threads", "--json"]).unwrap();
            match cli.command {
                Some(Command::Workspace {
                    json: true,
                    action: WorkspaceAction::Threads { workspace },
                    ..
                }) => assert_eq!(workspace, PathBuf::from(".")),
                other => panic!("expected workspace threads, got {other:?}"),
            }
        });
    }

    #[test]
    fn workspace_compaction_undo_is_explicit() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "workspace",
                "compact",
                "workspace-123",
                "--workspace",
                "project",
                "--undo",
            ])
            .unwrap();
            match cli.command {
                Some(Command::Workspace {
                    action:
                        WorkspaceAction::Compact {
                            thread,
                            workspace,
                            undo: true,
                        },
                    ..
                }) => {
                    assert_eq!(thread, "workspace-123");
                    assert_eq!(workspace, PathBuf::from("project"));
                }
                other => panic!("expected workspace compact --undo, got {other:?}"),
            }
        });
    }
}

#[derive(Debug, Subcommand)]
enum GaitAction {
    /// Clear the GAIT cache (profiles, quarantine, in-progress markers, and the
    /// DISABLE kill-file) under %LOCALAPPDATA%\Camelid\gait, fully reverting to
    /// the baseline path.
    Reset,
}

/// GPU acceleration mode for `serve` (`--gpu`, env `CAMELID_GPU`). Seeds the runtime
/// GPU switches at startup for headless/agent use where there is no UI to click; the
/// UI toggle (`POST /api/runtime/gpu`) can still flip state live afterwards.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum GpuMode {
    /// Use the GPU when a CUDA device is detected; fall back to CPU otherwise (default).
    /// Leaves the switches to lazy-seed exactly as today (gpu_accel from is_available(),
    /// hybrid Q8 matmul from CAMELID_CUDA_Q8).
    Auto,
    /// Force the GPU path on (still no-ops at runtime without a device).
    On,
    /// Force the CPU reference path; never use the GPU even if one is present.
    Off,
}

impl GpuMode {
    /// Resolve from `CAMELID_GPU` for the double-click launch path, which bypasses clap
    /// arg parsing (so the `env` on the `--gpu` arg does not apply there). Values match
    /// the clap value names (`auto`/`on`/`off`, case-insensitive); an unset or
    /// unrecognised value falls back to `Auto`, identical to the clap default.
    fn from_env() -> Self {
        match std::env::var("CAMELID_GPU") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "on" => GpuMode::On,
                "off" => GpuMode::Off,
                _ => GpuMode::Auto,
            },
            Err(_) => GpuMode::Auto,
        }
    }
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
        /// Lossless greedy speculative decoding mode: "ngram" (prompt lookup,
        /// no extra weights) or "draft" (a smaller same-tokenizer model
        /// drafts; requires --spec-draft-model). Default off. A serving
        /// optimization only — it makes no support claim for any lane.
        #[arg(long, env = "CAMELID_SPEC_DECODE")]
        spec_decode: Option<String>,
        /// Draft model GGUF for --spec-decode draft (must share the target's
        /// exact token mapping).
        #[arg(long, env = "CAMELID_SPEC_DRAFT_MODEL")]
        spec_draft_model: Option<PathBuf>,
        /// Draft tokens proposed per speculation round (default: 8 for
        /// ngram, 5 for draft).
        #[arg(long, env = "CAMELID_SPEC_DRAFT_TOKENS")]
        spec_draft_tokens: Option<usize>,
        /// Do not open the web UI in a browser on startup. By default, when run
        /// interactively, `serve` opens the chat surface automatically.
        #[arg(long, env = "CAMELID_NO_OPEN", default_value_t = false)]
        no_open: bool,
        /// Opt into deterministic inference: pin the forward pass to the order-stable
        /// CPU path (the whole Metal/GPU fast stack is forced off) so the supported
        /// TinyLlama 1.1B Q8_0 lane is bit-exact and reduction-order-stable across runs.
        /// Slower than the default GPU path; the default path is unchanged. Reduction
        /// order follows the llama.cpp reference Q8_0 layout (see DECISIONS.md §D9).
        #[arg(long, env = "CAMELID_DETERMINISTIC", default_value_t = false)]
        deterministic: bool,
        /// Default Qwen3/gemma4 thinking mode ON for chat requests that don't set
        /// it themselves. Opt-in and NOT parity-locked: thinking mode is supported
        /// only as a leading-trace lane (the first tokens match the llama.cpp
        /// reference before a benign f32 near-tie); the parity-locked exact-row
        /// mode remains thinking-DISABLED. A client that sends
        /// `camelid_enable_thinking` explicitly always wins over this default.
        #[arg(long, env = "CAMELID_ENABLE_THINKING", default_value_t = false)]
        enable_thinking: bool,
        /// Directory holding local GGUF models: scanned by the Models page
        /// (`/api/models/local`), the catalog download target, and the fallback
        /// base for RELATIVE model paths sent to the load endpoints (absolute
        /// paths, and relative paths that exist against the working directory,
        /// are used as given). Defaults to the first existing of
        /// `<exe dir>/models` or `./models` — the shipped layout — falling back
        /// to `./models`.
        #[arg(long, env = "CAMELID_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        /// GPU acceleration mode. `auto` (default) uses the GPU when a CUDA device is
        /// present, `on` forces it, `off` pins the CPU reference path. Seeds the runtime
        /// switches at startup for headless/agent runs with no UI; `on`/`off` override the
        /// env seed, and the Settings toggle can still flip state live after startup.
        #[arg(long = "gpu", value_enum, default_value_t = GpuMode::Auto, env = "CAMELID_GPU")]
        gpu: GpuMode,
        /// KV cache quantization format: "f16" (default, unquantized), "q8_0"
        /// (50% memory savings), or "q4_0" (75% memory savings).
        #[arg(long, env = "CAMELID_KV_QUANT", default_value_t = KvCacheQuantization::F16)]
        kv_quant: KvCacheQuantization,
    },
    /// Interactive terminal chat REPL over the local Camelid API.
    ///
    /// Attaches to (or spawns) a `camelid serve`, opens a supported-model picker,
    /// and streams `/v1/chat/completions` live. Switch models in-session with
    /// `/models`.
    Chat {
        /// Load this GGUF at startup (same semantics as `serve --model`). Omit to
        /// open the supported-model picker.
        #[arg(long, env = "CAMELID_MODEL")]
        model: Option<PathBuf>,
        /// Server to attach to, or spawn on if nothing is listening there.
        #[arg(long, default_value = "127.0.0.1:8181", env = "CAMELID_ADDR")]
        addr: SocketAddr,
        /// Initial system prompt.
        #[arg(long)]
        system: Option<String>,
        /// Maximum tokens to generate per turn.
        #[arg(long, default_value_t = 512)]
        max_tokens: u32,
        /// Sampling temperature (0 = greedy/deterministic).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Nucleus sampling top-p (omit to leave unset).
        #[arg(long)]
        top_p: Option<f32>,
        /// Top-k sampling (omit to leave unset).
        #[arg(long)]
        top_k: Option<u32>,
        /// Sampling seed (omit for the engine default).
        #[arg(long)]
        seed: Option<u64>,
        /// Print the full response after completion instead of streaming.
        #[arg(long, default_value_t = false)]
        no_stream: bool,
        /// Force the inline line REPL instead of the full-screen TUI.
        #[arg(long, default_value_t = false)]
        plain: bool,
        /// Directory holding downloaded GGUFs (picker availability + pull target).
        #[arg(long, env = "CAMELID_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        /// Enter agent mode: a sandboxed tool-calling loop (requires a
        /// tool-capable supported model and `--model`).
        #[arg(long, default_value_t = false)]
        agent: bool,
        /// Sandbox root for agent file/shell tools (default: current directory).
        #[arg(long)]
        workdir: Option<PathBuf>,
        /// Max agent steps (tool-call rounds) per goal.
        #[arg(long, default_value_t = 25)]
        max_steps: usize,
        /// Agent: run write/network tools WITHOUT prompting (exec tools stay
        /// gated; sandbox still enforced). Prints a warning; not recommended.
        #[arg(long, default_value_t = false)]
        auto_approve: bool,
        /// Agent: UNATTENDED — auto-approve EVERYTHING including exec tools
        /// (shell, GUI input, run_windows_command, spawn_subagent) so the agent
        /// runs a whole task without prompting — today is a good day to die.
        /// Bounded by --max-steps and /stop. Refused under CAMELID_PRODUCTION.
        /// Powerful + dangerous; opt-in. (`--yolo` is a compatible alias.)
        #[arg(
            long = "today-is-a-good-day-to-die",
            visible_alias = "yolo",
            default_value_t = false
        )]
        yolo: bool,
        /// Agent: offer the network tool (`http_fetch`). Off by default.
        #[arg(long, default_value_t = false)]
        allow_net: bool,
        /// Agent: let the file tools read/write anywhere on disk (computer
        /// control), not just under --workdir. Still approval-gated. Off by
        /// default (file tools are confined to the workspace root).
        #[arg(long, default_value_t = false)]
        allow_fs: bool,
        /// Agent: load MCP servers declared in `camelid.mcp.json` at the
        /// workspace root and offer their tools. Third-party code: every MCP
        /// tool is approval-gated like a shell command, its output is treated
        /// as untrusted data, and the whole feature is refused under
        /// CAMELID_PRODUCTION. Off by default.
        #[arg(long, default_value_t = false)]
        allow_mcp: bool,
        /// Agent: shell-command timeout in seconds.
        #[arg(long, default_value_t = 30)]
        shell_timeout: u64,
        /// Render Qwen3/gemma4 thinking mode: the model emits its own
        /// `<think>…</think>` reasoning before answering. Opt-in and NOT
        /// parity-locked — supported only as a leading-trace lane (see
        /// `--enable-thinking` on `serve`). Default off keeps the parity-locked
        /// thinking-DISABLED rendering.
        #[arg(long, env = "CAMELID_ENABLE_THINKING", default_value_t = false)]
        enable_thinking: bool,
        /// Agent: POST audit events (agent.tool_call / agent.tool_result) as JSON
        /// to this URL. Delivery is async + non-blocking (drops on backpressure).
        /// Unset → no audit. No endpoint is built in.
        #[arg(long, env = "CAMELID_AUDIT_WEBHOOK")]
        audit_webhook: Option<String>,
        /// Agent: run_shell confinement — `disabled` (tool not offered),
        /// `sandboxed` (default; seccomp+uid-drop on Linux, fails closed where
        /// unenforceable), or `unrestricted` (cwd-pinned + timed only).
        #[arg(long, default_value = "sandboxed")]
        shell_sandbox: String,
    },
    /// Tool-capability promotion harness: decide whether a model drives a clean
    /// tool-call round-trip (PASS / FAIL / INCONCLUSIVE) and emit a receipt. A
    /// contended box that can't load in time yields INCONCLUSIVE, never FAIL.
    AgentEval {
        /// GGUF to evaluate.
        #[arg(long)]
        model: PathBuf,
        /// Server to attach to / spawn on.
        #[arg(long, default_value = "127.0.0.1:8181")]
        addr: SocketAddr,
        /// Seconds to wait for the model to load before reporting INCONCLUSIVE.
        #[arg(long, default_value_t = 90)]
        load_timeout: u64,
        /// Max agent steps per case.
        #[arg(long, default_value_t = 6)]
        max_steps: usize,
        /// Max tokens per model turn.
        #[arg(long, default_value_t = 256)]
        max_tokens: u32,
        /// Directory for the receipt artifact.
        #[arg(long, default_value = "qa/agent-eval")]
        receipt_dir: PathBuf,
    },
    /// Phase-1 Windows system-control gate: exercise run_windows_command +
    /// inspect_system under the sandbox/approval contract and emit a sealed
    /// receipt (PASS / FAIL / INCONCLUSIVE). Rung-1 — promotes nothing.
    AgentSyscapEval {
        /// Directory for the receipt artifact.
        #[arg(long, default_value = "qa/agent-syscap")]
        receipt_dir: PathBuf,
    },
    /// Internal: run ONE scoped subagent task described by a task file and write
    /// its result file. Spawned by the spawn_subagent tool; not for direct use.
    #[command(name = "__subagent", hide = true)]
    Subagent {
        /// Path to the task_<id>.json written by the parent.
        #[arg(long)]
        task_file: PathBuf,
    },
    /// Phase-2 subagent-orchestration gate: spawn -> run -> collect a canned
    /// subagent plus caps/depth/reaping checks, emitting a sealed receipt
    /// (PASS / FAIL / INCONCLUSIVE). Rung-2 (stub) — promotes nothing.
    AgentOrchestrationEval {
        /// Directory for the receipt artifact.
        #[arg(long, default_value = "qa/agent-orchestration")]
        receipt_dir: PathBuf,
        /// Optional GGUF: run the rung-3 REAL-model round-trip instead of the
        /// canned rung-2 mechanics battery.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Server to attach to / spawn on (rung-3).
        #[arg(long, default_value = "127.0.0.1:8181")]
        addr: SocketAddr,
        /// Seconds to wait for the model to load before reporting INCONCLUSIVE.
        #[arg(long, default_value_t = 120)]
        load_timeout: u64,
    },
    /// Rung-4: measure concurrent vs sequential subagent wall-clock (I/O-bound;
    /// add --model for the inference-bound workload) and emit a sealed receipt.
    AgentOrchestrationBench {
        /// Directory for the receipt artifact.
        #[arg(long, default_value = "qa/agent-orchestration")]
        receipt_dir: PathBuf,
        /// Optional GGUF: also measure the inference-bound workload.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Server to attach to / spawn on.
        #[arg(long, default_value = "127.0.0.1:8181")]
        addr: SocketAddr,
        /// Seconds to wait for the model to load.
        #[arg(long, default_value_t = 120)]
        load_timeout: u64,
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
    #[command(hide = true)]
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
    /// Runnable-lane smoke-admission for a single GGUF: admit -> load -> greedy
    /// forward sanity -> coherence, on oracle-qualified combos only. Prints a
    /// RUNNABLE receipt (lane=runnable, never copper; attests deterministic
    /// execution, NOT parity) to stdout on pass; exits non-zero on refusal/failure.
    RunnableSmoke { path: PathBuf },
    /// Tokenize text through the model's tokenizer (parity-harness utility,
    /// mirrors llama.cpp's `llama-tokenize`). Input is either `--prompt` or
    /// `--file` (a JSON array of strings; exact bytes preserved). Prints one
    /// JSON object per input: {"ids":[...],"decoded":"..."} where `decoded`
    /// is the decode round-trip of the encoded ids (specials retained).
    Tokenize {
        /// GGUF model (tokenizer metadata source).
        #[arg(long)]
        model: PathBuf,
        /// Single prompt to tokenize.
        #[arg(long, short = 'p', conflicts_with = "file")]
        prompt: Option<String>,
        /// JSON file: array of strings to tokenize.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Parse special tokens (e.g. <|im_start|>) into their single IDs.
        #[arg(long)]
        parse_special: bool,
        /// Do not add BOS/EOS even if the model's metadata asks for them.
        #[arg(long)]
        no_add_special: bool,
    },
    /// Layer offloading — Phase 1: print the planned VRAM/host layer split for a
    /// model (no weights loaded, no compute). `--budget-mb` forces a small VRAM
    /// budget to demonstrate partial offload; `--arch <name>` plans a known
    /// architecture without its GGUF file.
    PlanOffload {
        /// GGUF model to plan (reads real per-layer tensor sizes).
        model: Option<PathBuf>,
        /// Known architecture to plan without a file (e.g. "llama-8b").
        #[arg(long)]
        arch: Option<String>,
        /// Override detected free VRAM (MiB) to force a split.
        #[arg(long)]
        budget_mb: Option<u64>,
        /// KV-cache context length to reserve (default 4096).
        #[arg(long)]
        context: Option<u64>,
        /// Safety margin in MiB (default 256).
        #[arg(long)]
        safety_mb: Option<u64>,
    },
    /// Download a supported model (a known-good Q8_0 GGUF) into ./models.
    ///
    /// Run with no argument to list the catalog. Accepts a catalog id or a
    /// fragment of the name, e.g. `camelid pull llama32_3b`.
    Pull {
        /// Catalog id or name fragment to download. Omit to list all models.
        model: Option<String>,
        /// Directory to download into (default: ./models).
        #[arg(long, env = "CAMELID_MODELS_DIR")]
        models_dir: Option<PathBuf>,
    },
    /// Generate text with a Gemma 4 model (correctness-first runtime).
    Gemma4Generate {
        path: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 24)]
        max_tokens: usize,
        /// BASALT forced-decode harness (basalt_eval_protocol.md §5.1):
        /// teacher-force the token ids in this file (newline-separated decimal
        /// ids, or one JSON array). Each step feeds the forced token as the
        /// next input REGARDLESS of the model's argmax, while the per-step
        /// argmax id + logit are still recorded (stdout JSON). Ignores
        /// --max-tokens and stop tokens: the list length defines the step count.
        #[arg(long)]
        force_tokens: Option<PathBuf>,
        /// Write each step's FULL logit vector into this directory as raw
        /// little-endian f32 `step_<i>.bin` files, plus a `meta.json` (vocab
        /// size, step count, prompt info, per-step top-32 (id, logit)). Works
        /// with or without --force-tokens.
        #[arg(long)]
        dump_step_logits: Option<PathBuf>,
    },
    /// Load-amortized BASALT eval-pack runner (`basalt_eval_protocol.md` §5.1):
    /// load a gemma4 model ONCE and run every prompt in the given packs. This is
    /// the load-once form of the per-prompt `Gemma4Generate` harness, for the G3(b)
    /// teacher-forced top-1 agreement metric when iterating over many quant rows.
    ///
    /// Without `--score`: greedy-generate each prompt and write
    /// `<baseline_dir>/<prompt_id>.txt` (newline token ids) — the Q8_0 reference.
    /// With `--score`: teacher-force each prompt's reference ids from
    /// `<baseline_dir>` through THIS model and report teacher-forced top-1
    /// agreement (overall + per prompt). CPU path; no engine math change.
    Gemma4EvalPack {
        path: PathBuf,
        /// Pack JSON file(s), e.g. qa/gemma4/prompt_packs/basic_v1.json.
        #[arg(long = "pack", required = true, num_args = 1..)]
        packs: Vec<PathBuf>,
        /// Directory holding (score mode) or receiving (baseline mode) the
        /// per-prompt reference token-id files `<prompt_id>.txt`.
        #[arg(long)]
        baseline_dir: PathBuf,
        /// Score this model's teacher-forced agreement against the reference ids
        /// in `--baseline-dir` instead of generating them.
        #[arg(long)]
        score: bool,
    },
    /// Generate with the CUDA-resident Gemma 4 lane (dev harness for the SSER build).
    #[cfg(feature = "cuda")]
    Gemma4CudaGenerate {
        path: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 24)]
        max_tokens: usize,
    },
    /// Generate text with a Gemma 4 model on the GPU (resident decode; macOS/Metal).
    Gemma4GenerateGpu {
        path: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 24)]
        max_tokens: usize,
    },
    /// Chat with a DiffusionGemma model: render the chat template, run the
    /// bit-exact multi-canvas block-autoregressive denoise loop, detokenize.
    /// CPU-only and slow (each denoise step is a full bidirectional forward);
    /// experimental — see the DiffusionGemma lane recon.
    DiffusionGemmaChat {
        path: PathBuf,
        #[arg(long, default_value = "Hello")]
        prompt: String,
        /// Max blocks (each block denoises one canvas_length window, then
        /// commits to the prefix). The answer stops earlier on an end token,
        /// a repetition loop, or the ubatch budget.
        #[arg(long, default_value_t = 4)]
        max_blocks: i32,
        /// Entropy-Bound sampler seed (reference default 0).
        #[arg(long, default_value_t = 0)]
        seed: u32,
        /// Max ubatch (the whole [prefix | canvas] must fit in one ubatch).
        #[arg(long, default_value_t = 1100)]
        max_ubatch: i32,
        /// Override the EB sampler's max denoise steps per block (reference
        /// default 48, with adaptive early stop). Lower it (e.g. 1-2) for a
        /// fast correctness signal — each step is a full bidirectional forward.
        #[arg(long)]
        max_steps: Option<i32>,
    },
    /// Serve the TAIL layers of a Gemma 4 model as a distributed worker
    /// (layer sharding over TCP; pair with gemma4-master on the other Mac).
    Gemma4Worker {
        path: PathBuf,
        #[arg(long, default_value = "0.0.0.0:5005")]
        addr: String,
        /// First (global) layer this worker owns; it owns through the final
        /// layer plus the output head. Must not split the shared-KV block.
        #[arg(long)]
        first_layer: usize,
    },
    /// Run the HEAD layers of a Gemma 4 model and drive a distributed worker
    /// for the tail (greedy decode; distributed layer sharding, not shared memory).
    Gemma4Master {
        path: PathBuf,
        #[arg(long)]
        worker_addr: String,
        /// Layers [0, split) run locally; [split, block_count) on the worker.
        #[arg(long)]
        split: usize,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 24)]
        max_tokens: usize,
    },
    /// Dump focused tensor descriptor, raw block, and f32 dequantization diagnostics.
    #[command(hide = true)]
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
    #[command(hide = true)]
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
    /// Hidden: decode zero-alloc gate — decode real tokens through a loaded
    /// model under the counting global allocator and report steady-state heap
    /// allocations per token. Requires `--features alloc-gate`.
    #[cfg(feature = "alloc-gate")]
    #[command(hide = true)]
    BenchAllocGate {
        /// GGUF model to decode.
        #[arg(long)]
        model: std::path::PathBuf,
        /// Unmeasured tokens to warm pools, binding cells, and KV growth.
        #[arg(long, default_value_t = 8)]
        warmup: usize,
        /// Measured steady-state tokens.
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        /// Skip the final norm + logits projection (attribution mode).
        #[arg(long, default_value_t = false)]
        skip_logits: bool,
        /// Print backtraces for the first few >=1MiB steady-state allocations.
        #[arg(long, default_value_t = false)]
        trace_big: bool,
        /// Fail (non-zero exit) if allocations per token exceed this.
        #[arg(long)]
        max_per_token: Option<f64>,
    },
    /// Hidden: micro-benchmark rayon fork-join region overhead on the global
    /// pool (hot = back-to-back regions; cold = workers idle between regions).
    #[command(hide = true)]
    BenchRayonRegion {
        /// Measured regions per point.
        #[arg(long, default_value_t = 10_000)]
        iterations: usize,
        /// Idle time between regions in microseconds (0 = hot).
        #[arg(long, default_value_t = 0)]
        idle_us: u64,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Hidden: hot-cache micro-benchmark of the attention f32 dot kernels
    /// (legacy scalar chain vs canonical blocked scalar vs blocked AVX2/FMA).
    #[command(hide = true)]
    BenchAttnDot {
        /// Vector lengths to measure (defaults cover the real head dims).
        #[arg(long = "len", default_values_t = [64usize, 128])]
        lens: Vec<usize>,
        /// Measured iterations per variant.
        #[arg(long, default_value_t = 2_000_000)]
        repeats: usize,
        /// Unreported warmup iterations per variant.
        #[arg(long, default_value_t = 100_000)]
        warmup: usize,
    },
    /// Load one GGUF Q8_0 tensor as retained blocks and benchmark bounded row dequantization/dot rows.
    #[command(hide = true)]
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
    #[command(hide = true)]
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
        /// Opt into deterministic inference: pin generation to the order-stable CPU
        /// forward pass (Metal/GPU fast stack forced off) so the supported TinyLlama
        /// 1.1B Q8_0 lane is bit-exact across runs, thread counts, and processes.
        /// Slower than the default GPU path; the default path is unchanged.
        #[arg(long, env = "CAMELID_DETERMINISTIC", default_value_t = false)]
        deterministic: bool,
    },
    /// Hidden: in-process INTERLEAVED owner-microkernel prefill sweep. Loads the model ONCE, then
    /// rotates owner configs (off / avx2 / vnni4x4 / vnni4x8) round-by-round so every config shares
    /// the same thermal/clock state, enabling drift-cancelling PAIRED comparison (the fix for the
    /// noise that made v3 inconclusive). The owner flag is read from env per linear call, so no
    /// reload is needed between configs. Emits one JSON line per (round, config) to stdout.
    #[command(hide = true)]
    BenchOwnerSweep {
        /// GGUF model path.
        model: PathBuf,
        /// Read the prompt from this UTF-8 file. Takes precedence over --prompt.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Inline prompt text (used when --prompt-file is absent).
        #[arg(long)]
        prompt: Option<String>,
        /// Tokens to generate per measurement (prefill dominates; keep small).
        #[arg(long, default_value_t = 1)]
        max_tokens: usize,
        /// Measured interleaved rounds (median + paired stats taken across rounds).
        #[arg(long, default_value_t = 10)]
        rounds: usize,
        /// Leading rounds discarded as warmup (reach steady thermal state).
        #[arg(long, default_value_t = 2)]
        warmup_rounds: usize,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// GAIT: run the parity-gated calibration tournament for this model on this
    /// machine. Times the supported execution profiles, disqualifies any whose
    /// greedy output diverges, picks the fastest parity-clean one that beats the
    /// baseline by a margin (else fails closed to baseline), and persists a
    /// `camelid.gait-receipt/v1`. Writes only a receipt; changes no decode path.
    /// The receipt is consumed later only when `CAMELID_GAIT` is set.
    #[command(hide = true)]
    GaitCalibrate {
        /// GGUF model path.
        model: PathBuf,
        /// Read the prompt from this UTF-8 file. Takes precedence over --prompt.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Inline prompt text (used when --prompt-file is absent).
        #[arg(long)]
        prompt: Option<String>,
        /// Tokens to generate per trial (greedy/deterministic).
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Measured interleaved rounds per variant (median is taken). More rounds
        /// reject thermal/clock noise at the cost of calibration time.
        #[arg(long, default_value_t = 4)]
        rounds: usize,
        /// Leading rounds discarded as warmup.
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// GAIT internal: run ONE calibration candidate trial in isolation and print
    /// its TrialResult as a single JSON line. Spawned as a child process by the
    /// calibration supervisor (§1.4 crash isolation); not for direct use.
    #[command(hide = true)]
    GaitTrial {
        /// GGUF model path.
        model: PathBuf,
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Profile label: auto | safe | experimental | debug.
        #[arg(long)]
        profile: String,
        /// Apply the compute-pool EcoQoS opt-out for this trial.
        #[arg(long, default_value_t = false)]
        eco_qos: bool,
        #[arg(long)]
        threads: Option<usize>,
        /// §5 groups_per_chunk tiling overrides (pass all three together).
        #[arg(long)]
        gpc_attn: Option<usize>,
        #[arg(long)]
        gpc_ffn: Option<usize>,
        #[arg(long)]
        gpc_matmul: Option<usize>,
    },
    /// Headless agent runs (e.g. `camelid agent exec "fix the failing test"`).
    Agent {
        #[command(subcommand)]
        action: AgentAction,
    },
    /// Grounded, durable, read-only conversations with a local folder.
    Workspace {
        /// Address of an already-running loopback Camelid server.
        #[arg(
            long,
            global = true,
            default_value = "127.0.0.1:8181",
            env = "CAMELID_ADDR"
        )]
        addr: SocketAddr,
        /// Emit compact JSON (JSON Lines for `ask`).
        #[arg(long, global = true, default_value_t = false)]
        json: bool,
        /// Maximum time to wait for one `ask` event stream.
        #[arg(long, global = true, default_value_t = 1800)]
        timeout_seconds: u64,
        #[command(subcommand)]
        action: WorkspaceAction,
    },
    /// GAIT cache maintenance (e.g. `camelid gait reset`).
    #[command(hide = true)]
    Gait {
        #[command(subcommand)]
        action: GaitAction,
    },
    /// SPEC_RECHECK measurement harness: run lossless greedy speculative decode and a plain
    /// greedy baseline back-to-back on one prompt, and emit a single JSON record with the
    /// per-run economics (acceptance rate, draft/verify latency split, f_draft, S_sync) plus
    /// the lossless verdict (the spec token stream's first divergence vs Camelid plain greedy).
    /// Default off; moves no support ledger; reuses the existing drafters + GPU verify.
    BenchSpeculative {
        /// Target GGUF model path (the model whose output must be reproduced exactly).
        model: PathBuf,
        /// Drafter: "ngram" (prompt lookup, no draft model) or "draft" (a smaller
        /// same-tokenizer model; requires --draft-model).
        #[arg(long, default_value = "ngram")]
        drafter: String,
        /// Draft model GGUF for --drafter draft. Must share the target's token mapping.
        #[arg(long)]
        draft_model: Option<PathBuf>,
        /// Drafted tokens per round (γ). Capped at MAX_VERIFY_K - 1 = 7 by the verify path.
        #[arg(long)]
        draft_tokens: Option<usize>,
        /// Force the draft model onto the CPU forward path (Path 3 in SPEC_RECHECK). Default
        /// leaves the draft GPU-resident (its own drafter cache), falling back to CPU only if
        /// it does not fit in VRAM.
        #[arg(long, default_value_t = false)]
        cpu_draft: bool,
        /// Build the coexistence target + resident draft once (drafter/reserve set BEFORE the
        /// target builds) and measure only the speculative run; the plain reference reuses that
        /// same resident target. Avoids an in-process full-size target rebuild whose VRAM the
        /// cudarc pool will not release back to the sizing probe. The plain tps reported here is
        /// the same-config (coexistence) target, not a full-resident-target baseline.
        #[arg(long, default_value_t = false)]
        spec_only: bool,
        /// Read the prompt from this UTF-8 file. Takes precedence over --prompt.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Inline prompt text (used when --prompt-file is absent).
        #[arg(long)]
        prompt: Option<String>,
        /// Workload label recorded in the JSON (e.g. "code", "json", "extraction").
        #[arg(long, default_value = "unlabeled")]
        workload: String,
        /// Maximum tokens to generate (fixed per workload for a reproducible matrix).
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Run one unmeasured warmup pair before the measured run.
        #[arg(long, default_value_t = false)]
        warmup: bool,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// EXPERIMENTAL ghost (layer-streaming) mode: execute a model one transformer block at
    /// a time, streaming each block's weights from a layer-contiguous `.cghost` file
    /// (see the `repack-ghost` tool) and holding only a one-layer working window plus the
    /// embedding/output ends in RAM. Trades throughput for a strict memory ceiling.
    /// Double-buffered prefetch by default; `--sync-stream` forces the v1 serial read.
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
        /// Stage-split streaming (WRAITH Phase 2): run read and decode on separate threads so
        /// layer N+1's disk read overlaps layer N's dequant. Parity-identical; biggest win in
        /// the cold-NVMe regime where read and decode are comparable. Overrides the default
        /// single-worker prefetch. Mutually exclusive with `--sync-stream`.
        #[arg(long, default_value_t = false)]
        stage_split: bool,
        /// Stage-split read-ahead: how many layers the reader may run ahead of the decoder
        /// (raw-buffer pool = read_ahead + 1 layer spans; folds into the memory ceiling).
        #[arg(long, default_value_t = 2)]
        read_ahead: usize,
        /// Speculative decode (WRAITH Phase 3): draft L tokens with a resident zero-weight
        /// n-gram, verify all L+1 in ONE streamed sweep, accept the greedy-identical prefix.
        /// Lossless — accepted output is byte-identical to non-spec greedy. Amortizes the fixed
        /// per-layer disk read across the accepted tokens; biggest win on repetitive text.
        #[arg(long, default_value_t = false)]
        spec: bool,
        /// Speculative draft length L (n-gram tokens proposed per verify sweep). Capped at 7.
        #[arg(long, default_value_t = 5)]
        draft_len: usize,
        /// Strict memory ceiling mode: bypass the OS page cache for `.cghost` reads so
        /// streamed pages never accumulate (macOS F_NOCACHE; Windows FILE_FLAG_NO_BUFFERING).
        /// Leave off when the model fits in RAM and you want throughput (the cache is a free
        /// win there); turn ON to measure true cold-disk streaming cost even on a box where
        /// the model would otherwise cache.
        #[arg(long, default_value_t = false)]
        evict_page_cache: bool,
    },
    /// Verify one exact GGUF by replaying a built-in, reference-anchored,
    /// deterministic request. Emits a digest-sealed report. A pass proves one
    /// request for one exact file; it is not a broad support claim.
    Verify {
        /// GGUF model path. Verification abstains when no exact-hash profile exists.
        model: PathBuf,
        /// Report output path. Defaults to `<model-stem>.verify.json`.
        #[arg(long, value_name = "PATH")]
        output: Option<PathBuf>,
        /// Override Rayon worker threads for the deterministic replay.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Verify a receipt. For a parity receipt: self-digest, lane identity, an
    /// in-process Camelid re-run, and a llama.cpp reference re-run (requires
    /// `--gguf`). For a sealed agent-family receipt (syscap / orchestration /
    /// bench): a self-contained tamper-evidence + honest-scope check, no GGUF.
    /// A verified receipt changes no support claim.
    VerifyReceipt {
        /// Path to the receipt JSON file.
        receipt: PathBuf,
        /// The exact GGUF the receipt names (its SHA-256 must match). Required
        /// for a parity receipt; agent-family receipts need no GGUF.
        #[arg(long)]
        gguf: Option<PathBuf>,
        /// llama-server binary for the reference re-run (path or name in PATH).
        #[arg(long, default_value = "llama-server")]
        llama_server: String,
        /// Run only the self half (digest, lane identity, Camelid re-run);
        /// honest for verifiers without llama.cpp, but full parity is NOT
        /// asserted.
        #[arg(long, conflicts_with = "reference_only")]
        self_only: bool,
        /// Run only the reference half (digest, lane identity, llama.cpp
        /// re-run); skips the in-process Camelid re-run.
        #[arg(long)]
        reference_only: bool,
        /// Context size passed to llama-server (-c).
        #[arg(long, default_value_t = 2048)]
        llama_ctx: u32,
        /// Port for the temporary llama-server instance.
        #[arg(long, default_value_t = 8189)]
        llama_port: u16,
        /// Override Rayon worker threads for the Camelid re-run.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Audit the self-digest of every sealed receipt under a directory tree
    /// (parity / distributed / agent families). A corrupted or hand-edited
    /// committed receipt fails; unsealed and non-receipt JSON is skipped. This is
    /// the mechanical companion to review for the receipt evidence base.
    VerifyReceipts {
        /// Directory walked recursively for `*.json` receipts.
        #[arg(default_value = "qa")]
        dir: PathBuf,
    },
    /// Recompute and stamp `receipt_id` on a receipt body. Emitters (e.g. the
    /// chat-parity harness) delegate sealing here so canonical serialization
    /// and digesting live in exactly one implementation.
    #[command(hide = true)]
    SealReceipt {
        /// Receipt JSON to seal (the existing receipt_id value is ignored).
        #[arg(long, value_name = "PATH")]
        input: PathBuf,
        /// Output path; defaults to sealing in place.
        #[arg(long, value_name = "PATH")]
        output: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Both before ANY CUDA probe: the hook must be in place before cudarc's lazy
    // loaders can panic, or the first (caught, handled) miss still prints.
    quiet_cudarc_loader_panics();
    // Make the GPU runtime discoverable before anything probes for a device, so
    // the shipped app needs no PATH setup (no-op off Windows / without CUDA).
    ensure_cuda_runtime_on_path();
    pin_to_high_performance_gpu();

    // No subcommand (a bare double-click of the exe) launches the open-and-use app.
    let command = Cli::parse().command.unwrap_or_else(default_launch_command);

    // §4 safe-boot: a gait/substrate that crashed or wedged the host on the
    // previous run left an `.applying` marker; detect it now — before anything is
    // applied — quarantine that profile, and boot the proven baseline so a crash
    // can never loop. Inert unless the CAMELID_GAIT gate is on, so the default
    // path is byte-identical to today.
    if camelid::gait::gait_enabled() {
        if let Some(dir) = camelid::gait::gait_dir() {
            let _ = camelid::gait::sentinel::reconcile_on_startup(&dir);
        }
    }
    // Deterministic mode opts out of the GPU fast stack entirely; otherwise the CLI
    // defaults to the measured-fastest Metal configuration. Branch before any env is set
    // so the deterministic path never even arms the GPU defaults.
    if command_requests_deterministic(&command) {
        apply_deterministic_mode();
    } else {
        apply_default_fast_stack();
    }

    match command {
        Command::Serve {
            addr,
            model,
            threads,
            parallel_linear_min_outputs,
            apple_accelerate_min_elements,
            metal_linear,
            metal_q8,
            log_acceleration,
            spec_decode,
            spec_draft_model,
            spec_draft_tokens,
            no_open,
            deterministic,
            enable_thinking,
            models_dir,
            gpu,
            kv_quant,
        } => {
            std::env::set_var("CAMELID_KV_QUANT", kv_quant.to_string());
            configure_rayon_threads(threads)?;
            camelid::capability::HardwareProfile::detect().log();
            // In deterministic mode the engine fails every Metal gate closed (see
            // `apply_deterministic_mode`); don't also arm the Metal tuning env or the
            // GPU fast-load nocopy default, which would only be contradictory no-ops.
            apply_runtime_tuning_env(
                parallel_linear_min_outputs,
                apple_accelerate_min_elements,
                metal_linear && !deterministic,
                metal_q8 && !deterministic,
            );
            apply_spec_decode_env(spec_decode, spec_draft_model, spec_draft_tokens);
            if !deterministic {
                apply_serve_nocopy_default();
            }
            if log_acceleration {
                log_acceleration_state();
            }
            // Seed the runtime GPU switches from --gpu / CAMELID_GPU before any model
            // load, for headless/agent runs with no Settings UI to click. `auto` (default)
            // leaves the atomics uninitialised so they lazy-seed exactly as today
            // (gpu_accel from is_available(), hybrid Q8 from CAMELID_CUDA_Q8). `on`/`off`
            // are authoritative at startup and override the env seed; the UI
            // `POST /api/runtime/gpu` can still flip state live afterwards. Deterministic
            // mode and CAMELID_CUDA_RESIDENT_DECODE=0 still force the GPU off at their own
            // call sites regardless of this seed.
            match gpu {
                GpuMode::Auto => {}
                GpuMode::On => {
                    camelid::cuda::set_gpu_accel_enabled(true);
                    camelid::cuda::set_runtime_enabled(true);
                }
                GpuMode::Off => {
                    camelid::cuda::set_gpu_accel_enabled(false);
                    camelid::cuda::set_runtime_enabled(false);
                }
            }
            tracing::info!(
                gpu_mode = ?gpu,
                cuda_available = camelid::cuda::is_available(),
                gpu_enabled = camelid::cuda::gpu_accel_enabled(),
                "camelid gpu mode resolved"
            );
            #[cfg(target_os = "macos")]
            unsafe {
                pthread_set_qos_class_self_np(0x09, 0); // QOS_CLASS_BACKGROUND (forces network I/O onto E-cores)
            }
            // Open-and-use launch: if no model was named, load the first GGUF we
            // can find (beside the exe or in ./models) so the UI lands in a chat.
            let model = model.or_else(auto_select_model);
            // Open the browser only when run interactively and not opted out.
            let open_ui = !no_open && std::io::IsTerminal::is_terminal(&std::io::stdout());
            api::serve(addr, threads, model, open_ui, enable_thinking, models_dir).await?
        }
        Command::Chat {
            model,
            addr,
            system,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            no_stream,
            plain,
            models_dir,
            agent,
            workdir,
            max_steps,
            auto_approve,
            yolo,
            allow_net,
            allow_fs,
            allow_mcp,
            shell_timeout,
            enable_thinking,
            audit_webhook,
            shell_sandbox,
        } => {
            let code = chat::run_chat(chat::ChatOptions {
                model,
                addr,
                system,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                no_stream,
                plain,
                models_dir: models_dir.unwrap_or_else(|| PathBuf::from("models")),
                exec_goal: None,
                agent,
                workdir,
                max_steps,
                auto_approve,
                yolo,
                allow_net,
                allow_fs,
                allow_mcp,
                shell_timeout,
                enable_thinking,
                audit_webhook,
                shell_sandbox,
            })?;
            if code != 0 {
                std::process::exit(code);
            }
        }
        Command::AgentEval {
            model,
            addr,
            load_timeout,
            max_steps,
            max_tokens,
            receipt_dir,
        } => {
            let code = chat::run_agent_eval(chat::AgentEvalOptions {
                model,
                addr,
                load_timeout,
                max_steps,
                max_tokens,
                receipt_dir,
            })?;
            std::process::exit(code);
        }
        Command::AgentSyscapEval { receipt_dir } => {
            let code = chat::run_agent_syscap_eval(chat::AgentSyscapOptions { receipt_dir })?;
            std::process::exit(code);
        }
        Command::Subagent { task_file } => {
            let code = chat::run_subagent_worker(&task_file)?;
            std::process::exit(code);
        }
        Command::AgentOrchestrationEval {
            receipt_dir,
            model,
            addr,
            load_timeout,
        } => {
            let code = chat::run_agent_orchestration_eval(chat::AgentOrchestrationOptions {
                receipt_dir,
                model,
                addr,
                load_timeout,
            })?;
            std::process::exit(code);
        }
        Command::AgentOrchestrationBench {
            receipt_dir,
            model,
            addr,
            load_timeout,
        } => {
            let code = chat::run_agent_orchestration_bench(chat::AgentOrchestrationBenchOptions {
                receipt_dir,
                model,
                addr,
                load_timeout,
            })?;
            std::process::exit(code);
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
                api::serve(addr, threads, Some(model), false, false, None).await?
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
        Command::Tokenize {
            model,
            prompt,
            file,
            parse_special,
            no_add_special,
        } => {
            // Deep-recursion headroom (large vocab BPE build): dedicated big-stack
            // thread so the harness behaves identically in debug and release.
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || -> anyhow::Result<()> {
                    let gguf = read_metadata(model)?;
                    let tokenizer = Tokenizer::from_gguf(&gguf)?;
                    let inputs: Vec<String> = if let Some(p) = prompt {
                        vec![p]
                    } else if let Some(f) = file {
                        serde_json::from_str(&std::fs::read_to_string(f)?)?
                    } else {
                        anyhow::bail!("tokenize: provide --prompt or --file");
                    };
                    for text in &inputs {
                        let ids = tokenizer.encode(text, !no_add_special, parse_special)?;
                        let decoded = tokenizer.decode(&ids, false)?;
                        println!("{}", serde_json::json!({ "ids": ids, "decoded": decoded }));
                    }
                    Ok(())
                })?
                .join()
                .map_err(|_| anyhow::anyhow!("tokenize worker panicked"))??;
        }
        Command::RunnableSmoke { path } => {
            let path_str = path.to_string_lossy();
            match camelid::runnable::smoke_admit(&path_str) {
                Ok(report) => {
                    eprintln!(
                        "smoke-admission PASSED: {}/{}/{:?}",
                        report.architecture, report.quant, report.tokenizer
                    );
                    eprintln!(
                        "  prompt_tokens={} logits=[{:.1}, {:.1}]",
                        report.prompt_token_count, report.logit_min, report.logit_max
                    );
                    eprintln!("  greedy: {:?}", report.generated_text);
                    eprintln!(
                        "  (runnable receipt below — attests deterministic execution, not parity)"
                    );
                    // The runnable receipt (lane=runnable, never copper) to stdout.
                    println!("{}", serde_json::to_string_pretty(&report.receipt)?);
                }
                Err(err) => {
                    eprintln!("smoke-admission REFUSED/FAILED: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::PlanOffload {
            model,
            arch,
            budget_mb,
            context,
            safety_mb,
        } => {
            let profile = camelid::capability::HardwareProfile::detect();
            profile.log();
            let free_vram = match budget_mb {
                Some(mb) => {
                    println!("[offload] forced VRAM budget: {mb} MiB");
                    mb * 1024 * 1024
                }
                None => {
                    anyhow::ensure!(
                        profile.cuda_available,
                        "no CUDA device — offloading is a no-op; the CPU backend already \
                         holds all weights in system RAM"
                    );
                    profile.cuda_vram_free_bytes
                }
            };
            let context = context.unwrap_or(4096);
            let safety_mb = safety_mb.unwrap_or(256);
            let (config, plan) = if let Some(path) = model {
                let gguf = read_metadata(&path)?;
                let config = LlamaModelConfig::from_gguf(&gguf)?;
                let plan = camelid::offload::OffloadPlan::from_gguf(
                    &gguf, &config, free_vram, context, safety_mb,
                );
                (config, plan)
            } else if let Some(arch) = arch {
                let config = known_arch_config(&arch)?;
                let plan = camelid::offload::OffloadPlan::from_dims(
                    &config, free_vram, context, safety_mb,
                );
                (config, plan)
            } else {
                anyhow::bail!("provide a model path or --arch <name>");
            };
            let head_dim = config
                .attention_key_length
                .unwrap_or(config.embedding_length / config.attention_head_count.max(1));
            println!(
                "model: layers={} hidden={} ffn={} heads={} kv_heads={} head_dim={} vocab={:?} | KV reserved at context={}",
                config.block_count,
                config.embedding_length,
                config.feed_forward_length,
                config.attention_head_count,
                config.attention_head_count_kv,
                head_dim,
                config.vocab_size,
                context,
            );
            println!("{}", plan.describe());
            let map: String = plan
                .layer_resident
                .iter()
                .map(|&r| if r { 'V' } else { 'H' })
                .collect();
            println!("[offload] layer map (V=VRAM, H=host): {map}");
        }
        Command::Pull { model, models_dir } => {
            let dir = models_dir.unwrap_or_else(|| PathBuf::from("models"));
            camelid::catalog::run_pull(model.as_deref(), &dir)?;
        }
        Command::Gemma4Generate {
            path,
            prompt,
            max_tokens,
            force_tokens,
            dump_step_logits,
        } => {
            eprintln!("[gemma4] loading {}...", path.display());
            let t0 = std::time::Instant::now();
            let runtime = camelid::gemma4_runtime::Gemma4Runtime::load(&path)?;
            if force_tokens.is_none() && dump_step_logits.is_none() {
                // Default arm — byte-identical behavior to before the BASALT
                // harness flags existed.
                eprintln!(
                    "[gemma4] loaded in {:.1}s; generating {max_tokens} tokens...",
                    t0.elapsed().as_secs_f32()
                );
                let t1 = std::time::Instant::now();
                let (out, ids) = runtime.generate_greedy(&prompt, max_tokens)?;
                let gen = t1.elapsed().as_secs_f32();
                eprintln!(
                    "[gemma4] generated in {gen:.1}s ({:.2} tok/s)",
                    ids.len() as f32 / gen
                );
                eprintln!("[gemma4] token_ids: {ids:?}");
                println!("{prompt}{out}");
            } else {
                // BASALT Phase 3 harness surface (basalt_eval_protocol.md §5.1):
                // forced decode and/or per-step full-logit dumps. NO engine math
                // changes — both modes drive the same step loop as generate_greedy.
                eprintln!("[gemma4] loaded in {:.1}s", t0.elapsed().as_secs_f32());
                let forced: Option<Vec<u32>> = match &force_tokens {
                    Some(p) => {
                        let text = std::fs::read_to_string(p)?;
                        let ids = parse_forced_tokens(&text)
                            .map_err(|e| anyhow::anyhow!("--force-tokens {}: {e}", p.display()))?;
                        // Vocab bound known here (post-load): refuse out-of-range
                        // ids before any decode step runs.
                        validate_forced_token_vocab(&ids, runtime.vocab_size())
                            .map_err(|e| anyhow::anyhow!("--force-tokens {}: {e}", p.display()))?;
                        eprintln!(
                            "[gemma4] teacher-forcing {} tokens from {}",
                            ids.len(),
                            p.display()
                        );
                        Some(ids)
                    }
                    None => None,
                };
                let dump_dir = dump_step_logits.clone();
                if let Some(dir) = &dump_dir {
                    // Refuse mixing this run's step_<i>.bin dumps into a
                    // directory that already has contents.
                    ensure_dump_dir_empty(dir).map_err(|e| anyhow::anyhow!(e))?;
                    std::fs::create_dir_all(dir)?;
                }

                let mut records: Vec<Gemma4StepRecord> = Vec::new();
                let mut vocab_size = 0usize;
                let mut dump_err: Option<std::io::Error> = None;
                let t1 = std::time::Instant::now();

                let (mode, prompt_token_ids, greedy_out) = match &forced {
                    Some(ids) => {
                        let ptoks = runtime.forced_decode(&prompt, ids, |i, logits| {
                            vocab_size = logits.len();
                            let (argmax_id, argmax_logit) = greedy_argmax(logits);
                            records.push(Gemma4StepRecord {
                                step: i,
                                forced_id: Some(ids[i]),
                                argmax_id,
                                argmax_logit,
                                top32: top_n_logits(logits, 32),
                            });
                            if let Some(dir) = &dump_dir {
                                if dump_err.is_none() {
                                    if let Err(e) = write_step_logits(dir, i, logits) {
                                        dump_err = Some(e);
                                    }
                                }
                            }
                        })?;
                        ("forced", ptoks, None)
                    }
                    None => {
                        // --dump-step-logits alone: observed greedy decode
                        // (token-identical to generate_greedy on the plain loop).
                        let ptoks = runtime.tokenizer().encode(&prompt, true, true)?;
                        let (out, ids) = runtime.generate_greedy_observed(
                            &prompt,
                            max_tokens,
                            |i, logits| {
                                vocab_size = logits.len();
                                let (argmax_id, argmax_logit) = greedy_argmax(logits);
                                records.push(Gemma4StepRecord {
                                    step: i,
                                    forced_id: None,
                                    argmax_id,
                                    argmax_logit,
                                    top32: top_n_logits(logits, 32),
                                });
                                if let Some(dir) = &dump_dir {
                                    if dump_err.is_none() {
                                        if let Err(e) = write_step_logits(dir, i, logits) {
                                            dump_err = Some(e);
                                        }
                                    }
                                }
                            },
                        )?;
                        ("greedy", ptoks, Some((out, ids)))
                    }
                };
                if let Some(e) = dump_err {
                    return Err(anyhow::anyhow!(
                        "--dump-step-logits write failed in {}: {e}",
                        dump_dir
                            .as_ref()
                            .expect("dump dir set when dump_err set")
                            .display()
                    ));
                }
                let gen = t1.elapsed().as_secs_f32();
                eprintln!(
                    "[gemma4] {} steps in {gen:.1}s ({:.2} steps/s)",
                    records.len(),
                    records.len() as f32 / gen.max(1e-9)
                );

                let meta = Gemma4StepMeta {
                    protocol: "basalt_eval_protocol.md §5.1/§5.3",
                    mode,
                    model: path.display().to_string(),
                    prompt: prompt.clone(),
                    prompt_token_ids,
                    vocab_size,
                    step_count: records.len(),
                    logits_dtype: "f32_le",
                    logits_file_pattern: "step_<i>.bin",
                    steps: records,
                };
                if let Some(dir) = &dump_dir {
                    let meta_path = dir.join("meta.json");
                    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
                    eprintln!(
                        "[gemma4] wrote {} step_<i>.bin dumps + meta.json to {}",
                        meta.step_count,
                        dir.display()
                    );
                }
                match (mode, &greedy_out) {
                    // Forced mode: stdout is the machine-readable step record.
                    ("forced", _) => println!("{}", serde_json::to_string_pretty(&meta)?),
                    // Greedy+dump mode: stdout keeps the default arm's shape.
                    (_, Some((out, ids))) => {
                        eprintln!("[gemma4] token_ids: {ids:?}");
                        println!("{prompt}{out}");
                    }
                    _ => unreachable!("greedy mode always carries generate output"),
                }
            }
        }
        Command::Gemma4EvalPack {
            path,
            packs,
            baseline_dir,
            score,
        } => {
            #[derive(serde::Deserialize)]
            struct PackPrompt {
                id: String,
                text: String,
                #[serde(default)]
                max_new_tokens: usize,
            }
            #[derive(serde::Deserialize)]
            struct Pack {
                prompts: Vec<PackPrompt>,
            }
            let mut prompts: Vec<PackPrompt> = Vec::new();
            for p in &packs {
                let txt = std::fs::read_to_string(p)
                    .map_err(|e| anyhow::anyhow!("--pack {}: {e}", p.display()))?;
                let parsed: Pack = serde_json::from_str(&txt)
                    .map_err(|e| anyhow::anyhow!("--pack {}: {e}", p.display()))?;
                prompts.extend(parsed.prompts);
            }
            eprintln!(
                "[gemma4-eval] loading {} ({} prompts, mode={})...",
                path.display(),
                prompts.len(),
                if score { "score" } else { "baseline" }
            );
            let t0 = std::time::Instant::now();
            let runtime = camelid::gemma4_runtime::Gemma4Runtime::load(&path)?;
            eprintln!("[gemma4-eval] loaded in {:.1}s", t0.elapsed().as_secs_f32());
            if !score {
                std::fs::create_dir_all(&baseline_dir)?;
            }
            let mut total = 0usize;
            let mut agree = 0usize;
            let t1 = std::time::Instant::now();
            for pr in &prompts {
                let f = baseline_dir.join(format!("{}.txt", pr.id));
                if score {
                    let text = std::fs::read_to_string(&f)
                        .map_err(|e| anyhow::anyhow!("baseline {}: {e}", f.display()))?;
                    let ids = parse_forced_tokens(&text)
                        .map_err(|e| anyhow::anyhow!("baseline {}: {e}", f.display()))?;
                    validate_forced_token_vocab(&ids, runtime.vocab_size())
                        .map_err(|e| anyhow::anyhow!("baseline {}: {e}", f.display()))?;
                    let mut m = 0usize;
                    runtime.forced_decode(&pr.text, &ids, |i, logits| {
                        let (argmax_id, _) = greedy_argmax(logits);
                        if argmax_id == ids[i] {
                            m += 1;
                        }
                    })?;
                    total += ids.len();
                    agree += m;
                    eprintln!(
                        "[gemma4-eval]   {:<16} {:>3}/{:<3} = {:.1}%",
                        pr.id,
                        m,
                        ids.len(),
                        100.0 * m as f64 / ids.len().max(1) as f64
                    );
                } else {
                    let (_out, ids) = runtime.generate_greedy(&pr.text, pr.max_new_tokens)?;
                    let body = ids
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    std::fs::write(&f, body)?;
                    total += ids.len();
                    eprintln!(
                        "[gemma4-eval]   {:<16} {} tokens -> {}",
                        pr.id,
                        ids.len(),
                        f.display()
                    );
                }
            }
            let secs = t1.elapsed().as_secs_f32();
            if score {
                let pct = 100.0 * agree as f64 / total.max(1) as f64;
                eprintln!(
                    "[gemma4-eval] TEACHER-FORCED TOP-1 AGREEMENT: {agree}/{total} = {pct:.1}% ({secs:.1}s)"
                );
                println!(
                    "{{\"model\":{:?},\"agreement_pct\":{:.1},\"agree\":{},\"total\":{}}}",
                    path.display(),
                    pct,
                    agree,
                    total
                );
            } else {
                eprintln!(
                    "[gemma4-eval] baseline: {total} tokens across {} prompts ({secs:.1}s)",
                    prompts.len()
                );
                println!(
                    "{{\"model\":{:?},\"baseline_total\":{}}}",
                    path.display(),
                    total
                );
            }
        }
        #[cfg(feature = "cuda")]
        Command::Gemma4CudaGenerate {
            path,
            prompt,
            max_tokens,
        } => {
            eprintln!("[gemma4-cuda] loading resident {}...", path.display());
            let t0 = std::time::Instant::now();
            let mut runtime = camelid::gemma4_runtime::Gemma4CudaResident::load(&path, 4096)?;
            eprintln!(
                "[gemma4-cuda] resident loaded in {:.1}s; generating {max_tokens} tokens...",
                t0.elapsed().as_secs_f32()
            );
            let t1 = std::time::Instant::now();
            let (out, ids, per_token) = runtime.generate_greedy_timed(&prompt, max_tokens)?;
            let gen = t1.elapsed().as_secs_f32();
            eprintln!(
                "[gemma4-cuda] generated in {gen:.1}s ({:.2} tok/s overall incl. prefill)",
                ids.len() as f32 / gen
            );
            // Warm-up curve: mean tok/s over successive 8-token windows of decode-only
            // wall time (excludes prefill). With the SSER cache ON this should accelerate
            // as the hot experts populate VRAM.
            if !per_token.is_empty() {
                let win = 8usize;
                eprint!("[gemma4-cuda] warm-up curve (tok/s per {win}-tok window):");
                let mut i = 0;
                while i < per_token.len() {
                    let end = (i + win).min(per_token.len());
                    let secs: f64 = per_token[i..end].iter().sum();
                    let n = (end - i) as f64;
                    eprint!(
                        " [{}-{}]={:.2}",
                        i,
                        end - 1,
                        if secs > 0.0 { n / secs } else { 0.0 }
                    );
                    i = end;
                }
                eprintln!();
                // Steady-state: decode-only tok/s over the SECOND HALF of the tokens
                // (past warm-up) — the honest cache-warm rate.
                let half = per_token.len() / 2;
                let steady: f64 = per_token[half..].iter().sum();
                let sn = (per_token.len() - half) as f64;
                let decode_all: f64 = per_token.iter().sum();
                eprintln!(
                    "[gemma4-cuda] decode-only: {:.2} tok/s all, {:.2} tok/s steady (2nd half, {} tok)",
                    per_token.len() as f64 / decode_all.max(1e-9),
                    sn / steady.max(1e-9),
                    per_token.len() - half,
                );
            }
            if let Some((hits, misses, resident, cap)) = runtime.sser_stats() {
                let total = hits + misses;
                eprintln!(
                    "[gemma4-cuda] SSER cache: {hits} hits / {misses} misses = {:.1}% hit-rate; {resident}/{cap} experts resident",
                    if total > 0 { 100.0 * hits as f64 / total as f64 } else { 0.0 }
                );
            }
            eprintln!("[gemma4-cuda] token_ids: {ids:?}");
            println!("{prompt}{out}");
        }
        Command::Gemma4GenerateGpu {
            path,
            prompt,
            max_tokens,
        } => {
            #[cfg(target_os = "macos")]
            {
                let max_positions = 512.max(max_tokens + 64);
                eprintln!("[gemma4-gpu] loading {} (resident)...", path.display());
                let t0 = std::time::Instant::now();
                let runtime =
                    camelid::gemma4_runtime::Gemma4GpuRuntime::load(&path, max_positions)?;
                eprintln!(
                    "[gemma4-gpu] loaded in {:.1}s; generating {max_tokens} tokens...",
                    t0.elapsed().as_secs_f32()
                );
                let t1 = std::time::Instant::now();
                let (out, ids) = runtime.generate_greedy(&prompt, max_tokens)?;
                let gen = t1.elapsed().as_secs_f32();
                eprintln!(
                    "[gemma4-gpu] generated in {gen:.1}s ({:.2} tok/s)",
                    ids.len() as f32 / gen
                );
                eprintln!("[gemma4-gpu] token_ids: {ids:?}");
                println!("{prompt}{out}");
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (&path, &prompt, max_tokens);
                return Err(camelid::BackendError::UnsupportedModelArchitecture(
                    "gemma4 GPU runtime requires macOS/Metal".into(),
                )
                .into());
            }
        }
        Command::DiffusionGemmaChat {
            path,
            prompt,
            max_blocks,
            seed,
            max_ubatch,
            max_steps,
        } => {
            use camelid::diffusion_gemma::chat::DgChat;
            use camelid::diffusion_gemma::DgEbParams;
            eprintln!("[dg] loading {} (CPU, lazy mmap)...", path.display());
            let t0 = std::time::Instant::now();
            let chat = DgChat::load(&path)?;
            eprintln!(
                "[dg] loaded in {:.1}s; canvas_length={}; denoising (CPU — minutes per step)...",
                t0.elapsed().as_secs_f32(),
                chat.canvas_length()
            );
            let defaults = DgEbParams::default();
            let params = DgEbParams {
                seed,
                max_steps: max_steps.map(|m| m.max(1)).unwrap_or(defaults.max_steps),
                ..defaults
            };
            eprintln!(
                "[dg] max_steps={} max_blocks={}",
                params.max_steps, max_blocks
            );
            let t1 = std::time::Instant::now();
            // CAMELID_DG_LIVE=1: print the forming answer after every denoise
            // step — the whole draft exists from step 0 and refines in place.
            let live = std::env::var("CAMELID_DG_LIVE").as_deref() == Ok("1");
            let (text, stop, ids) = chat.generate_live(
                &prompt,
                &params,
                max_blocks,
                max_ubatch,
                |b, step, draft| {
                    if live {
                        let one_line = draft.replace('\n', " ");
                        let preview: String = one_line.chars().take(160).collect();
                        eprintln!(
                            "[dg-live b{b} s{step} {:.0}s] {preview}",
                            t1.elapsed().as_secs_f32()
                        );
                    }
                },
                |b, committed| {
                    eprintln!(
                        "[dg] block {b}: committed {} tokens ({:.0}s)",
                        committed.len(),
                        t1.elapsed().as_secs_f32()
                    );
                },
            )?;
            eprintln!(
                "[dg] done in {:.1}s (stop: {stop}, {} tokens)",
                t1.elapsed().as_secs_f32(),
                ids.len()
            );
            println!("{text}");
        }
        Command::Gemma4Worker {
            path,
            addr,
            first_layer,
        } => {
            // Blocks forever serving sessions; honest claim: distributed layer
            // sharding (memory headroom), not shared memory.
            let gguf = camelid::gguf::read_metadata(&path)?;
            let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
            let block_count = config.block_count as usize;
            camelid::gemma4_distributed::run_worker(&path, &addr, first_layer..block_count)?;
        }
        Command::Gemma4Master {
            path,
            worker_addr,
            split,
            prompt,
            max_tokens,
        } => {
            eprintln!(
                "[gemma4-master] layers 0..{split} local, {split}.. on {worker_addr}; loading..."
            );
            let t0 = std::time::Instant::now();
            let (out, ids, stats) = camelid::gemma4_distributed::run_master(
                &path,
                &worker_addr,
                split,
                &prompt,
                max_tokens,
                false,
            )?;
            eprintln!(
                "[gemma4-master] done in {:.1}s; stats: {}",
                t0.elapsed().as_secs_f32(),
                serde_json::to_string(&stats)?
            );
            eprintln!("[gemma4-master] token_ids: {ids:?}");
            println!("{prompt}{out}");
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
        #[cfg(feature = "alloc-gate")]
        Command::BenchAllocGate {
            model,
            warmup,
            tokens,
            skip_logits,
            trace_big,
            max_per_token,
        } => {
            let report = camelid::alloc_gate::run_decode_alloc_gate(
                &model,
                warmup,
                tokens,
                !skip_logits,
                trace_big,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if let Some(max_per_token) = max_per_token {
                let per_token = report["allocations_per_token"]
                    .as_f64()
                    .expect("report always carries allocations_per_token");
                if per_token > max_per_token {
                    anyhow::bail!(
                        "decode alloc gate FAILED: {per_token} allocations/token exceeds the \
                         allowed {max_per_token}"
                    );
                }
            }
        }
        Command::BenchRayonRegion {
            iterations,
            idle_us,
            threads,
        } => {
            configure_rayon_threads(threads)?;
            let us_per_region = camelid::inference::rayon_region_microbench(iterations, idle_us);
            let record = serde_json::json!({
                "schema": "camelid.bench-rayon-region/v1",
                "threads": rayon::current_num_threads(),
                "iterations": iterations,
                "idle_us_between": idle_us,
                "us_per_region": us_per_region,
            });
            println!("{}", serde_json::to_string(&record)?);
        }
        Command::BenchAttnDot {
            lens,
            repeats,
            warmup,
        } => {
            for len in lens {
                for (variant, ns_per_call) in
                    camelid::inference::attn_f32_dot_microbench(len, repeats, warmup)
                {
                    let record = serde_json::json!({
                        "schema": "camelid.bench-attn-dot/v1",
                        "len": len,
                        "variant": variant,
                        "ns_per_call": ns_per_call,
                    });
                    println!("{}", serde_json::to_string(&record)?);
                }
            }
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
            deterministic,
        } => {
            // Match `serve`'s fast-load plan so benchmark startup/RSS measures
            // the production Metal path rather than eagerly materializing Q8.
            if !deterministic {
                apply_serve_nocopy_default();
            }
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
        Command::BenchOwnerSweep {
            model,
            prompt_file,
            prompt,
            max_tokens,
            rounds,
            warmup_rounds,
            threads,
        } => {
            run_bench_owner_sweep(
                model,
                prompt_file,
                prompt,
                max_tokens,
                rounds,
                warmup_rounds,
                threads,
            )?;
        }
        Command::GaitCalibrate {
            model,
            prompt_file,
            prompt,
            max_tokens,
            rounds,
            warmup,
            threads,
        } => {
            run_gait_calibrate(
                model,
                prompt_file,
                prompt,
                max_tokens,
                rounds,
                warmup,
                threads,
            )?;
        }
        Command::GaitTrial {
            model,
            prompt_file,
            prompt,
            max_tokens,
            profile,
            eco_qos,
            threads,
            gpc_attn,
            gpc_ffn,
            gpc_matmul,
        } => {
            run_gait_trial(
                model,
                prompt_file,
                prompt,
                max_tokens,
                profile,
                eco_qos,
                threads,
                gpc_attn,
                gpc_ffn,
                gpc_matmul,
            )?;
        }
        Command::Workspace {
            addr,
            json,
            timeout_seconds,
            action,
        } => {
            let action = match action {
                WorkspaceAction::Ask {
                    workspace,
                    goal,
                    thread,
                    max_steps,
                    max_tokens,
                    temperature,
                } => chat::WorkspaceCliAction::Ask {
                    workspace,
                    goal,
                    thread_id: thread,
                    max_steps,
                    max_tokens,
                    temperature,
                },
                WorkspaceAction::Threads { workspace } => {
                    chat::WorkspaceCliAction::Threads { workspace }
                }
                WorkspaceAction::Show { thread, workspace } => chat::WorkspaceCliAction::Show {
                    workspace,
                    thread_id: thread,
                },
                WorkspaceAction::Compact {
                    thread,
                    workspace,
                    undo,
                } => chat::WorkspaceCliAction::Compact {
                    workspace,
                    thread_id: thread,
                    undo,
                },
                WorkspaceAction::Delete { thread, workspace } => chat::WorkspaceCliAction::Delete {
                    workspace,
                    thread_id: thread,
                },
            };
            let code = chat::run_workspace_cli(chat::WorkspaceCliOptions {
                addr,
                json,
                timeout: std::time::Duration::from_secs(timeout_seconds),
                action,
            })?;
            std::process::exit(code);
        }
        Command::Agent { action } => match action {
            AgentAction::Exec {
                goal,
                model,
                addr,
                workdir,
                max_steps,
                max_tokens,
                auto_approve,
                yolo,
                allow_net,
                allow_fs,
                allow_mcp,
                shell_sandbox,
                shell_timeout,
                models_dir,
            } => {
                // The goal may come from stdin so a caller can pipe a long or
                // generated prompt in without shell quoting.
                let goal = match goal {
                    Some(g) => g,
                    None => {
                        let mut buf = String::new();
                        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
                        buf
                    }
                };
                if goal.trim().is_empty() {
                    eprintln!("agent exec needs a goal (as an argument or on stdin)");
                    std::process::exit(1);
                }
                let code = chat::run_chat(chat::ChatOptions {
                    model: Some(model),
                    addr,
                    system: None,
                    max_tokens,
                    temperature: 0.0,
                    top_p: None,
                    top_k: None,
                    seed: None,
                    no_stream: true,
                    models_dir: models_dir.unwrap_or_else(|| PathBuf::from("models")),
                    plain: true,
                    agent: true,
                    workdir,
                    max_steps,
                    auto_approve,
                    yolo,
                    allow_net,
                    allow_fs,
                    allow_mcp,
                    shell_timeout,
                    enable_thinking: false,
                    audit_webhook: None,
                    shell_sandbox,
                    exec_goal: Some(goal),
                })?;
                std::process::exit(code);
            }
        },
        Command::Gait { action } => match action {
            GaitAction::Reset => run_gait_reset()?,
        },
        Command::BenchSpeculative {
            model,
            drafter,
            draft_model,
            draft_tokens,
            cpu_draft,
            spec_only,
            prompt_file,
            prompt,
            workload,
            max_tokens,
            warmup,
            threads,
        } => {
            run_bench_speculative(
                model,
                drafter,
                draft_model,
                draft_tokens,
                cpu_draft,
                spec_only,
                prompt_file,
                prompt,
                workload,
                max_tokens,
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
            stage_split,
            read_ahead,
            spec,
            draft_len,
            evict_page_cache,
        } => {
            run_ghost(
                model,
                cghost,
                prompt,
                max_tokens,
                threads,
                sync_stream,
                stage_split,
                read_ahead,
                spec,
                draft_len,
                evict_page_cache,
            )?;
        }
        Command::Verify {
            model,
            output,
            threads,
        } => {
            configure_rayon_threads(threads)?;
            let report = camelid::verify::run(&model, threads)
                .await
                .map_err(anyhow::Error::msg)?;
            let output = output.unwrap_or_else(|| camelid::verify::default_report_path(&model));
            camelid::verify::write_report(&output, &report).map_err(anyhow::Error::msg)?;
            println!(
                "{} model={} report_id={} output={}",
                match report.outcome {
                    camelid::verify::VerificationOutcome::Verified => "VERIFIED",
                    camelid::verify::VerificationOutcome::NotVerified => "NOT VERIFIED",
                    camelid::verify::VerificationOutcome::NoProfile => "NO PROFILE",
                },
                report.model.gguf_filename,
                report.report_id,
                output.display()
            );
            println!("{}", report.detail);
            std::process::exit(report.outcome.exit_code());
        }
        Command::VerifyReceipt {
            receipt,
            gguf,
            llama_server,
            self_only,
            reference_only,
            llama_ctx,
            llama_port,
            threads,
        } => {
            // Route a sealed agent-family receipt to its self-contained verifier
            // (no model, no GGUF). Any doubt about the schema falls through to the
            // parity path below, which is left unchanged.
            let is_agent = std::fs::read_to_string(&receipt)
                .ok()
                .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
                .and_then(|value| {
                    value
                        .get("schema")
                        .and_then(|schema| schema.as_str())
                        .map(camelid::receipt::agent::is_agent_schema)
                })
                .unwrap_or(false);
            if is_agent {
                let outcome = camelid::receipt::agent::run(&receipt);
                std::process::exit(outcome.exit_code());
            }

            configure_rayon_threads(threads)?;
            let gguf = gguf.ok_or_else(|| {
                anyhow::anyhow!(
                    "parity verification requires the exact GGUF via --gguf; agent-family \
                     receipts (syscap / orchestration / bench) need no GGUF"
                )
            })?;
            let mode = if self_only {
                camelid::receipt::verify::VerifyMode::SelfOnly
            } else if reference_only {
                camelid::receipt::verify::VerifyMode::ReferenceOnly
            } else {
                camelid::receipt::verify::VerifyMode::Full
            };
            let outcome = camelid::receipt::verify::run(camelid::receipt::verify::VerifyOptions {
                receipt_path: receipt,
                gguf,
                llama_server,
                mode,
                llama_ctx,
                llama_port,
                threads,
            })
            .await;
            std::process::exit(outcome.exit_code());
        }
        Command::VerifyReceipts { dir } => {
            std::process::exit(camelid::receipt::audit::run(&dir));
        }
        Command::SealReceipt { input, output } => {
            let raw = std::fs::read_to_string(&input)?;
            let mut receipt: camelid::receipt::ParityReceipt = serde_json::from_str(&raw)?;
            anyhow::ensure!(
                receipt.schema == camelid::receipt::RECEIPT_SCHEMA_V1,
                "unknown receipt schema {:?} (expected {:?})",
                receipt.schema,
                camelid::receipt::RECEIPT_SCHEMA_V1
            );
            receipt.seal()?;
            let out_path = output.unwrap_or(input);
            let mut serialized = serde_json::to_string_pretty(&receipt)?;
            serialized.push('\n');
            std::fs::write(&out_path, serialized)?;
            println!(
                "sealed receipt_id={} -> {}",
                receipt.receipt_id,
                out_path.display()
            );
        }
    }

    // §4 safe-boot: an orderly exit — clear any in-progress gait marker so a
    // healthy run is never mistaken for a crash on the next launch. (No-op unless
    // a gait was applied this process.)
    if let Some(dir) = camelid::gait::gait_dir() {
        camelid::gait::sentinel::clean_shutdown(&dir);
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
    /// v3 stage-split (`--stage-split`): read and decode run on SEPARATE threads, so the read
    /// of layer N+1 overlaps the dequant of layer N (v2's single worker serializes them).
    Pipelined { prefetcher: GhostPipelinePrefetcher },
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

    fn new_pipelined(
        ghost: Arc<GhostFile>,
        range: std::ops::Range<usize>,
        read_ahead: usize,
    ) -> Self {
        Self {
            range,
            kind: GhostStreamerKind::Pipelined {
                prefetcher: GhostPipelinePrefetcher::spawn(ghost, read_ahead),
            },
        }
    }

    /// Queue the first chunk's layer reads (prefetched / stage-split modes; no-op for sync).
    fn prime(&self) -> anyhow::Result<()> {
        match &self.kind {
            GhostStreamerKind::Prefetched { prefetcher } => {
                for layer_idx in self.range.clone() {
                    prefetcher.request(layer_idx)?;
                }
            }
            GhostStreamerKind::Pipelined { prefetcher } => {
                for layer_idx in self.range.clone() {
                    prefetcher.request(layer_idx)?;
                }
            }
            GhostStreamerKind::Sync { .. } => {}
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
    /// Returns `(weights, bytes, blocked_us, read_us, decode_us)`. `blocked_us` is the stall
    /// charged to the streaming path (the whole critical-path read+decode in sync mode; only
    /// the handoff wait in prefetched mode). `read_us`/`decode_us` are the worker's actual
    /// I/O-vs-dequant split (Phase-0 attribution), independent of how much of it overlapped.
    fn fetch(
        &mut self,
        layer_idx: usize,
        last_in_chunk: bool,
    ) -> anyhow::Result<(LlamaLayerWeights, u64, u128, u128, u128)> {
        let range = self.range.clone();
        match &mut self.kind {
            GhostStreamerKind::Sync { ghost, buf } => {
                let started = Instant::now();
                let (layer, span, read_us, decode_us) = ghost.read_layer(layer_idx, buf)?;
                Ok((
                    layer,
                    span,
                    started.elapsed().as_micros(),
                    read_us,
                    decode_us,
                ))
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
                    prefetched.read_us,
                    prefetched.decode_us,
                ))
            }
            GhostStreamerKind::Pipelined { prefetcher } => {
                if last_in_chunk {
                    for next_idx in range {
                        prefetcher.request(next_idx)?;
                    }
                }
                let started = Instant::now();
                let prefetched = prefetcher.next()?;
                anyhow::ensure!(
                    prefetched.layer_idx == layer_idx,
                    "stage-split returned layer {} but layer {layer_idx} was expected",
                    prefetched.layer_idx
                );
                Ok((
                    prefetched.weights,
                    prefetched.bytes,
                    started.elapsed().as_micros(),
                    prefetched.read_us,
                    prefetched.decode_us,
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
) -> anyhow::Result<(CpuTensor, u64, u128, u128, u128, u128)> {
    let range = streamer.range.clone();
    let mut hidden = hidden;
    let mut bytes_total = 0u64;
    let mut wait_us_total = 0u128;
    let mut forward_us_total = 0u128;
    let mut read_us_total = 0u128;
    let mut decode_us_total = 0u128;
    for layer_idx in range.clone() {
        let (layer, span, wait_us, read_us, decode_us) =
            streamer.fetch(layer_idx, layer_idx + 1 == range.end)?;
        Arc::make_mut(&mut session.weights).layers[layer_idx] = layer;
        let forward_started = Instant::now();
        hidden = session.ghost_forward_one_layer(&hidden, layer_idx, pos, seq_len)?;
        let forward_us = forward_started.elapsed().as_micros();
        // Drop the streamed weights immediately; the window never accumulates.
        Arc::make_mut(&mut session.weights).layers[layer_idx] = placeholder.clone();
        bytes_total += span;
        wait_us_total += wait_us;
        forward_us_total += forward_us;
        read_us_total += read_us;
        decode_us_total += decode_us;
        if log_layers {
            // read/decode is the worker's true I/O-vs-dequant split; wait is the main
            // thread's stall (only the unhidden remainder after prefetch overlap).
            eprintln!(
                "[ghost] layer {layer_idx:>3}: wait {:7.1} ms | read {:7.1} decode {:7.1} \
                 ({:6.1} MiB) | forward {:7.1} ms",
                wait_us as f64 / 1000.0,
                read_us as f64 / 1000.0,
                decode_us as f64 / 1000.0,
                span as f64 / (1024.0 * 1024.0),
                forward_us as f64 / 1000.0,
            );
        }
    }
    session.ghost_advance_position(seq_len);
    Ok((
        hidden,
        bytes_total,
        wait_us_total,
        forward_us_total,
        read_us_total,
        decode_us_total,
    ))
}

/// EXPERIMENTAL ghost (layer-streaming) mode: greedy generation with the model executed one
/// transformer block at a time from a `.cghost` file. RAM holds the embedding/output ends +
/// KV cache + the streaming window (one layer sync, two prefetched); everything else stays
/// on disk.
#[allow(clippy::too_many_arguments)]
fn run_ghost(
    model: PathBuf,
    cghost: PathBuf,
    prompt: String,
    max_tokens: usize,
    threads: Option<usize>,
    sync_stream: bool,
    stage_split: bool,
    read_ahead: usize,
    spec: bool,
    draft_len: usize,
    evict_page_cache: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !(sync_stream && stage_split),
        "--sync-stream and --stage-split are mutually exclusive"
    );
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
    } else if stage_split {
        GhostStreamer::new_pipelined(Arc::clone(&ghost), 0..n_layers, read_ahead)
    } else {
        GhostStreamer::new_prefetched(Arc::clone(&ghost), 0..n_layers)
    };
    let mode_label = if sync_stream {
        "sync".to_string()
    } else if stage_split {
        format!("stage-split (read\u{2016}decode, read-ahead {read_ahead})")
    } else {
        "double-buffered prefetch".to_string()
    };
    println!(
        "[ghost] resident ends loaded in {:.1}s; {} layers x {:.1} MiB max streaming window \
         ({}, page cache {}); footprint {:.2} GiB",
        load_started.elapsed().as_secs_f64(),
        n_layers,
        ghost.max_layer_span() as f64 / (1024.0 * 1024.0),
        mode_label,
        if evict_page_cache { "bypassed" } else { "on" },
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
    let (mut hidden, bytes, wait_us, forward_us, read_us, dec_us) = ghost_stream_layers(
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
        "[ghost] prefill: {:.1}s ({:.2} GiB streamed, blocked {:.1}s | read {:.1}s decode \
         {:.1}s | forward {:.1}s); footprint {:.2} GiB",
        prefill_started.elapsed().as_secs_f64(),
        gib(bytes),
        wait_us as f64 / 1_000_000.0,
        read_us as f64 / 1_000_000.0,
        dec_us as f64 / 1_000_000.0,
        forward_us as f64 / 1_000_000.0,
        gib(phys_footprint_bytes()),
    );

    if spec {
        ghost_spec_decode(
            &mut session,
            &mut streamer,
            &placeholder,
            &tokenizer,
            hidden,
            &token_ids,
            max_tokens,
            draft_len,
        )?;
    } else {
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
            let (next_hidden, bytes, wait_us, forward_us, read_us, dec_us) = ghost_stream_layers(
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
                "[ghost] token {:>3}: {:6.0} ms ({:.2} GiB streamed, blocked {:5.0} ms | read \
                 {:5.0} decode {:5.0} | forward {:5.0} ms)",
                step + 1,
                token_us as f64 / 1000.0,
                gib(bytes),
                wait_us as f64 / 1000.0,
                read_us as f64 / 1000.0,
                dec_us as f64 / 1000.0,
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
    }
    println!(
        "[ghost] final footprint {:.2} GiB, peak RSS {:.2} GiB",
        gib(phys_footprint_bytes()),
        gib(peak_rss_bytes()),
    );
    Ok(())
}

/// WRAITH Phase-3 speculative ghost decode. Draft L tokens with a resident zero-weight n-gram,
/// verify `[anchor, draft_1..draft_L]` in ONE streamed sweep (each layer read once, applied to
/// all L+1 positions), accept the greedy-identical prefix, then roll the KV cache position back
/// over the rejected tail. Because a single causal `position` cursor bounds every attention
/// read, that rollback alone makes the rejected slots unreadable — no buffer truncation — so the
/// accepted stream is byte-identical to non-spec ghost greedy. A ghost sweep's cost is dominated
/// by the fixed per-layer disk read, so amortizing it across `1 + accepted` committed tokens is
/// the win. An EMA auto-disable drops drafting to single-token sweeps when acceptance collapses
/// (novel text) and re-probes periodically, so spec never badly regresses a non-repetitive load.
#[allow(clippy::too_many_arguments)]
fn ghost_spec_decode(
    session: &mut LlamaInferenceSession,
    streamer: &mut GhostStreamer,
    placeholder: &LlamaLayerWeights,
    tokenizer: &Tokenizer,
    prefill_hidden: CpuTensor,
    prompt_ids: &[u32],
    max_tokens: usize,
    draft_len: usize,
) -> anyhow::Result<()> {
    use camelid::inference::speculative::{accepted_draft_prefix, NGramDrafter};

    let draft_len = draft_len.min(7); // verify-batch width cap (MAX_VERIFY_K - 1)
    let drafter = NGramDrafter::default();
    let mut history: Vec<u32> = prompt_ids.to_vec();
    let is_stop = |t: u32| tokenizer.special.eos == Some(t) || tokenizer.special.eot == Some(t);

    // Per-row greedy argmax via the SAME sampler the non-spec path uses, so tie-breaking (and
    // therefore the accepted token stream) is identical.
    let greedy_rows = |logits: &CpuTensor| -> anyhow::Result<Vec<u32>> {
        let rows = logits.dim(0)?;
        let vocab = logits.dim(1)?;
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let start = r * vocab;
            let row = CpuTensor::from_f32(
                "ghost_spec_logits",
                vec![1, vocab],
                logits.data[start..start + vocab].to_vec(),
            )?;
            out.push(LlamaSampler::Greedy.sample(&row)?);
        }
        Ok(out)
    };

    // The first token comes free from the prefill hidden (no sweep), exactly as non-spec does:
    // argmax the LAST prefill row only (not all N prompt rows).
    let ttft = {
        let logits = session.forward_final_norm_and_logits(&prefill_hidden)?;
        let vocab = logits.dim(1)?;
        let rows = logits.dim(0)?;
        let last = (rows - 1) * vocab;
        let row = CpuTensor::from_f32(
            "ghost_spec_ttft",
            vec![1, vocab],
            logits.data[last..last + vocab].to_vec(),
        )?;
        LlamaSampler::Greedy.sample(&row)?
    };
    let mut generated: Vec<u32> = vec![ttft];
    print!("{}", tokenizer.decode(&[ttft], true)?);
    std::io::stdout().flush()?;
    history.push(ttft);
    let mut current = ttft;

    let decode_started = Instant::now();
    let mut sweeps = 0usize;
    let mut drafted_total = 0usize;
    let mut accepted_total = 0usize;
    // Rounds where at least one drafted token was REJECTED — i.e. the KV rollback discarded a
    // non-empty rejected tail. If rejected KV leaked, parity vs non-spec would break; a run with
    // rejected_rounds > 0 that stays byte-identical is the rejected-KV isolation proof.
    let mut rejected_rounds = 0usize;
    let mut ema_accepted = draft_len as f64; // optimistic start; drives auto-disable
    let mut since_probe = 0usize;
    let mut auto_disabled_ever = false;

    'outer: while generated.len() < max_tokens && !is_stop(current) {
        // Auto-disable: when recent acceptance collapses, stop drafting (single-token sweeps)
        // and re-probe every 64 rounds so a return to repetitive text is picked back up.
        let drafting_on = ema_accepted >= 0.5 || since_probe >= 64;
        if drafting_on {
            since_probe = 0;
        } else {
            since_probe += 1;
            auto_disabled_ever = true;
        }
        let room = max_tokens - generated.len();
        let budget = if drafting_on {
            draft_len.min(room.saturating_sub(1))
        } else {
            0
        };
        let drafts = if budget > 0 {
            drafter.draft(&history, budget)
        } else {
            Vec::new()
        };
        drafted_total += drafts.len();

        // Verify batch = [anchor, draft_1..draft_L]; ONE streamed sweep over all layers writes
        // KV for positions [base, base+len) and yields per-position greedy predictions.
        let base = session.kv_position();
        let mut batch = Vec::with_capacity(1 + drafts.len());
        batch.push(current);
        batch.extend_from_slice(&drafts);
        let embedding = session
            .weights
            .token_embedding
            .embedding_lookup(&batch, "token_embedding_ghost_spec")?;
        let (rows_hidden, _bytes, _wait, _fwd, _read, _dec) = ghost_stream_layers(
            session,
            streamer,
            placeholder,
            embedding,
            base,
            batch.len(),
            false,
        )?;
        sweeps += 1;
        let logits = session.forward_final_norm_and_logits(&rows_hidden)?;
        let predictions = greedy_rows(&logits)?;
        let accepted = accepted_draft_prefix(&drafts, &predictions);
        accepted_total += accepted;
        if accepted < drafts.len() {
            rejected_rounds += 1; // a non-empty rejected tail was rolled back this round
        }
        ema_accepted = 0.85 * ema_accepted + 0.15 * accepted as f64;

        // Commit the anchor (position `base`) + `accepted` drafts; roll the position back over
        // the rest so rejected KV at [base+1+accepted .. base+len) is causally unreadable.
        session.rollback_to_position(base + 1 + accepted)?;

        // Emit predictions[0..=accepted] = the accepted drafts plus one correction token.
        for &token in &predictions[..=accepted] {
            if generated.len() >= max_tokens {
                break;
            }
            generated.push(token);
            history.push(token);
            print!("{}", tokenizer.decode(&[token], true)?);
            std::io::stdout().flush()?;
            current = token;
            if is_stop(token) {
                break 'outer;
            }
        }
    }
    println!();

    let secs = decode_started.elapsed().as_secs_f64();
    let decoded = generated.len().saturating_sub(1); // the TTFT was free
    let accept_rate = if drafted_total > 0 {
        accepted_total as f64 / drafted_total as f64
    } else {
        0.0
    };
    println!(
        "[ghost] spec decode: {decoded} tokens in {secs:.1}s = {:.3} tok/s | {sweeps} sweeps \
         ({:.2} tok/sweep) | draft_len {draft_len}, drafted {drafted_total}, accepted \
         {accepted_total} ({:.0}%), rejected-tail rounds {rejected_rounds}, mean {:.2}/round, \
         auto-disable {}",
        if secs > 0.0 {
            decoded as f64 / secs
        } else {
            0.0
        },
        if sweeps > 0 {
            decoded as f64 / sweeps as f64
        } else {
            0.0
        },
        accept_rate * 100.0,
        if sweeps > 0 {
            accepted_total as f64 / sweeps as f64
        } else {
            0.0
        },
        if auto_disabled_ever {
            "fired"
        } else {
            "not fired"
        },
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
    /// GPU layer-offload split for this run (Phase 4 honest labeling). `None` on the
    /// CPU path; `source == "none"` means fully resident on the GPU. A non-zero
    /// `layers_offloaded` means a tok/s number here is a capacity-mode result, not a
    /// fully-resident one — the field carries the split + measured PCIe so it can't be
    /// read as native.
    #[serde(skip_serializing_if = "Option::is_none")]
    offload: Option<camelid::offload::OffloadRunStatus>,
    output_text: String,
    output_token_ids: Vec<u32>,
}

/// §5: set (or clear) the three managed x86 Q8 `groups_per_chunk` env knobs for a
/// trial. `None` clears them so the trial measures the profile's default tiling;
/// `Some` pins the search candidate's values. Must be called AFTER
/// `PlannerEnv::apply` (these keys are managed) so the override is authoritative.
fn apply_groups_per_chunk(gpc: Option<camelid::gait::calibrate::GroupsPerChunk>) {
    const ATTN: &str = "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK";
    const FFN: &str = "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK";
    const MATMUL: &str = "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK";
    match gpc {
        Some(g) => {
            std::env::set_var(ATTN, g.attn_qkv_decode.to_string());
            std::env::set_var(FFN, g.ffn_gate_up_decode.to_string());
            std::env::set_var(MATMUL, g.packed_rows4_matmul.to_string());
        }
        None => {
            std::env::remove_var(ATTN);
            std::env::remove_var(FFN);
            std::env::remove_var(MATMUL);
        }
    }
}

fn gait_profile_env_value(profile: &camelid::execution_plan::ExecutionProfile) -> &'static str {
    use camelid::execution_plan::ExecutionProfile::*;
    match profile {
        Auto => "auto",
        Safe => "safe",
        Experimental => "experimental",
        Debug => "debug",
    }
}

/// Time one candidate: select its profile for the planner, reload weights (the
/// Q8 repack / kernel choice happens at load time, so each candidate needs its
/// own load), run one unmeasured warmup, then a measured greedy decode. The
/// parity token is the SHA-256 of the greedy output token ids — a candidate that
/// changes the output is disqualified by the tournament.
fn gait_profile_trial(
    model: &std::path::Path,
    threads: Option<usize>,
    prompt_token_ids: &[u32],
    max_tokens: usize,
    candidate: &camelid::gait::calibrate::Candidate,
) -> anyhow::Result<camelid::gait::calibrate::TrialResult> {
    std::env::set_var(
        "CAMELID_PROFILE",
        gait_profile_env_value(&candidate.profile),
    );
    // Apply this candidate's Windows scheduling substrate before timing, so the
    // measured decode reflects it. §1.2-scoped to the compute pool (the Rayon
    // workers + this thread), matching what production applies.
    let eco_status = camelid::gait::substrate::set_compute_pool_eco_qos(candidate.eco_qos_opt_out);
    if candidate.eco_qos_opt_out && eco_status != camelid::gait::substrate::EcoQosStatus::OptedOut {
        eprintln!(
            "[gait]   {} eco_qos opt-out unavailable -> {eco_status:?}",
            candidate.label
        );
    }

    let gguf = read_metadata(model)?;
    // Apply this candidate's plan before loading weights, exactly as bench-generate does.
    let plan = camelid::execution_plan::plan_for_model(model, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan.env_updates);
    // §5: apply this candidate's groups_per_chunk tiling AFTER the planner's
    // env_updates — the gpc knobs are MANAGED_ENV_KEYS, so PlannerEnv::apply would
    // otherwise clear/overwrite them. Applying here lets the search override win.
    apply_groups_per_chunk(candidate.groups_per_chunk);

    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);
    let sampler = LlamaSampler::Greedy;

    let _ = generate_run(
        &config,
        &weights,
        &tokenizer,
        prompt_token_ids,
        &sampler,
        max_tokens,
    )?;
    camelid::inference::reset_stage_timings();
    let run = generate_run(
        &config,
        &weights,
        &tokenizer,
        prompt_token_ids,
        &sampler,
        max_tokens,
    )?;

    let decode_tokens = run.generated.len().saturating_sub(1);
    let tokens_per_s = if run.decode_ms > 0.0 && decode_tokens > 0 {
        decode_tokens as f64 / (run.decode_ms / 1000.0)
    } else {
        0.0
    };
    let mut id_bytes = Vec::with_capacity(run.generated.len() * 4);
    for id in &run.generated {
        id_bytes.extend_from_slice(&id.to_le_bytes());
    }
    let parity_token = camelid::receipt::sha256_hex(&id_bytes);
    Ok(camelid::gait::calibrate::TrialResult {
        tokens_per_s,
        parity_token,
    })
}

fn run_gait_calibrate(
    model: PathBuf,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    max_tokens: usize,
    rounds: usize,
    warmup: usize,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    use camelid::execution_plan::ExecutionProfile;
    use camelid::gait::calibrate::{
        calibrate_and_store, default_store_dir, Candidate, TournamentConfig,
    };

    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    anyhow::ensure!(rounds >= 1, "--rounds must be at least 1");
    // Calibration must measure the candidates we choose — never let a previously
    // cached gait receipt override the candidate's profile mid-trial.
    std::env::remove_var("CAMELID_GAIT");
    // §1.2: calibrate under the same core-reserve cap production will run with, so
    // the measured tok/s reflects the host-safe thread budget. The gate is off
    // here, so configure_rayon_threads won't apply the cap itself.
    configure_rayon_threads(host_safe_thread_count(threads))?;
    camelid::capability::HardwareProfile::detect().log();

    anyhow::ensure!(
        prompt_file.is_some() || prompt.is_some(),
        "provide --prompt-file <path> or --prompt <text>"
    );

    // The gguf is needed for the fingerprint, memory measurement, and roofline
    // numerator; each candidate's prompt encoding + decode happens in its own
    // child trial (§1.4 crash isolation), so the parent does not load weights.
    let gguf = read_metadata(&model)?;

    // Baseline = today's behavior (Auto profile, OS-managed throttling). The
    // candidates vary the EcoQoS substrate (and profile) so the tournament
    // measures whether disabling throttling helps on this machine, parity-held.
    let baseline = Candidate {
        label: "auto".to_string(),
        profile: ExecutionProfile::Auto,
        eco_qos_opt_out: false,
        groups_per_chunk: None,
    };
    // §5 bounded local search: the EcoQoS substrate dimension (auto+ecoqos) plus
    // the experimental kernel under each groups_per_chunk neighbor. Every
    // candidate is parity-gated + crash-isolated; the tournament fails closed to
    // baseline if none beats it by margin (the honest outcome on a
    // memory-bandwidth-bound box, where the tiling knob is expected to be flat).
    let mut candidates = vec![Candidate {
        label: "auto+ecoqos".to_string(),
        profile: ExecutionProfile::Auto,
        eco_qos_opt_out: true,
        groups_per_chunk: None,
    }];
    for gpc in camelid::gait::calibrate::groups_per_chunk_neighbors() {
        candidates.push(Candidate {
            label: format!(
                "exp+gpc[{},{},{}]",
                gpc.attn_qkv_decode, gpc.ffn_gate_up_decode, gpc.packed_rows4_matmul
            ),
            profile: ExecutionProfile::Experimental,
            eco_qos_opt_out: false,
            groups_per_chunk: Some(gpc),
        });
    }

    let store_dir = default_store_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve the gait store directory"))?;

    let config = TournamentConfig {
        rounds,
        warmup_rounds: warmup,
        ..TournamentConfig::default()
    };
    eprintln!(
        "[gait] calibrating {} candidates (+baseline), {} measured rounds (+{} warmup), interleaved, on {} ...",
        candidates.len(),
        config.rounds,
        config.warmup_rounds,
        model.display()
    );
    // §1.1 host-safety: do not launch the calibration allocation campaign (each
    // child trial loads multi-GB weights) if the box is already low on free RAM.
    #[cfg(windows)]
    if let Some((total, avail)) = camelid::gait::host_ram_status() {
        if !camelid::gait::ram_headroom_ok(total, avail) {
            let floor = camelid::gait::ram_headroom_floor(total);
            eprintln!(
                "[gait] insufficient free RAM (avail {:.1} GiB < floor {:.1} GiB) -> skipping calibration; baseline serves",
                avail as f64 / 1e9,
                floor as f64 / 1e9
            );
            return Ok(());
        }
    }

    // §1.4 crash isolation: each candidate runs in a supervised CHILD PROCESS, so
    // a candidate that segfaults or hangs cannot take down this process. The
    // per-candidate timeout is min(3x the baseline's wall time, the absolute
    // ceiling); the baseline (timed first) gets the full ceiling.
    let exe = std::env::current_exe()
        .map_err(|err| anyhow::anyhow!("cannot resolve current exe for child trials: {err}"))?;
    let baseline_label = baseline.label.clone();
    let mut baseline_wall: Option<std::time::Duration> = None;
    let trial = |candidate: &Candidate| -> Option<camelid::gait::calibrate::TrialResult> {
        let timeout = match baseline_wall {
            Some(bw) => bw.mul_f64(3.0).min(CAL_TRIAL_CEILING),
            None => CAL_TRIAL_CEILING,
        };
        let started = std::time::Instant::now();
        let result = run_trial_in_child(
            &exe,
            &model,
            &prompt_file,
            &prompt,
            max_tokens,
            candidate,
            threads,
            timeout,
        );
        if candidate.label == baseline_label && baseline_wall.is_none() {
            baseline_wall = Some(started.elapsed());
        }
        match &result {
            Some(r) => eprintln!(
                "[gait] {:<13} {:>7.2} tok/s  parity {}",
                candidate.label,
                r.tokens_per_s,
                &r.parity_token[..12.min(r.parity_token.len())]
            ),
            None => eprintln!(
                "[gait] {:<13} disqualified (timeout/crash/parse)",
                candidate.label
            ),
        }
        result
    };

    let (outcome, path) =
        calibrate_and_store(&store_dir, &gguf, &baseline, &candidates, &config, trial);

    println!("{}", serde_json::to_string_pretty(&outcome)?);
    match path {
        Some(p) => eprintln!(
            "[gait] selected {:?} :: {} :: receipt {}",
            outcome.selected_profile,
            outcome.reason,
            p.display()
        ),
        None => eprintln!(
            "[gait] selected {:?} :: {} :: receipt NOT stored (store write failed)",
            outcome.selected_profile, outcome.reason
        ),
    }
    Ok(())
}

/// `camelid gait reset` — clear the entire GAIT cache, reverting fully to the
/// baseline path (§1.3). Best-effort and idempotent: a missing cache is not an
/// error. Deleting the folder is the documented manual revert; this is the
/// in-CLI equivalent.
fn run_gait_reset() -> anyhow::Result<()> {
    match camelid::gait::gait_dir() {
        Some(dir) if dir.exists() => {
            std::fs::remove_dir_all(&dir)?;
            println!("gait: cleared cache at {}", dir.display());
        }
        Some(dir) => {
            println!("gait: nothing to clear ({} does not exist)", dir.display());
        }
        None => println!("gait: no cache directory could be resolved"),
    }
    Ok(())
}

/// Per-candidate absolute timeout ceiling for a child trial (§1.4). The live
/// per-candidate budget is `min(3x the baseline's wall time, this ceiling)`.
const CAL_TRIAL_CEILING: std::time::Duration = std::time::Duration::from_secs(180);

/// `camelid gait-trial` (internal): run ONE candidate trial in this isolated
/// child process and print its TrialResult as a single JSON line to stdout. The
/// crash/hang isolation lives in the PARENT supervisor ([`run_trial_in_child`]);
/// here we just run the trial and report.
#[allow(clippy::too_many_arguments)]
fn run_gait_trial(
    model: PathBuf,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    max_tokens: usize,
    profile: String,
    eco_qos: bool,
    threads: Option<usize>,
    gpc_attn: Option<usize>,
    gpc_ffn: Option<usize>,
    gpc_matmul: Option<usize>,
) -> anyhow::Result<()> {
    use camelid::execution_plan::ExecutionProfile;
    use camelid::gait::calibrate::{Candidate, GroupsPerChunk};

    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    std::env::remove_var("CAMELID_GAIT");
    configure_rayon_threads(host_safe_thread_count(threads))?;

    let profile_label = profile.clone();
    let profile = match profile.to_ascii_lowercase().as_str() {
        "auto" => ExecutionProfile::Auto,
        "safe" => ExecutionProfile::Safe,
        "experimental" => ExecutionProfile::Experimental,
        "debug" => ExecutionProfile::Debug,
        other => anyhow::bail!("unknown profile {other:?} (want auto|safe|experimental|debug)"),
    };

    let prompt_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };
    let gguf = read_metadata(&model)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let prompt_token_ids = tokenizer.encode(&prompt_text, true, false)?;
    anyhow::ensure!(
        !prompt_token_ids.is_empty(),
        "prompt encoded to zero tokens"
    );

    // §5: the three gpc knobs travel together — all present, or none.
    let groups_per_chunk = match (gpc_attn, gpc_ffn, gpc_matmul) {
        (Some(a), Some(f), Some(m)) => Some(GroupsPerChunk {
            attn_qkv_decode: a,
            ffn_gate_up_decode: f,
            packed_rows4_matmul: m,
        }),
        (None, None, None) => None,
        _ => anyhow::bail!(
            "groups_per_chunk override requires all three of --gpc-attn / --gpc-ffn / --gpc-matmul"
        ),
    };
    let candidate = Candidate {
        label: profile_label,
        profile,
        eco_qos_opt_out: eco_qos,
        groups_per_chunk,
    };
    let result = gait_profile_trial(&model, threads, &prompt_token_ids, max_tokens, &candidate)?;
    // The ONLY stdout line — the JSON result the parent supervisor parses.
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

/// Run ONE candidate trial in a supervised child process, returning its result or
/// `None` if it timed out, crashed, exited non-zero, or its output could not be
/// parsed (the candidate is disqualified upstream). This is the §1.4 crash-
/// isolation boundary: a segfaulting/hanging candidate kernel cannot take down
/// the calibrating (or, in production, serving) process.
#[allow(clippy::too_many_arguments)]
fn run_trial_in_child(
    exe: &std::path::Path,
    model: &std::path::Path,
    prompt_file: &Option<PathBuf>,
    prompt: &Option<String>,
    max_tokens: usize,
    candidate: &camelid::gait::calibrate::Candidate,
    threads: Option<usize>,
    timeout: std::time::Duration,
) -> Option<camelid::gait::calibrate::TrialResult> {
    use camelid::gait::calibrate::{supervise, TrialResult, WatchdogOutcome};
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(exe);
    cmd.arg("gait-trial")
        .arg(model)
        .arg("--profile")
        .arg(gait_profile_env_value(&candidate.profile))
        .arg("--max-tokens")
        .arg(max_tokens.to_string());
    if candidate.eco_qos_opt_out {
        cmd.arg("--eco-qos");
    }
    if let Some(t) = threads {
        cmd.arg("--threads").arg(t.to_string());
    }
    if let Some(g) = candidate.groups_per_chunk {
        cmd.arg("--gpc-attn")
            .arg(g.attn_qkv_decode.to_string())
            .arg("--gpc-ffn")
            .arg(g.ffn_gate_up_decode.to_string())
            .arg("--gpc-matmul")
            .arg(g.packed_rows4_matmul.to_string());
    }
    match (prompt_file, prompt) {
        (Some(path), _) => {
            cmd.arg("--prompt-file").arg(path);
        }
        (None, Some(text)) => {
            cmd.arg("--prompt").arg(text);
        }
        (None, None) => {}
    }
    // CPU calibration; keep the child off the GPU and out of the gait selector.
    cmd.env("CUDA_VISIBLE_DEVICES", "-1");
    cmd.env_remove("CAMELID_GAIT");
    // stdout carries the JSON result; stderr is inherited so the child's logs show.
    cmd.stdout(Stdio::piped());

    let child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            eprintln!("[gait] {:<13} child spawn failed: {err}", candidate.label);
            return None;
        }
    };

    match supervise(child, timeout, std::time::Duration::from_millis(100)) {
        WatchdogOutcome::Completed(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            // The result is the last parseable JSON line (robust to stray stdout).
            stdout
                .lines()
                .rev()
                .find_map(|line| serde_json::from_str::<TrialResult>(line.trim()).ok())
        }
        WatchdogOutcome::Completed(out) => {
            eprintln!(
                "[gait] {:<13} child exited {} -> disqualified",
                candidate.label, out.status
            );
            None
        }
        WatchdogOutcome::TimedOut => {
            eprintln!(
                "[gait] {:<13} TIMED OUT after {timeout:?} -> disqualified",
                candidate.label
            );
            None
        }
        WatchdogOutcome::Errored => {
            eprintln!(
                "[gait] {:<13} child supervision error -> disqualified",
                candidate.label
            );
            None
        }
    }
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
        None,
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
    // CAMELID_DECODE_TIME=1: split per-token wall into forward / sample / other.
    let time_decode = std::env::var_os("CAMELID_DECODE_TIME").is_some();
    // Phase 0 instrumentation: per-stage decode time sinks (CPU path). Aggregates the
    // already-collected per-layer timings across all decode steps. GPU resident decode
    // runs as one fused call so these stay ~0 there; meaningful on the CPU forward.
    let stage_timings = std::env::var_os("CAMELID_STAGE_TIMINGS").is_some();
    let mut stage_us: std::collections::BTreeMap<&'static str, u128> =
        std::collections::BTreeMap::new();
    let (mut fwd_us, mut sample_us, mut steps, mut wall_us) = (0u128, 0u128, 0u64, 0u128);
    let (mut emb_us, mut layers_us) = (0u128, 0u128);
    let greedy = matches!(sampler, LlamaSampler::Greedy)
        && std::env::var_os("CAMELID_NO_GPU_SAMPLE").is_none();
    // CAMELID_SPEC_NGRAM=<max_draft>: greedy GPU speculative decoding (lossless).
    let spec_draft = std::env::var("CAMELID_SPEC_NGRAM")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0);
    // Adaptive drafting: an EMA of how many drafts get accepted per round tunes the
    // n-gram length. Start conservative (precise 4-gram, which rarely drafts on
    // non-repetitive text so it isn't slowed) and only loosen to an aggressive
    // 2-gram once repetition is proven by a high acceptance rate.
    let mut spec_ema = 0.5f32;
    let decode_start = Instant::now();
    while !finished && generated.len() < max_tokens {
        let step_started = Instant::now();
        // Greedy speculative decoding: one batched verify can emit several tokens.
        // Falls through to the single-token path when no draft / engine not ready.
        if greedy {
            if let Some(nd) = spec_draft {
                let ngram = if spec_ema >= 2.0 {
                    2
                } else if spec_ema >= 0.9 {
                    3
                } else {
                    4
                };
                if let Some(toks) =
                    session.generate_next_tokens_speculative(input[0], &history, nd, ngram)?
                {
                    // accepted drafts = tokens emitted minus the always-present bonus.
                    let accepted = toks.len().saturating_sub(1) as f32;
                    spec_ema = 0.7 * spec_ema + 0.3 * accepted;
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        steps += 1;
                    }
                    for t in toks {
                        if generated.len() >= max_tokens {
                            break;
                        }
                        generated.push(t);
                        history.push(t);
                        if tokenizer.special.eog.contains(&t) {
                            finished = true;
                            break;
                        }
                    }
                    input.clear();
                    input.push(*generated.last().expect("at least one token"));
                    continue;
                }
            }
        }
        // Greedy decode rides the resident fast lane (GPU argmax + embedding gather,
        // next graph pre-released); anything else takes the general sampling path.
        let next = if greedy {
            match session.generate_next_token_greedy_resident(input[0])? {
                Some((id, forward_us)) => {
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        fwd_us += forward_us;
                        steps += 1;
                    }
                    id
                }
                None => {
                    let step = session.generate_next_token_with_history_diagnostics(
                        &input,
                        sampler.clone(),
                        &history,
                        false,
                        None,
                    )?;
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        fwd_us += step.timings.total;
                        sample_us += step.sample;
                        emb_us += step.timings.embedding;
                        layers_us += step.timings.layers_total;
                        steps += 1;
                    }
                    if stage_timings {
                        accumulate_stage_timings(&mut stage_us, &step.timings);
                    }
                    step.next_token_id
                }
            }
        } else {
            // Temperature-only sampling rides the GPU Gumbel-max fast lane (no host
            // logits copy, no CPU sort); top-k / top-p / penalties fall through to
            // the CPU sampler.
            let gpu_sampled = match &sampler {
                LlamaSampler::Sampling(cfg) => {
                    session.generate_next_token_sampled_resident(input[0], cfg)?
                }
                LlamaSampler::Greedy => None,
            };
            match gpu_sampled {
                Some((id, forward_us)) => {
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        fwd_us += forward_us;
                        steps += 1;
                    }
                    id
                }
                None => {
                    let step = session.generate_next_token_with_history_diagnostics(
                        &input,
                        sampler.clone(),
                        &history,
                        false,
                        None,
                    )?;
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        fwd_us += step.timings.total;
                        sample_us += step.sample;
                        emb_us += step.timings.embedding;
                        layers_us += step.timings.layers_total;
                        steps += 1;
                    }
                    if stage_timings {
                        accumulate_stage_timings(&mut stage_us, &step.timings);
                    }
                    step.next_token_id
                }
            }
        };
        generated.push(next);
        history.push(next);
        finished = tokenizer.special.eog.contains(&next);
        input.clear();
        input.push(next);
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    if stage_timings && !stage_us.is_empty() {
        let total: u128 = stage_us.values().sum();
        let mut ranked: Vec<(&&str, &u128)> = stage_us.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1));
        eprintln!("[stage-timings] per-decode-step CPU breakdown (sum of all layers), total {:.2} ms/token over {} steps:", total as f64 / generated.len().max(1) as f64 / 1000.0, generated.len());
        for (name, us) in ranked {
            eprintln!(
                "  {:>18}  {:6.2}%  {:8.3} ms/token",
                name,
                *us as f64 / total as f64 * 100.0,
                *us as f64 / generated.len().max(1) as f64 / 1000.0,
            );
        }
    }
    if time_decode && steps > 0 {
        eprintln!(
            "[decode-time] per token: step wall {:.2}ms | forward {:.2}ms (embed {:.3} layers {:.2}) | sample {:.2}ms | in-step other {:.2}ms | loop other {:.2}ms",
            wall_us as f64 / steps as f64 / 1000.0,
            fwd_us as f64 / steps as f64 / 1000.0,
            emb_us as f64 / steps as f64 / 1000.0,
            layers_us as f64 / steps as f64 / 1000.0,
            sample_us as f64 / steps as f64 / 1000.0,
            (wall_us - fwd_us - sample_us) as f64 / steps as f64 / 1000.0,
            (decode_start.elapsed().as_micros() - wall_us) as f64 / steps as f64 / 1000.0,
        );
    }

    Ok(GenerationRun {
        generated,
        prefill_ms,
        ttft_ms,
        decode_ms,
    })
}

/// Phase 0 instrumentation: fold one forward step's per-stage timings into a
/// running per-stage accumulator, so a decode run can report where CPU time goes.
fn accumulate_stage_timings(
    acc: &mut std::collections::BTreeMap<&'static str, u128>,
    t: &LlamaForwardTimings,
) {
    *acc.entry("embedding").or_default() += t.embedding;
    *acc.entry("final_norm").or_default() += t.final_norm;
    *acc.entry("logits(output_proj)").or_default() += t.logits;
    for l in &t.layers {
        *acc.entry("attn_norm").or_default() += l.attention_norm;
        *acc.entry("attn_q_proj").or_default() += l.attention_q;
        *acc.entry("attn_k_proj").or_default() += l.attention_k;
        *acc.entry("attn_v_proj").or_default() += l.attention_v;
        *acc.entry("attn_rope").or_default() += l.attention_rope;
        *acc.entry("kv_write").or_default() += l.kv_cache_write;
        *acc.entry("attn_context").or_default() += l.attention_context;
        *acc.entry("attn_out_proj").or_default() += l.attention_output;
        *acc.entry("attn_residual").or_default() += l.attention_residual;
        *acc.entry("ffn_norm").or_default() += l.ffn_norm;
        *acc.entry("ffn_gate").or_default() += l.ffn_gate;
        *acc.entry("ffn_up").or_default() += l.ffn_up;
        *acc.entry("ffn_activation").or_default() += l.ffn_activation;
        *acc.entry("ffn_down").or_default() += l.ffn_down;
        *acc.entry("ffn_residual").or_default() += l.ffn_residual;
    }
}

/// A LlamaModelConfig for a known architecture, so `plan-offload --arch` can size
/// a model whose GGUF isn't on disk. Only the fields the offload planner reads
/// (dims, vocab, quant) are meaningful; the rest take neutral defaults.
fn known_arch_config(arch: &str) -> anyhow::Result<LlamaModelConfig> {
    // (block_count, hidden, ffn, heads, kv_heads, vocab, context)
    let (
        block_count,
        embedding_length,
        feed_forward_length,
        heads,
        kv_heads,
        vocab,
        context_length,
    ) = match arch.to_lowercase().as_str() {
        "llama-8b" | "llama3-8b" | "llama3.1-8b" | "8b" => (32, 4096, 14336, 32, 8, 128256, 131072),
        other => anyhow::bail!("unknown --arch {other:?}; known: llama-8b"),
    };
    Ok(LlamaModelConfig {
        context_length,
        embedding_length,
        block_count,
        feed_forward_length,
        attention_head_count: heads,
        attention_head_count_kv: kv_heads,
        kv_quant: camelid::model::KvCacheQuantization::F16,
        rope_dimension_count: None,
        rope_freq_base: None,
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-5,
        vocab_size: Some(vocab),
        file_type: Some(7), // Q8_0
        attention_key_length: None,
        rope_neox_pairing: false,
        moe: None,
        gemma4: None,
        qwen35: None,
        logit_scale: None,
        mla: None,
    })
}

/// Hardened owner-microkernel prefill measurement: load ONCE, then measure all configs INTERLEAVED
/// within each round so per-round paired deltas cancel slow thermal/clock drift. Emits raw
/// per-(round, config) JSONL; paired stats + significance are computed downstream.
#[allow(clippy::too_many_arguments)]
fn run_bench_owner_sweep(
    model: PathBuf,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    max_tokens: usize,
    rounds: usize,
    warmup_rounds: usize,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    anyhow::ensure!(rounds >= 1, "--rounds must be at least 1");
    // The sweep mutates owner env keys between configs in-process; without
    // this bypass the process-lifetime runtime-plan caches would freeze the
    // first config and every later one would silently measure it (fake null).
    // Must be set before the first inference resolves the plan.
    std::env::set_var("CAMELID_BENCH_UNCACHED_RUNTIME_PLAN", "1");
    configure_rayon_threads(threads)?;
    camelid::capability::HardwareProfile::detect().log();

    let prompt_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };

    // Load once. The owner is selected at runtime (env read per linear call), so a single load
    // serves every config; the PackedRows4 repack the owner consumes is built at load regardless.
    let gguf = read_metadata(&model)?;
    let plan_outcome = camelid::execution_plan::plan_for_model(&model, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan_outcome.env_updates);
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);

    let prompt_token_ids = tokenizer.encode(&prompt_text, true, false)?;
    let prompt_tokens = prompt_token_ids.len();
    anyhow::ensure!(prompt_tokens >= 1, "prompt encoded to zero tokens");
    let sampler = LlamaSampler::Greedy;

    // Owner keys cleared before each config so "off" is the true default path.
    let owner_keys = [
        "CAMELID_X86_Q8_MATMUL_OWNER",
        "CAMELID_X86_Q8_MATMUL_OWNER_AVX2",
        "CAMELID_X86_Q8_MATMUL_OWNER_VNNI",
        "CAMELID_X86_Q8_MATMUL_OWNER_4X8",
    ];
    // (label, owner_expected_to_fire, env). "off" is EXPLICIT since D15 made
    // the owner default-on for win-x86_64 — an empty env would measure the
    // default (owner on), not the baseline.
    type SweepConfig<'a> = (&'a str, bool, &'a [(&'a str, &'a str)]);
    let configs: &[SweepConfig] = &[
        ("off", false, &[("CAMELID_X86_Q8_MATMUL_OWNER", "off")]),
        (
            "owner_avx2",
            true,
            &[
                ("CAMELID_X86_Q8_MATMUL_OWNER", "all"),
                ("CAMELID_X86_Q8_MATMUL_OWNER_VNNI", "0"),
            ],
        ),
        (
            "owner_vnni4x4",
            true,
            &[
                ("CAMELID_X86_Q8_MATMUL_OWNER", "all"),
                ("CAMELID_X86_Q8_MATMUL_OWNER_VNNI", "1"),
                ("CAMELID_X86_Q8_MATMUL_OWNER_4X8", "0"),
            ],
        ),
        (
            "owner_vnni4x8",
            true,
            &[
                ("CAMELID_X86_Q8_MATMUL_OWNER", "all"),
                ("CAMELID_X86_Q8_MATMUL_OWNER_VNNI", "1"),
                ("CAMELID_X86_Q8_MATMUL_OWNER_4X8", "1"),
            ],
        ),
    ];
    let apply = |envs: &[(&str, &str)]| {
        for k in owner_keys {
            std::env::remove_var(k);
        }
        for (k, v) in envs {
            std::env::set_var(k, v);
        }
    };

    let model_label = model.display().to_string();
    let commit = std::env::var("CAMELID_COMMIT").unwrap_or_else(|_| "unknown".to_string());
    let total_rounds = warmup_rounds + rounds;
    eprintln!(
        "[bench-owner-sweep] {prompt_tokens} prompt tokens, {} configs, {warmup_rounds} warmup + {rounds} measured rounds interleaved",
        configs.len()
    );
    for round in 0..total_rounds {
        let measured = round >= warmup_rounds;
        for (label, owner_expected, envs) in configs {
            apply(envs);
            camelid::inference::reset_stage_timings();
            camelid::inference::reset_q8_schedule_telemetry();
            let run = generate_run(
                &config,
                &weights,
                &tokenizer,
                &prompt_token_ids,
                &sampler,
                max_tokens,
            )?;
            // Engaged-check: an owner-on config that never dispatched the
            // owner arm (e.g. env mutation swallowed by a cached plan, or the
            // planner disabled the repack) would measure a fake null. Applies
            // to warmup rounds too — fail fast.
            let owner_taken =
                camelid::inference::snapshot_q8_schedule_telemetry().matmul_owner_prefill_taken;
            anyhow::ensure!(
                *owner_expected == (owner_taken > 0),
                "engaged-check failed for config '{label}': owner_expected={owner_expected} \
                 but owner_prefill_taken={owner_taken} — the sweep would mint a fake receipt"
            );
            if !measured {
                continue;
            }
            let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
            let prefill_tok_s = if run.prefill_ms > 0.0 {
                prompt_tokens as f64 / (run.prefill_ms / 1000.0)
            } else {
                0.0
            };
            let decode_tokens = run.generated.len().saturating_sub(1);
            let decode_tok_s = if run.decode_ms > 0.0 && decode_tokens > 0 {
                decode_tokens as f64 / (run.decode_ms / 1000.0)
            } else {
                0.0
            };
            let rec = serde_json::json!({
                "schema": "camelid.bench-owner-sweep/v2",
                "round": round - warmup_rounds,
                "config": label,
                "model": model_label,
                "commit": commit,
                "prompt_tokens": prompt_tokens,
                "prefill_ms": r3(run.prefill_ms),
                "prefill_tok_s": r3(prefill_tok_s),
                "decode_tok_s": r3(decode_tok_s),
                "owner_prefill_taken": owner_taken,
            });
            println!("{}", serde_json::to_string(&rec)?);
        }
    }
    for k in owner_keys {
        std::env::remove_var(k);
    }
    Ok(())
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
    camelid::capability::HardwareProfile::detect().log();

    let prompt_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };

    // Load the model once; this cost is measured separately from generation.
    let load_start = Instant::now();
    let gguf = read_metadata(&model)?;
    // Apply the model's execution plan (as serve/chat do) BEFORE loading weights so the
    // CPU Q8 runtime repack + packed-rows4 fast path is selected at load time. Without
    // this, bench-generate measures the unplanned safe (scalar) path.
    let plan_outcome = camelid::execution_plan::plan_for_model(&model, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan_outcome.env_updates);
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

    // Drop any warmup/prefill contributions so the dump reflects only measured decode.
    camelid::inference::reset_stage_timings();
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
            offload: camelid::offload::offload_run_status(),
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
    // Per-stage CPU decode profile (no-op unless CAMELID_STAGE_TIMINGS=1).
    camelid::inference::dump_stage_timings();
    Ok(())
}

/// One full speculative generation, instrumented for SPEC_RECHECK economics. Mirrors the
/// server's accept/verify/rollback loop (`api::generate`): a normal greedy first step seeds
/// the resident engine, then each round drafts ≤γ tokens, verifies them in ONE batched
/// forward (`verify_drafts_gpu` on the resident GPU, else the CPU chunk verify), accepts the
/// longest confirmed prefix plus the target's own next token, and rolls the rest back. Every
/// emitted token is the target's greedy argmax — lossless by construction. The draft and
/// verify spans are timed separately so f_draft = draft / (draft + verify) is observable.
struct SpeculativeRun {
    generated: Vec<u32>,
    ttft_ms: f64,
    decode_ms: f64,
    rounds: u64,
    drafted: u64,
    accepted_drafts: u64,
    draft_us: u128,
    /// SPECULATIVE VERIFY time only: the batched verify calls (GPU tree, GPU linear, CPU
    /// chunk) plus failed verify attempts. It must NOT accumulate plain single-token step
    /// time — those go to `normal_step_us`. Conflating the two made `verify_ms` read as
    /// ~100% of `spec_decode_ms` and silently charged plain decode to verify overhead
    /// (BARCHAN Phase 0, amendment A4).
    verify_us: u128,
    /// Plain single-token step time, for the `normal_steps` below. Kept separate from
    /// `verify_us` so the per-round verify cost curve is attributable.
    normal_step_us: u128,
    /// Single-token plain steps taken when the drafter proposed nothing (no n-gram match).
    normal_steps: u64,
    gpu_verify_rounds: u64,
    cpu_verify_rounds: u64,
}

/// Flatten a [`TokenTree`]'s PRIMARY chain (first-child path from the root):
/// for a `branch = 1` drafter the tree IS this chain; for a branching tree it
/// is the drafter's highest-ranked continuation (children are emitted in
/// frequency order). Used by the CPU verify arm, which is strictly
/// linear-causal (no ancestor-masked chunk attention).
fn spec_tree_primary_chain(tree: &camelid::inference::spec_tree::TokenTree) -> Vec<u32> {
    let mut chain = Vec::new();
    let mut current: i32 = 0;
    loop {
        let mut next = None;
        for i in (current as usize + 1)..tree.tokens.len() {
            if tree.parent[i] == current {
                next = Some(i);
                break;
            }
        }
        match next {
            Some(i) => {
                chain.push(tree.tokens[i]);
                current = i as i32;
            }
            None => break,
        }
    }
    chain
}

fn generate_run_speculative(
    config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    drafter: &mut SpeculativeDrafter,
    draft_tokens: usize,
) -> anyhow::Result<SpeculativeRun> {
    let mut session = LlamaInferenceSession::new(config.clone(), weights.clone())?;
    // Keep the target on the resident GPU path so verify_drafts_gpu engages (this mirrors
    // the server with CAMELID_SPEC_GPU on); resident decode is the default when a CUDA
    // device is present, so this is the natural state, asserted explicitly here.
    session.set_resident_paths_disabled(false);

    let mut history: Vec<u32> = prompt_tokens.to_vec();
    let mut input: Vec<u32> = prompt_tokens.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    // TTFT span: prefill + first token. This normal step also seeds the resident engine.
    let ttft_start = Instant::now();
    let step = session.generate_next_token_with_history_diagnostics(
        &input,
        LlamaSampler::Greedy,
        &history,
        false,
        None,
    )?;
    let ttft_ms = ttft_start.elapsed().as_secs_f64() * 1000.0;
    let first = step.next_token_id;
    generated.push(first);
    history.push(first);
    let mut finished = tokenizer.special.eog.contains(&first);
    input.clear();
    input.push(first);

    let mut run = SpeculativeRun {
        generated: Vec::new(),
        ttft_ms,
        decode_ms: 0.0,
        rounds: 0,
        drafted: 0,
        accepted_drafts: 0,
        draft_us: 0,
        verify_us: 0,
        normal_step_us: 0,
        normal_steps: 0,
        gpu_verify_rounds: 0,
        cpu_verify_rounds: 0,
    };

    // Tree-speculation lane (CAMELID_SPEC_TREE): a branching drafter proposes a tree of
    // candidate continuations and one batched forward (`verify_tree_gpu`) confirms whichever
    // branch the model takes. Lossless (every emitted token is the target's greedy argmax).
    // Default off; falls back to the linear path per-round if the GPU engine isn't ready.
    let spec_tree = std::env::var_os("CAMELID_SPEC_TREE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let mut tree_drafter = camelid::inference::suffix_decoding::SuffixDecodingDrafter::default();
    // ACCEPTANCE-GATED DRAFTING (CAMELID_SPEC_TREE lane only).
    //
    // The suffix drafter only PROPOSES; `verify_tree_gpu` is the exact greedy gate, so any
    // budget we pick here stays lossless. The PROBLEM it solves: a wide tree costs a batched
    // verify + KV compaction every round regardless of how many tokens land. On low-acceptance
    // workloads (prose) most branches reject, so the wide-tree overhead exceeds the ~1 token it
    // commits and the spec lane runs SLOWER than plain decode. On high-acceptance workloads
    // (repetitive) a wide tree commits many tokens per weight read and wins big.
    //
    // Fix: gate the tree budget by RECENT acceptance (a run-length latch over accepted DRAFT tokens
    // per round; the +1 bonus is free and excluded). MEASURED full-tree acceptance on this box
    // (3B Q8, RTX 3060 6GB) cleanly separates the workloads by their net speedup:
    //   repetitive ~2.6 accepted/round -> S_sync ~1.28x (clear win)
    //   code       ~1.2               -> ~0.87x (regress)
    //   json       ~0.9               -> ~0.83x (regress)
    //   prose      ~0.5               -> ~0.76x (regress)
    // The batched-verify + KV-compaction per round only pays off at HIGH acceptance, so the policy
    // is binary: speculate (full tree every round) on the repetitive stream, SKIP (plain decode,
    // ~1.0x) on everything else. Two design points make it robust:
    //  (1) RUN-LENGTH latch (see below) keeps a speculating stream latched ON through isolated low
    //      rounds — only a RUN of consecutive non-productive rounds turns it off — so repetitive's
    //      bursty per-round variance doesn't bleed away the win via stray skips.
    //  (2) Acceptance is always measured on the SAME full tree the latch uses (warm-up and the
    //      periodic re-probe draw the full tree, never a throttled one) — a smaller probe tree would
    //      CAP how many drafts can be accepted and under-read repetitive's true ~2.6 into code's
    //      range, collapsing the gate into never speculating.
    // Thresholds are deliberately simple and collected here so they are easy to find and tune.
    //
    // The latch is RUN-LENGTH based, not a noisy per-round EWMA threshold: real-text acceptance is
    // bursty (a repetitive list still has occasional 0-accept rounds), so an EWMA Schmitt-trigger
    // flips the win off mid-stream. Instead:
    //   - While speculating, draw the full tree EVERY round (identical to the ungated path). Stay
    //     latched ON until EXIT_RUN *consecutive* rounds each accept fewer than PRODUCTIVE_DRAFTS
    //     drafts. One good round resets the run, so repetitive (which keeps landing multi-token
    //     accepts) never trips the exit; prose/code (which consistently accept ~0-1) trip it fast.
    //   - While latched OFF, SKIP (plain decode, ~1.0x). Every LOW_REPROBE skips, spend ONE
    //     full-tree probe; if it lands >= ENTER_DRAFTS accepted, re-latch ON (a stream that turned
    //     repetitive recovers). The probe is rare, so a novel stream pays ~1 wasted verify / 64 tok.
    // The latch itself now lives in `speculative::SpecLatch` (STAMPEDE P5.2) so
    // the GPU-verified and CPU-verified rounds — and, staged, the serve loop —
    // drive ONE policy. The measured constants (2/4/2/1/64) are its defaults.
    // Escape hatch for A/B measurement: CAMELID_SPEC_TREE_GATE=0 forces the OLD ungated policy
    // (full tree every round, never skip) so the gated-vs-ungated S_sync can be measured from the
    // SAME binary. Default ON (gated). The gate only changes which budget the drafter PROPOSES;
    // losslessness is the verify's job either way.
    let gate_enabled = std::env::var_os("CAMELID_SPEC_TREE_GATE")
        .map(|v| v != "0")
        .unwrap_or(true);
    let mut latch = camelid::inference::speculative::SpecLatch::default();
    // STAMPEDE Phase 5 (P5.1): when the resident GPU verify is unavailable
    // (CPU-only box, CUDA hidden, resident decode off), verify the drafted
    // chain on the CPU via the batched chunk forward + KV rollback — the same
    // shipped pattern the linear lane below uses. Kill-switch:
    // CAMELID_SPEC_CPU_VERIFY=0 restores the old skip-to-plain behavior.
    let cpu_verify_allowed = std::env::var_os("CAMELID_SPEC_CPU_VERIFY")
        .map(|v| v != "0")
        .unwrap_or(true);
    // One-way ratchet: after the first CPU-verified round the session is
    // pinned off the resident paths (the chunk-verify rollback requires
    // CPU-authoritative KV, and `rollback_to_position` drops the resident
    // engine anyway — never alternate modes mid-run).
    let mut cpu_verify_pinned = false;

    let decode_start = Instant::now();
    while !finished && generated.len() < max_tokens {
        let remaining = max_tokens.saturating_sub(generated.len());
        let context_room = session.remaining_context();
        let budget = draft_tokens
            .min(remaining.saturating_sub(1))
            .min(context_room.saturating_sub(1));

        // Tree round: draft a branching tree and verify it in one batched forward.
        if spec_tree && budget > 0 && context_room > 0 {
            use camelid::inference::spec_tree::{TreeDrafter, TREE_MAX_NODES};
            let anchor = input[0];

            // Choose this round's tree budget from the run-length latch (the gate). Returns None to
            // SKIP speculation: take a plain greedy step instead. The FULL tree is the original
            // ungated budget: a gamma-deep chain (the suffix drafter may branch within the node
            // cap). Every band that speculates draws this same full tree so acceptance is measured
            // at the size the latched-ON band actually uses (a throttled probe would under-read it).
            let full_tree = ((budget + 1).min(TREE_MAX_NODES), budget);
            // Warm-up / latched-ON / re-probe rounds all draw the SAME full
            // tree (acceptance must be measured at the size the latched-ON
            // band uses); latched-OFF rounds skip speculation entirely.
            let chosen_budget: Option<(usize, usize)> = if !gate_enabled {
                // Ungated baseline (A/B): the original always-full-tree policy.
                Some(full_tree)
            } else if latch.should_speculate() {
                Some(full_tree)
            } else {
                None
            };

            if std::env::var_os("CAMELID_SPEC_TREE_TRACE").is_some() {
                eprintln!(
                    "[spec-tree] round_seen={} spec={} nonprod_run={} skips={} budget={} -> {:?}",
                    latch.rounds_done(),
                    latch.speculating(),
                    latch.nonproductive_run(),
                    latch.consecutive_skips(),
                    budget,
                    chosen_budget
                );
            }
            let Some((max_nodes, max_depth)) = chosen_budget else {
                // Latched OFF: one plain resident greedy step (no speculation this round). Recovery
                // is handled by the periodic full-tree re-probe; no speculation cost is paid here.
                latch.note_skip();
                let step_started = Instant::now();
                let next = match session.generate_next_token_greedy_resident(input[0])? {
                    Some((id, _us)) => id,
                    None => {
                        session
                            .generate_next_token_with_history_diagnostics(
                                &input,
                                LlamaSampler::Greedy,
                                &history,
                                false,
                                None,
                            )?
                            .next_token_id
                    }
                };
                run.normal_step_us += step_started.elapsed().as_micros();
                run.normal_steps += 1;
                generated.push(next);
                history.push(next);
                if tokenizer.special.eog.contains(&next) {
                    finished = true;
                }
                input.clear();
                input.push(*generated.last().expect("just pushed a token"));
                continue;
            };

            let draft_started = Instant::now();
            let tree = tree_drafter.draft_tree(&history, anchor, max_nodes, max_depth);
            run.draft_us += draft_started.elapsed().as_micros();
            if tree.nodes() > 1 {
                let verify_started = Instant::now();
                let gpu_emitted = if cpu_verify_pinned {
                    None
                } else {
                    session.verify_tree_gpu(&tree)?
                };
                if let Some(emitted) = gpu_emitted {
                    run.verify_us += verify_started.elapsed().as_micros();
                    // A verified round drives the run-length latch. accepted_drafts = emitted minus
                    // the guaranteed +1 bonus.
                    let accepted_drafts = (emitted.len() as u64).saturating_sub(1) as u32;
                    latch.note_verified(accepted_drafts);
                    run.gpu_verify_rounds += 1;
                    run.rounds += 1;
                    run.drafted += (tree.nodes() - 1) as u64;
                    run.accepted_drafts += (emitted.len() as u64).saturating_sub(1);
                    for token in emitted {
                        if generated.len() >= max_tokens {
                            break;
                        }
                        generated.push(token);
                        history.push(token);
                        if tokenizer.special.eog.contains(&token) {
                            finished = true;
                            break;
                        }
                    }
                    input.clear();
                    input.push(*generated.last().expect("a tree round emits >=1 token"));
                    continue;
                }
                // STAMPEDE Phase 5 (P5.1): resident GPU verify unavailable —
                // verify the tree's PRIMARY CHAIN on the CPU via the batched
                // chunk forward + KV rollback (the linear lane's shipped
                // pattern). Lossless: every emitted token is the target's own
                // greedy argmax given the accepted prefix.
                if cpu_verify_allowed {
                    let chain = spec_tree_primary_chain(&tree);
                    if !chain.is_empty() {
                        if !cpu_verify_pinned {
                            // One-way ratchet: pin the session off the resident
                            // paths (rollback requires CPU-authoritative KV) and
                            // switch the drafter to linear chains — deeper
                            // proposals within the same node budget, and the
                            // primary-chain flatten becomes exact.
                            cpu_verify_pinned = true;
                            session.set_resident_paths_disabled(true);
                            tree_drafter.branch = 1;
                        }
                        let base_position = session.kv_position();
                        let mut batch = Vec::with_capacity(1 + chain.len());
                        batch.push(anchor);
                        batch.extend_from_slice(&chain);
                        let (predictions, verify_timings) =
                            session.forward_greedy_verify_chunk(&batch)?;
                        // Small-M verify economics profiling (STAMPEDE P5
                        // follow-up): component split of the chunk forward.
                        if std::env::var_os("CAMELID_SPEC_VERIFY_TIMINGS").is_some() {
                            let mut sums = [0u128; 15];
                            for l in &verify_timings.layers {
                                for (slot, v) in sums.iter_mut().zip([
                                    l.attention_norm,
                                    l.attention_q,
                                    l.attention_k,
                                    l.attention_v,
                                    l.attention_rope,
                                    l.kv_cache_write,
                                    l.attention_context,
                                    l.attention_output,
                                    l.attention_residual,
                                    l.ffn_norm,
                                    l.ffn_gate,
                                    l.ffn_up,
                                    l.ffn_activation,
                                    l.ffn_down,
                                    l.ffn_residual,
                                ]) {
                                    *slot += v;
                                }
                            }
                            eprintln!(
                                "[spec-verify] rows={} layers_us={} logits_us={} | anorm={} q={} k={} v={} rope={} kvw={} actx={} aout={} ares={} fnorm={} gate={} up={} act={} down={} fres={}",
                                batch.len(),
                                verify_timings.layers_total,
                                verify_timings.logits,
                                sums[0], sums[1], sums[2], sums[3], sums[4], sums[5], sums[6],
                                sums[7], sums[8], sums[9], sums[10], sums[11], sums[12], sums[13],
                                sums[14]
                            );
                        }
                        let accepted = accepted_draft_prefix(&chain, &predictions);
                        session.rollback_to_position(base_position + 1 + accepted)?;
                        run.verify_us += verify_started.elapsed().as_micros();
                        latch.note_verified(accepted as u32);
                        run.cpu_verify_rounds += 1;
                        run.rounds += 1;
                        run.drafted += chain.len() as u64;
                        run.accepted_drafts += accepted as u64;
                        for token in &predictions[..=accepted] {
                            if generated.len() >= max_tokens {
                                break;
                            }
                            generated.push(*token);
                            history.push(*token);
                            if tokenizer.special.eog.contains(token) {
                                finished = true;
                                break;
                            }
                        }
                        input.clear();
                        input.push(*generated.last().expect("a verify round emits >=1 token"));
                        continue;
                    }
                }
                // Engine not ready and CPU verify disabled/empty-chain: fall through to a plain
                // step. Don't score this as a low-acceptance round — it's an engine-readiness
                // miss, not a drafter miss — so leave the latch untouched.
                run.verify_us += verify_started.elapsed().as_micros();
            } else {
                // The drafter found NO recurrence (anchor-only tree). This is TRANSIENT and common
                // early in a stream (the recurrence hasn't built up yet), so it must NOT crash the
                // EWMA into the LOW band before the stream's true acceptance is ever observed. Treat
                // it exactly like the ungated path does: take a cheap plain step and try again next
                // round. The suffix scan that produced it is O(window) and cheap, and crucially NO
                // batched verify or KV compaction ran (the expensive part) — so on purely novel
                // text this costs essentially the same as plain decode. Leave the EWMA untouched.
            }
            // No usable tree this round → one plain resident greedy step.
            let step_started = Instant::now();
            let next = match session.generate_next_token_greedy_resident(input[0])? {
                Some((id, _us)) => id,
                None => {
                    session
                        .generate_next_token_with_history_diagnostics(
                            &input,
                            LlamaSampler::Greedy,
                            &history,
                            false,
                            None,
                        )?
                        .next_token_id
                }
            };
            run.normal_step_us += step_started.elapsed().as_micros();
            run.normal_steps += 1;
            generated.push(next);
            history.push(next);
            if tokenizer.special.eog.contains(&next) {
                finished = true;
            }
            input.clear();
            input.push(*generated.last().expect("just pushed a token"));
            continue;
        }

        let draft_started = Instant::now();
        let drafts = if budget > 0 && context_room > 0 {
            drafter.draft(&history, budget)?
        } else {
            Vec::new()
        };
        run.draft_us += draft_started.elapsed().as_micros();

        if drafts.is_empty() {
            // No draft proposed → one plain resident greedy step (same path the plain
            // baseline takes), so a config that never drafts measures as plain decode.
            let step_started = Instant::now();
            let next = match session.generate_next_token_greedy_resident(input[0])? {
                Some((id, _us)) => id,
                None => {
                    session
                        .generate_next_token_with_history_diagnostics(
                            &input,
                            LlamaSampler::Greedy,
                            &history,
                            false,
                            None,
                        )?
                        .next_token_id
                }
            };
            run.normal_step_us += step_started.elapsed().as_micros();
            run.normal_steps += 1;
            generated.push(next);
            history.push(next);
            if tokenizer.special.eog.contains(&next) {
                finished = true;
            }
            input.clear();
            input.push(*generated.last().expect("just pushed a token"));
            continue;
        }

        // Verify all drafts in one batched forward: GPU resident when ready, else the CPU
        // chunk verify with an explicit KV rollback. Both are lossless — the emitted tokens
        // are the target's own greedy argmax given the accepted prefix.
        let verify_started = Instant::now();
        let emitted: Vec<u32> = match session.verify_drafts_gpu(input[0], &drafts)? {
            Some(accepted) => {
                run.gpu_verify_rounds += 1;
                run.rounds += 1;
                run.drafted += drafts.len() as u64;
                run.accepted_drafts += (accepted.len() as u64).saturating_sub(1);
                accepted
            }
            None => {
                let base_position = session.kv_position();
                let mut batch = Vec::with_capacity(1 + drafts.len());
                batch.push(input[0]);
                batch.extend_from_slice(&drafts);
                let (predictions, _timings) = session.forward_greedy_verify_chunk(&batch)?;
                let accepted = accepted_draft_prefix(&drafts, &predictions);
                session.rollback_to_position(base_position + 1 + accepted)?;
                run.cpu_verify_rounds += 1;
                run.rounds += 1;
                run.drafted += drafts.len() as u64;
                run.accepted_drafts += accepted as u64;
                predictions[..=accepted].to_vec()
            }
        };
        run.verify_us += verify_started.elapsed().as_micros();

        for token in emitted {
            if generated.len() >= max_tokens {
                break;
            }
            generated.push(token);
            history.push(token);
            if tokenizer.special.eog.contains(&token) {
                finished = true;
                break;
            }
        }
        input.clear();
        input.push(*generated.last().expect("a round emits at least one token"));
    }
    run.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    run.generated = generated;
    Ok(run)
}

#[derive(Serialize)]
struct BenchSpeculativeRecord {
    runtime: &'static str,
    commit: String,
    workload: String,
    model: String,
    draft_model: Option<String>,
    quantization: String,
    drafter: String,
    cpu_draft: bool,
    /// True when --spec-only: the plain fields below are the coexistence-config target (draft
    /// resident), NOT a full-resident-target baseline. S_sync here is speculation efficiency on
    /// the same target; the full-resident denominator comes from the n-gram/bench-generate runs.
    spec_only: bool,
    draft_tokens: usize,
    prompt_tokens: usize,
    max_tokens: usize,

    // Plain greedy baseline (this run, same target, same machine).
    plain_generated_tokens: usize,
    plain_ttft_ms: f64,
    plain_decode_ms: f64,
    plain_tokens_per_second: f64,

    // Speculative run.
    spec_generated_tokens: usize,
    spec_ttft_ms: f64,
    spec_decode_ms: f64,
    spec_tokens_per_second: f64,

    // Economics.
    rounds: u64,
    drafted: u64,
    accepted_drafts: u64,
    accept_rate: f64,
    mean_accepted_tokens_per_round: f64,
    draft_ms: f64,
    /// SPECULATIVE VERIFY time only — batched verify calls plus failed verify attempts.
    /// Plain single-token step time is reported separately in `normal_step_ms`; before
    /// BARCHAN Phase 0 the two were summed here, which made this field read as ~100% of
    /// `spec_decode_ms` and made any per-round verify cost derived from it wrong.
    verify_ms: f64,
    /// Plain single-token step time for the `normal_steps` rounds (drafter proposed
    /// nothing, latch skipped, or the engine was not ready). Not speculation overhead.
    normal_step_ms: f64,
    /// draft / (draft + verify): the fraction of round time spent drafting. The Phase-4
    /// decision gate turns on this — ~0 for n-gram (nothing to hide with concurrency).
    f_draft: f64,
    /// Synchronous speedup over this machine's own plain greedy decode (spec t/s ÷ plain t/s).
    s_sync: f64,
    normal_steps: u64,
    gpu_verify_rounds: u64,
    cpu_verify_rounds: u64,

    // Lossless gate (intra-Camelid: spec stream vs this run's plain greedy stream).
    first_divergent_generated_token_index: i64,
    lossless: bool,

    peak_memory_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    offload: Option<camelid::offload::OffloadRunStatus>,
}

/// Load a draft GGUF and wrap it as a `ModelDrafter`. Mirrors the target load path so the
/// draft rides the same execution plan; the drafter routes to its own resident cache.
fn load_model_drafter(
    path: &std::path::Path,
    target_tokenizer: &Tokenizer,
    cpu_draft: bool,
    threads: Option<usize>,
) -> anyhow::Result<SpeculativeDrafter> {
    let gguf = read_metadata(path)?;
    let plan_outcome = camelid::execution_plan::plan_for_model(path, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan_outcome.env_updates);
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(path, &gguf);
    let draft_tokenizer = Tokenizer::from_gguf(&gguf)?;
    // Drafted token ids must mean the same text in the target vocabulary. Lossless either
    // way (the verify is authoritative), but a mismatched vocab silently drives accept to ~0.
    anyhow::ensure!(
        draft_tokenizer.model == target_tokenizer.model,
        "draft model tokenizer ({:?}) differs from target ({:?}); drafted ids would not share \
         the target vocabulary",
        draft_tokenizer.model,
        target_tokenizer.model
    );
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);
    let mut session = LlamaInferenceSession::new(config, weights)?;
    if cpu_draft {
        // Path 3 (SPEC_RECHECK): force the draft onto the CPU forward (the previously
        // "blocked" configuration). Otherwise the draft stays GPU-resident by default.
        session.set_resident_paths_disabled(true);
    }
    Ok(SpeculativeDrafter::Model(Box::new(ModelDrafter::new(
        session,
    ))))
}

#[allow(clippy::too_many_arguments)]
fn run_bench_speculative(
    model: PathBuf,
    drafter_kind: String,
    draft_model: Option<PathBuf>,
    draft_tokens: Option<usize>,
    cpu_draft: bool,
    spec_only: bool,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    workload: String,
    max_tokens: usize,
    warmup: bool,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    configure_rayon_threads(threads)?;
    camelid::capability::HardwareProfile::detect().log();

    let prompt_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };

    // Load the target exactly as bench-generate does (execution plan applied before weights).
    let gguf = read_metadata(&model)?;
    let plan_outcome = camelid::execution_plan::plan_for_model(&model, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan_outcome.env_updates);
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);

    let prompt_token_ids = tokenizer.encode(&prompt_text, true, false)?;
    let prompt_tokens = prompt_token_ids.len();
    anyhow::ensure!(prompt_tokens >= 1, "prompt encoded to zero tokens");

    let gamma = draft_tokens.unwrap_or(match drafter_kind.as_str() {
        "draft" => DEFAULT_MODEL_DRAFT_TOKENS,
        _ => DEFAULT_NGRAM_DRAFT_TOKENS,
    });

    let build_drafter = || -> anyhow::Result<SpeculativeDrafter> {
        match drafter_kind.as_str() {
            "ngram" => Ok(SpeculativeDrafter::NGram(NGramDrafter::default())),
            "draft" => {
                let path = draft_model.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("--drafter draft requires --draft-model <gguf>")
                })?;
                load_model_drafter(path, &tokenizer, cpu_draft, threads)
            }
            other => anyhow::bail!("unknown --drafter {other:?}; expected \"ngram\" or \"draft\""),
        }
    };

    let sampler = LlamaSampler::Greedy;

    // Two orderings. Default: the plain baseline runs FIRST on a full-resident target (the truest
    // S_sync denominator), then the drafter is added. spec_only: the drafter (and its coexistence
    // reserve) is established BEFORE any target build, so the target builds once under the
    // coexistence budget and the draft stays GPU-resident; the plain reference then reuses that
    // same resident target (so its tps is the coexistence-config target, flagged in the record).
    let (plain, spec, (draft_fwd_us, draft_resident_steps, draft_cpu_steps)) = if spec_only {
        if warmup {
            eprintln!("[bench-speculative] warmup (unmeasured, spec-only)...");
            let mut w = build_drafter()?;
            let _ = generate_run_speculative(
                &config,
                &weights,
                &tokenizer,
                &prompt_token_ids,
                max_tokens,
                &mut w,
                gamma,
            )?;
        }
        camelid::inference::reset_stage_timings();
        let mut drafter = build_drafter()?;
        let spec = generate_run_speculative(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            max_tokens,
            &mut drafter,
            gamma,
        )?;
        let draft_stats = drafter.take_forward_stats();
        // Plain reference reuses the resident coexistence target engine (no rebuild).
        let plain = generate_run(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            &sampler,
            max_tokens,
        )?;
        (plain, spec, draft_stats)
    } else {
        if warmup {
            eprintln!("[bench-speculative] warmup (unmeasured)...");
            let _ = generate_run(
                &config,
                &weights,
                &tokenizer,
                &prompt_token_ids,
                &sampler,
                max_tokens,
            )?;
            let mut warm = build_drafter()?;
            let _ = generate_run_speculative(
                &config,
                &weights,
                &tokenizer,
                &prompt_token_ids,
                max_tokens,
                &mut warm,
                gamma,
            )?;
        }
        camelid::inference::reset_stage_timings();
        // Single-model baseline: clear any reserve a warmup drafter set so the denominator is a
        // full-resident target, not one that left room for a draft.
        camelid::inference::set_spec_coexist_reserve(0);
        let plain = generate_run(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            &sampler,
            max_tokens,
        )?;
        let mut drafter = build_drafter()?;
        let spec = generate_run_speculative(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            max_tokens,
            &mut drafter,
            gamma,
        )?;
        let draft_stats = drafter.take_forward_stats();
        (plain, spec, draft_stats)
    };
    let plain_decode_tokens = plain.generated.len().saturating_sub(1);
    let plain_tps = if plain.decode_ms > 0.0 && plain_decode_tokens > 0 {
        plain_decode_tokens as f64 / (plain.decode_ms / 1000.0)
    } else {
        0.0
    };
    let spec_decode_tokens = spec.generated.len().saturating_sub(1);
    let spec_tps = if spec.decode_ms > 0.0 && spec_decode_tokens > 0 {
        spec_decode_tokens as f64 / (spec.decode_ms / 1000.0)
    } else {
        0.0
    };
    // Draft-decode profiling: the GPU forward time of the draft steps vs the wall-clock draft
    // time tells whether the draft cost is in the forward kernels or in sync/overhead around them.
    if draft_resident_steps + draft_cpu_steps > 0 {
        eprintln!(
            "[draft-profile] resident steps {} ({:.1} ms/step GPU forward) | cpu-fallback steps {} | \
             wall draft {:.1} ms total = {:.1} ms/step | GPU-forward fraction {:.0}%",
            draft_resident_steps,
            if draft_resident_steps > 0 {
                draft_fwd_us as f64 / 1000.0 / draft_resident_steps as f64
            } else {
                0.0
            },
            draft_cpu_steps,
            spec.draft_us as f64 / 1000.0,
            if draft_resident_steps + draft_cpu_steps > 0 {
                spec.draft_us as f64 / 1000.0 / (draft_resident_steps + draft_cpu_steps) as f64
            } else {
                0.0
            },
            if spec.draft_us > 0 {
                draft_fwd_us as f64 / spec.draft_us as f64 * 100.0
            } else {
                0.0
            },
        );
    }

    // Lossless gate: first index where the spec stream diverges from plain greedy (-1 if the
    // two streams are identical). A positive cell with any divergence is a correctness bug.
    let first_divergent = first_divergence(&spec.generated, &plain.generated);

    let accept_rate = if spec.drafted > 0 {
        spec.accepted_drafts as f64 / spec.drafted as f64
    } else {
        0.0
    };
    // Each verify round emits accepted drafts + 1 bonus token.
    let mean_accepted_tokens_per_round = if spec.rounds > 0 {
        (spec.accepted_drafts + spec.rounds) as f64 / spec.rounds as f64
    } else {
        0.0
    };
    let draft_ms = spec.draft_us as f64 / 1000.0;
    let verify_ms = spec.verify_us as f64 / 1000.0;
    let normal_step_ms = spec.normal_step_us as f64 / 1000.0;
    let f_draft = if draft_ms + verify_ms > 0.0 {
        draft_ms / (draft_ms + verify_ms)
    } else {
        0.0
    };
    let s_sync = if plain_tps > 0.0 {
        spec_tps / plain_tps
    } else {
        0.0
    };

    let record = BenchSpeculativeRecord {
        runtime: "camelid",
        commit: std::env::var("CAMELID_COMMIT").unwrap_or_else(|_| "unknown".to_string()),
        workload,
        model: model.display().to_string(),
        draft_model: draft_model.as_ref().map(|p| p.display().to_string()),
        quantization: infer_quantization(&model),
        drafter: drafter_kind,
        cpu_draft,
        spec_only,
        draft_tokens: gamma,
        prompt_tokens,
        max_tokens,
        plain_generated_tokens: plain.generated.len(),
        plain_ttft_ms: plain.ttft_ms,
        plain_decode_ms: plain.decode_ms,
        plain_tokens_per_second: plain_tps,
        spec_generated_tokens: spec.generated.len(),
        spec_ttft_ms: spec.ttft_ms,
        spec_decode_ms: spec.decode_ms,
        spec_tokens_per_second: spec_tps,
        rounds: spec.rounds,
        drafted: spec.drafted,
        accepted_drafts: spec.accepted_drafts,
        accept_rate,
        mean_accepted_tokens_per_round,
        draft_ms,
        verify_ms,
        normal_step_ms,
        f_draft,
        s_sync,
        normal_steps: spec.normal_steps,
        gpu_verify_rounds: spec.gpu_verify_rounds,
        cpu_verify_rounds: spec.cpu_verify_rounds,
        first_divergent_generated_token_index: first_divergent,
        lossless: first_divergent < 0,
        peak_memory_bytes: peak_rss_bytes(),
        offload: camelid::offload::offload_run_status(),
    };

    let stdout = std::io::stdout();
    {
        let mut handle = stdout.lock();
        writeln!(handle, "{}", serde_json::to_string(&record)?)?;
        handle.flush()?;
    }
    eprintln!(
        "[bench-speculative] {} | {} γ={}{} | accept {:.1}% | tok/round {:.2} | f_draft {:.3} | \
         draft {:.1} ms/tok | plain {:.2} t/s → spec {:.2} t/s | S_sync {:.2}x | {} | gpu/cpu verify {}/{} | drafted {} rounds {}",
        record.workload,
        record.drafter,
        record.draft_tokens,
        if record.spec_only { " spec-only" } else { "" },
        record.accept_rate * 100.0,
        record.mean_accepted_tokens_per_round,
        record.f_draft,
        if record.drafted > 0 { record.draft_ms / record.drafted as f64 } else { 0.0 },
        record.plain_tokens_per_second,
        record.spec_tokens_per_second,
        record.s_sync,
        if record.lossless {
            "LOSSLESS ✓".to_string()
        } else {
            format!("DIVERGED @ {}", record.first_divergent_generated_token_index)
        },
        record.gpu_verify_rounds,
        record.cpu_verify_rounds,
        record.drafted,
        record.rounds,
    );
    Ok(())
}

/// First index at which `a` and `b` differ, or `-1` if one is a prefix of the other AND they
/// are the same length (i.e. identical). Differing lengths count as a divergence at the
/// shorter length — for the lossless gate, two greedy streams must be byte-identical.
fn first_divergence(a: &[u32], b: &[u32]) -> i64 {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return i as i64;
        }
    }
    if a.len() == b.len() {
        -1
    } else {
        n as i64
    }
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

/// The measured-fastest Metal configuration is on by default for the CLI: Q8_0 weights
/// upload in wire format, NSG=8 GEMV dispatch, f32-activation GEMV chain, tiled decode
/// attention, and the one-command-buffer GPU prefill. Each remains overridable: set the
/// variable to 0 to opt out, and the resident decode itself stays opt-in via
/// CAMELID_METAL_RESIDENT_DECODE. (Library defaults are unchanged: this runs only in the
/// CLI entry, so test suites and embedders see the conservative paths unless they enable.)
fn apply_default_fast_stack() {
    for key in [
        "CAMELID_METAL_RESIDENT_DECODE",
        "CAMELID_METAL_F32Y",
        "CAMELID_METAL_WIRE",
        "CAMELID_METAL_WIRE_NSG8",
        "CAMELID_METAL_ATTN2",
        "CAMELID_METAL_RESIDENT_PREFILL",
        "CAMELID_METAL_MM",
    ] {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, "1");
        }
    }
}

/// True when the parsed subcommand opted into deterministic inference (`--deterministic`).
/// Only `serve` and `bench-generate` expose the flag today (the supported single-node
/// generate/serve path); every other subcommand keeps the default fast stack.
fn command_requests_deterministic(command: &Command) -> bool {
    matches!(
        command,
        Command::Serve {
            deterministic: true,
            ..
        } | Command::BenchGenerate {
            deterministic: true,
            ..
        }
    )
}

/// Pin the process to the opt-in deterministic CPU forward pass. Sets
/// `CAMELID_DETERMINISTIC=1` — which the engine reads (`inference::deterministic_mode_enabled`)
/// to fail every Metal/GPU dispatch gate closed to its order-stable CPU equivalent — then
/// forces the whole Metal fast stack off and disables GPU sampling so greedy decode also
/// stays on the CPU path. The result is bit-exact, reduction-order-stable logits for the
/// supported TinyLlama 1.1B Q8_0 lane. Only the CLI entry calls this; library defaults and
/// the default (GPU) fast path are byte-for-byte unchanged. The pinned reduction order
/// mirrors the llama.cpp reference block-wise Q8_0 dot layout the parity contract is gated
/// against (see DECISIONS.md §D9 and `qa/determinism/determinism-baseline-*.md`).
fn apply_deterministic_mode() {
    std::env::set_var("CAMELID_DETERMINISTIC", "1");
    for key in [
        "CAMELID_METAL_RESIDENT_DECODE",
        "CAMELID_METAL_F32Y",
        "CAMELID_METAL_WIRE",
        "CAMELID_METAL_WIRE_NSG8",
        "CAMELID_METAL_ATTN2",
        "CAMELID_METAL_RESIDENT_PREFILL",
        "CAMELID_METAL_MM",
        "CAMELID_METAL_LINEAR",
        "CAMELID_METAL_Q8",
        "CAMELID_METAL_Q8_RETAINED",
        "CAMELID_HYBRID_Q8_RETAINED",
        "CAMELID_METAL_NOCOPY",
    ] {
        std::env::set_var(key, "0");
    }
    std::env::set_var("CAMELID_NO_GPU_SAMPLE", "1");
    eprintln!(
        "[deterministic] pinned to the order-stable CPU forward pass (Metal/GPU stack off). \
         Reduction order follows the llama.cpp reference block-wise Q8_0 layout; see DECISIONS.md \u{a7}D9."
    );
}

/// Default the single-node serve/benchmark path to fast-load
/// (CAMELID_METAL_NOCOPY): Q8_0
/// weights map straight into page-aligned wire pages the GPU reads in place — same
/// decode speed, ~36% lower peak RSS, and warm reloads in seconds instead of the
/// full disk pass. Gated to exactly the configuration that can consume wire pages:
/// macOS, the resident decode path active, and the wire kernel stack on. This is
/// why callers apply it after command-specific mode selection instead of from
/// `apply_default_fast_stack` — speculative
/// decoding disables resident decode (its CPU repack plan needs the materialized
/// blocks), any wire-off override falls back to the block path, and the
/// distributed nodes (whose CPU forward needs `q8_0_blocks`) never run this arm.
/// Opt out with CAMELID_METAL_NOCOPY=0.
fn apply_serve_nocopy_default() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let on = |key: &str| std::env::var(key).map(|v| v == "1").unwrap_or(false);
    if should_default_serve_nocopy(
        std::env::var_os("CAMELID_METAL_NOCOPY").is_some(),
        on("CAMELID_METAL_RESIDENT_DECODE"),
        on("CAMELID_METAL_WIRE"),
        on("CAMELID_METAL_F32Y"),
    ) {
        std::env::set_var("CAMELID_METAL_NOCOPY", "1");
    }
}

/// Pure decision for [`apply_serve_nocopy_default`]: default fast-load on only when
/// the user has not set the flag either way AND the wire-resident stack that can
/// consume wire pages is active. Speculative decoding turns resident decode off, so
/// `resident == false` keeps NOCOPY off; an explicit `=0` sets `already_set` and is
/// honored.
fn should_default_serve_nocopy(already_set: bool, resident: bool, wire: bool, f32y: bool) -> bool {
    !already_set && resident && wire && f32y
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

/// Peak resident set size of this process in bytes. macOS `getrusage` reports
/// bytes; other Unix reports kilobytes (scaled here).
#[cfg(unix)]
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

/// Peak resident set size of this process in bytes. Windows exposes the peak
/// working set directly via `GetProcessMemoryInfo`.
#[cfg(windows)]
fn peak_rss_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: GetCurrentProcess returns a valid pseudo-handle; `counters` is
    // sized via its `cb` field per the API contract.
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        return 0;
    }
    counters.PeakWorkingSetSize as u64
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
            let (out, _bytes, _wait_us, _forward_us, _read_us, _decode_us) = ghost_stream_layers(
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

fn apply_spec_decode_env(
    spec_decode: Option<String>,
    spec_draft_model: Option<PathBuf>,
    spec_draft_tokens: Option<usize>,
) {
    let mode = spec_decode.filter(|mode| {
        let trimmed = mode.trim();
        !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("off")
    });
    if let Some(mode) = mode {
        std::env::set_var("CAMELID_SPEC_DECODE", mode);
        // GPU speculative verify (CAMELID_SPEC_GPU=1) runs the batched `verify_batch` on the
        // target's resident engine, which owns the weights — so keep the Metal resident paths
        // ON for it. Without GPU verify the CPU chunk verify needs CPU-resident packed Q8
        // weights, but the Metal-resident plan deliberately keeps CPU-side weights file-backed
        // (the GPU owns the resident copy), so each verify round would pay a file-speed weight
        // pass — fall back to the validated CPU repack plan in that case only.
        let spec_gpu = matches!(
            std::env::var("CAMELID_SPEC_GPU").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        );
        if !spec_gpu {
            std::env::set_var("CAMELID_METAL_RESIDENT_DECODE", "0");
            std::env::set_var("CAMELID_METAL_RESIDENT_PREFILL", "0");
            tracing::info!(
                "speculative decoding enabled; selecting the CPU execution plan \
                 (Metal resident paths disabled server-wide)"
            );
        } else {
            tracing::info!(
                "speculative decoding enabled with GPU verify (CAMELID_SPEC_GPU=1); \
                 keeping the resident decode engine for the batched verify"
            );
        }
    }
    if let Some(path) = spec_draft_model {
        std::env::set_var("CAMELID_SPEC_DRAFT_MODEL", path);
    }
    if let Some(tokens) = spec_draft_tokens.filter(|tokens| *tokens > 0) {
        std::env::set_var("CAMELID_SPEC_DRAFT_TOKENS", tokens.to_string());
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
    // Probe CUDA at startup so the selected GPU (index, name, compute capability,
    // VRAM) is logged at launch via cuda::init_backend, and surface availability
    // here. A present CUDA device is always the discrete NVIDIA GPU — the Intel
    // iGPU is not CUDA-capable and is never enumerated.
    let cuda = camelid::cuda::detect_cuda_device();
    tracing::info!(
        cuda_available = cuda.available,
        cuda_device = cuda.device_name.as_deref().unwrap_or("none"),
        cuda_reason = cuda.reason.as_deref().unwrap_or(""),
        "camelid cuda state"
    );
}

// Physical-core detection moved into the library (the decode thread-policy
// default needs it too); this alias keeps the call sites below unchanged.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
use camelid::inference::windows_physical_core_count;

/// §1.2 core-cap: clamp a resolved compute-thread count so the OS always keeps
/// its reserve (see [`camelid::gait::compute_thread_budget`]). Windows x86-64
/// only — the GAIT substrate's scope; elsewhere `requested` is returned
/// unchanged. A `None` request resolves to the full safe budget.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn host_safe_thread_count(requested: Option<usize>) -> Option<usize> {
    let phys = windows_physical_core_count()?;
    let budget = camelid::gait::compute_thread_budget(phys);
    Some(
        requested
            .map(|r| r.min(budget.threads))
            .unwrap_or(budget.threads),
    )
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
fn host_safe_thread_count(requested: Option<usize>) -> Option<usize> {
    requested
}

fn configure_rayon_threads(threads: Option<usize>) -> anyhow::Result<()> {
    if let Some(t) = threads {
        anyhow::ensure!(t > 0, "--threads must be greater than zero");
    }
    // When the caller did not pin a thread count, default the Windows pool to the
    // physical core count (SMT siblings hurt compute-bound decode). Other targets
    // keep their existing defaults.
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    let resolved = threads.or_else(windows_physical_core_count);
    #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
    let resolved = threads;

    // §1.2 host-safety: when GAIT is engaged, cap the pool so the OS keeps a core
    // reserve. Gated on the bring-up flag so the default path is byte-identical;
    // when GAIT becomes the baseline the cap becomes unconditional.
    let resolved = if camelid::gait::gait_enabled() {
        host_safe_thread_count(resolved)
    } else {
        resolved
    };

    #[cfg(target_os = "macos")]
    let should_configure = true;
    #[cfg(not(target_os = "macos"))]
    let should_configure = resolved.is_some();

    if should_configure {
        let mut builder = ThreadPoolBuilder::new();
        if let Some(t) = resolved {
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

// ---------------------------------------------------------------------------
// BASALT Phase 3 forced-decode harness helpers (basalt_eval_protocol.md §5.1) —
// `camelid gemma4-generate --force-tokens/--dump-step-logits` records. Pure
// CLI-side plumbing: no engine math lives here.
// ---------------------------------------------------------------------------

/// One recorded step of the §5.1 harness: the argmax (id, logit) of the step's
/// full logit vector plus its top-32 excerpt (§5.3 bundle convention), and the
/// teacher-forced token when in forced mode.
#[derive(Debug, Serialize)]
struct Gemma4StepRecord {
    step: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    forced_id: Option<u32>,
    argmax_id: u32,
    argmax_logit: f32,
    /// Top-32 (token id, logit) pairs, logit-descending (ties: lower id first).
    top32: Vec<(u32, f32)>,
}

/// The harness `meta.json` (and forced-mode stdout) document.
#[derive(Debug, Serialize)]
struct Gemma4StepMeta {
    protocol: &'static str,
    /// "forced" (--force-tokens) or "greedy" (--dump-step-logits alone).
    mode: &'static str,
    model: String,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    vocab_size: usize,
    step_count: usize,
    logits_dtype: &'static str,
    logits_file_pattern: &'static str,
    steps: Vec<Gemma4StepRecord>,
}

/// Parse a `--force-tokens` file: either one JSON array of token ids
/// (`[5, 6, 7]`) or newline-separated decimal ids (blank lines, CR, and a BOM
/// tolerated). Empty files are an error — a forced decode with zero steps is
/// always a harness mistake.
fn parse_forced_tokens(text: &str) -> Result<Vec<u32>, String> {
    let trimmed = text.trim_start_matches('\u{feff}').trim();
    if trimmed.is_empty() {
        return Err("forced-token file is empty".into());
    }
    if trimmed.starts_with('[') {
        let ids = serde_json::from_str::<Vec<u32>>(trimmed)
            .map_err(|e| format!("forced-token JSON parse failed: {e}"))?;
        // The JSON branch must not bypass the emptiness guard above: `[]`
        // parses fine but a zero-step forced decode is always a harness mistake.
        if ids.is_empty() {
            return Err("forced-token list is empty".into());
        }
        return Ok(ids);
    }
    trimmed
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.parse::<u32>()
                .map_err(|e| format!("bad forced token id {l:?}: {e}"))
        })
        .collect()
}

/// Refuse forced token ids outside the model's vocab (BASALT Amendment 3 review
/// fix): an out-of-range id would panic (or silently mis-embed) deep inside the
/// forward step, so it is validated at the CLI call site — the first point where
/// the vocab size is known post-load. Names the offending id and step.
fn validate_forced_token_vocab(ids: &[u32], vocab: usize) -> Result<(), String> {
    match ids.iter().enumerate().find(|(_, &id)| id as usize >= vocab) {
        Some((step, &id)) => Err(format!(
            "forced token id {id} at step {step} is out of range for this model's \
             vocab size {vocab}"
        )),
        None => Ok(()),
    }
}

/// Refuse a non-empty existing `--dump-step-logits` directory (BASALT
/// Amendment 3 review fix): `step_<i>.bin` files from a previous run would
/// silently mix with this run's dumps and corrupt the §5.3 exact-KL input.
/// A missing directory is fine (created after this check); an existing empty
/// directory is fine; anything else is a named error listing the offending dir.
fn ensure_dump_dir_empty(dir: &std::path::Path) -> Result<(), String> {
    match std::fs::read_dir(dir) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                Err(format!(
                    "--dump-step-logits directory {} already exists and is not empty; \
                     refusing to mix step dumps with pre-existing files (pass a fresh \
                     or empty directory)",
                    dir.display()
                ))
            } else {
                Ok(())
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        // Exists but is not a listable directory (e.g. a plain file).
        Err(e) => Err(format!(
            "--dump-step-logits {} is not usable as a dump directory: {e}",
            dir.display()
        )),
    }
}

/// Per-step argmax (id, logit) with `Gemma4Runtime::generate_greedy`'s EXACT
/// tie convention (`max_by` + `partial_cmp`: the last of equal maxima wins), so
/// the recorded argmax is the token the greedy decoder would emit at this step.
fn greedy_argmax(logits: &[f32]) -> (u32, f32) {
    let (i, v) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .expect("non-empty logits");
    (i as u32, *v)
}

/// Top-`n` (id, logit) pairs by logit descending, ties broken by lower id —
/// deterministic (`total_cmp`) and independent of the argmax tie convention
/// above (the two can differ on an exact tie; the argmax field is authoritative
/// for greedy-parity questions).
fn top_n_logits(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let take = n.min(logits.len());
    if take == 0 {
        return Vec::new();
    }
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    let cmp = |a: &u32, b: &u32| {
        logits[*b as usize]
            .total_cmp(&logits[*a as usize])
            .then(a.cmp(b))
    };
    if take < ids.len() {
        ids.select_nth_unstable_by(take - 1, cmp);
        ids.truncate(take);
    }
    ids.sort_unstable_by(cmp);
    ids.into_iter().map(|i| (i, logits[i as usize])).collect()
}

/// Write one step's FULL logit vector as raw little-endian f32 bytes
/// (`step_<i>.bin` — the §5.3 exact-KL input; dumps are temporary, the bundle
/// keeps the meta.json top-32 excerpts).
fn write_step_logits(dir: &std::path::Path, step: usize, logits: &[f32]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(logits.len() * 4);
    for v in logits {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join(format!("step_{step}.bin")), buf)
}

#[cfg(test)]
mod basalt_forced_decode_tests {
    use super::*;

    fn on_cli_test_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("cli-parse-test".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(test)
            .expect("spawn CLI parse test")
            .join()
            .expect("CLI parse test panicked");
    }

    #[test]
    fn gemma4_generate_parses_forced_decode_flags() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "gemma4-generate",
                "model.gguf",
                "--force-tokens",
                "toks.txt",
                "--dump-step-logits",
                "dumps",
                "--max-tokens",
                "8",
            ])
            .expect("parse");
            match cli.command {
                Some(Command::Gemma4Generate {
                    path,
                    max_tokens,
                    force_tokens,
                    dump_step_logits,
                    ..
                }) => {
                    assert_eq!(path, PathBuf::from("model.gguf"));
                    assert_eq!(max_tokens, 8);
                    assert_eq!(force_tokens, Some(PathBuf::from("toks.txt")));
                    assert_eq!(dump_step_logits, Some(PathBuf::from("dumps")));
                }
                other => panic!("expected Gemma4Generate, got {other:?}"),
            }
        });
    }

    #[test]
    fn gemma4_generate_harness_flags_default_off() {
        on_cli_test_stack(|| {
            let cli =
                Cli::try_parse_from(["camelid", "gemma4-generate", "model.gguf"]).expect("parse");
            match cli.command {
                Some(Command::Gemma4Generate {
                    prompt,
                    max_tokens,
                    force_tokens,
                    dump_step_logits,
                    ..
                }) => {
                    // Default behavior unchanged: no harness flags, prior defaults intact.
                    assert_eq!(force_tokens, None);
                    assert_eq!(dump_step_logits, None);
                    assert_eq!(prompt, "The capital of France is");
                    assert_eq!(max_tokens, 24);
                }
                other => panic!("expected Gemma4Generate, got {other:?}"),
            }
        });
    }

    #[test]
    fn forced_token_file_parses_newline_and_json_forms() {
        assert_eq!(parse_forced_tokens("5\n6\n7\n").unwrap(), vec![5, 6, 7]);
        assert_eq!(parse_forced_tokens("[5, 6, 7]").unwrap(), vec![5, 6, 7]);
        // CRLF + blank lines + BOM tolerated (Windows-authored token files).
        assert_eq!(
            parse_forced_tokens("\u{feff}5\r\n\r\n6\r\n").unwrap(),
            vec![5, 6]
        );
        assert!(parse_forced_tokens("").is_err());
        assert!(parse_forced_tokens("   \n  ").is_err());
        assert!(parse_forced_tokens("notanid").is_err());
        assert!(parse_forced_tokens("[1, -2]").is_err());
        // Review fix: the JSON branch must not bypass the emptiness guard.
        for empty_json in ["[]", "[ ]"] {
            match parse_forced_tokens(empty_json) {
                Err(e) => assert_eq!(e, "forced-token list is empty", "input {empty_json:?}"),
                Ok(ids) => panic!("empty JSON list {empty_json:?} must error, got {ids:?}"),
            }
        }
    }

    #[test]
    fn forced_token_vocab_validation_names_the_offending_id() {
        // In-range ids (including vocab-1) pass.
        validate_forced_token_vocab(&[0, 5, 261_143], 261_144).expect("in-range ids admit");
        validate_forced_token_vocab(&[], 16).expect("empty list is vacuously in range");
        // First offending id + its step are named.
        let err = validate_forced_token_vocab(&[3, 16, 2], 16).expect_err("16 >= vocab 16");
        assert!(err.contains("forced token id 16"), "{err}");
        assert!(err.contains("at step 1"), "{err}");
        assert!(err.contains("vocab size 16"), "{err}");
        // Boundary: id == vocab is out of range (ids are 0-based).
        assert!(validate_forced_token_vocab(&[8], 8).is_err());
    }

    #[test]
    fn dump_dir_check_refuses_non_empty_existing_directory() {
        let root = tempfile::tempdir().expect("tempdir");

        // Nonexistent path: fine (created later by create_dir_all).
        let fresh = root.path().join("fresh-dumps");
        ensure_dump_dir_empty(&fresh).expect("missing dir is usable");

        // Existing but empty: fine.
        let empty = root.path().join("empty-dumps");
        std::fs::create_dir(&empty).expect("mkdir");
        ensure_dump_dir_empty(&empty).expect("empty dir is usable");

        // Existing with contents: named refusal listing the offending dir.
        let dirty = root.path().join("dirty-dumps");
        std::fs::create_dir(&dirty).expect("mkdir");
        std::fs::write(dirty.join("step_0.bin"), b"stale").expect("write");
        let err = ensure_dump_dir_empty(&dirty).expect_err("non-empty dir must refuse");
        assert!(err.contains("already exists and is not empty"), "{err}");
        assert!(err.contains(&dirty.display().to_string()), "{err}");

        // A plain file at the path is also a named error, not a panic.
        let file_path = root.path().join("not-a-dir");
        std::fs::write(&file_path, b"x").expect("write");
        let err = ensure_dump_dir_empty(&file_path).expect_err("file path must refuse");
        assert!(err.contains("not usable as a dump directory"), "{err}");
    }

    #[test]
    fn step_record_helpers_are_deterministic() {
        let logits = [0.5f32, 2.5, -1.0, 2.5, 0.0];
        // generate_greedy's max_by(partial_cmp) keeps the LAST of equal maxima.
        assert_eq!(greedy_argmax(&logits), (3, 2.5));
        // top-n orders logit-descending with lower-id-first ties.
        assert_eq!(top_n_logits(&logits, 3), vec![(1, 2.5), (3, 2.5), (0, 0.5)]);
        // n larger than vocab clamps.
        assert_eq!(top_n_logits(&logits, 64).len(), 5);
        assert_eq!(top_n_logits(&[], 32), Vec::new());
    }
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
    fn serve_nocopy_default_only_with_active_wire_resident_stack() {
        // Default on: fresh (unset) + full wire-resident stack.
        assert!(should_default_serve_nocopy(false, true, true, true));
        // User set it either way (incl. an explicit =0): never override.
        assert!(!should_default_serve_nocopy(true, true, true, true));
        // Speculative decoding turns resident decode off -> stay off (its CPU
        // repack plan needs materialized blocks, not wire pages).
        assert!(!should_default_serve_nocopy(false, false, true, true));
        // Any wire-stack component off -> the wire kernels can't consume pages.
        assert!(!should_default_serve_nocopy(false, true, false, true));
        assert!(!should_default_serve_nocopy(false, true, true, false));
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
