//! `camelid chat` — an interactive terminal chat client for the local Camelid
//! engine.
//!
//! Two front ends share one [`session::Session`] core (state, sampling, request
//! shape — no I/O):
//! - [`tui`]: a full-screen ratatui app (scrollable chat, status bar, sidebar,
//!   modal picker) — the default on an interactive terminal.
//! - [`inline`]: a scrollback-friendly line REPL — used for `--plain`, pipes,
//!   and non-TTY contexts (the lane the smoke scripts and tests drive).
//!
//! Both stream `/v1/chat/completions` over the same audited HTTP/SSE client, so
//! terminal output matches the validated lane. The picker is derived from the
//! `/api/capabilities` ledger at runtime (supported rows only); pointing
//! `--model` at an unsupported GGUF is refused with the engine's typed error.
//! See `DECISIONS.md` D6 and `RECON_CHAT.md`.

pub(crate) mod agent;
mod agent_bench;
mod agent_eval;
mod agent_orchestration;
mod agent_session;
mod agent_syscap;
mod agent_tui;
mod audit;
mod banner;
mod checkpoint;
pub(crate) mod client;
mod clipboard;
mod inline;
mod markdown;
mod mcp;
mod models;
mod palette;
mod plan;
pub(crate) mod semantic_search;
mod server;
mod session;
mod shell_sandbox;
mod subagent;
mod term_guard;
mod theme;
mod tool_parse;
mod tools;
mod tui;
#[cfg(windows)]
mod win_clipboard;
#[cfg(windows)]
mod win_console;
#[cfg(windows)]
mod win_input;
#[cfg(windows)]
mod win_job;
#[cfg(windows)]
mod win_uia;
pub(crate) mod workspace_bridge;
mod workspace_cli;
pub(crate) mod workspace_memory;

pub use workspace_cli::{run as run_workspace_cli, WorkspaceCliAction, WorkspaceCliOptions};

use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use client::Client;
use server::ServerHandle;
use session::{LoadResult, Session, Settings};

pub(crate) const VERSION: &str = match option_env!("CAMELID_GIT_DESCRIBE") {
    Some(describe) => describe,
    None => env!("CARGO_PKG_VERSION"),
};

/// Parsed `camelid chat` flags.
pub struct ChatOptions {
    pub model: Option<PathBuf>,
    pub addr: SocketAddr,
    pub system: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
    pub no_stream: bool,
    pub models_dir: PathBuf,
    /// Force the inline line REPL instead of the full-screen TUI.
    pub plain: bool,
    /// Enter agent mode (tool-calling loop) instead of plain chat.
    pub agent: bool,
    /// Sandbox root for agent tools (default: cwd).
    pub workdir: Option<PathBuf>,
    pub max_steps: usize,
    pub auto_approve: bool,
    /// `--yolo` (unattended): auto-approve EXEC tools too so the agent runs a
    /// whole task without prompting. Refused under production.
    pub yolo: bool,
    pub allow_net: bool,
    /// `--allow-fs`: agent file tools may read/write anywhere on disk (still
    /// approval-gated), not just under the workspace root.
    pub allow_fs: bool,
    /// `--allow-mcp`: permit MCP support. No workspace command is executed
    /// without a matching entry in `trust_mcp_servers`.
    pub allow_mcp: bool,
    /// Server names explicitly trusted on the command line. Each corresponding
    /// workspace-declared command starts immediately during agent startup.
    pub trust_mcp_servers: Vec<String>,
    pub shell_timeout: u64,
    /// Opt-in thinking mode (`chat --enable-thinking`): the model emits its own
    /// `<think>…</think>` reasoning. NOT parity-locked (leading-trace lane only).
    pub enable_thinking: bool,
    /// Audit webhook URL (`--audit-webhook` / `CAMELID_AUDIT_WEBHOOK`). When unset,
    /// the agent uses the no-op sink and emits nothing.
    pub audit_webhook: Option<String>,
    /// `run_shell` confinement: `disabled` | `sandboxed` (default) | `unrestricted`.
    pub shell_sandbox: String,
    /// Headless one-shot (`camelid agent exec`): run this goal to completion and
    /// exit, instead of opening a REPL. Implies `agent` + `plain`.
    pub exec_goal: Option<String>,
    /// Secret-safe benchmark trace emitted only by headless `agent exec`.
    pub benchmark_events: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgentRuntimeBudget {
    /// Total prompt-plus-reply budget passed to the exact prompt fitter.
    context_tokens: u32,
    /// Reply allowance sent on every generation request.
    reply_tokens: u32,
}

/// Resolve the Full coding-agent budget against every live runtime ceiling.
/// The server enforces prompt and generation limits independently, while the
/// model and validated-agent limits are total-context ceilings, so the fitter
/// receives `min(model, validated, server_prompt + reserved_reply)`.
fn derive_agent_runtime_budget(
    active_model_context: Option<u32>,
    server_prompt_ceiling: usize,
    server_generation_ceiling: u32,
    requested_reply_tokens: u32,
) -> Result<AgentRuntimeBudget, String> {
    let model_context = active_model_context.ok_or_else(|| {
        "agent mode cannot determine the active model's runtime context length".to_string()
    })?;
    if server_prompt_ceiling == 0 || server_generation_ceiling == 0 {
        return Err(
            "agent mode requires max_prompt_tokens and max_generation_tokens from /v1/health; \
             restart or upgrade the local Camelid server so runtime ceilings are available"
                .to_string(),
        );
    }
    if requested_reply_tokens == 0 {
        return Err("agent mode requires a positive reply-token allowance".to_string());
    }

    let reply_tokens = requested_reply_tokens.min(server_generation_ceiling);
    let prompt_ceiling = u32::try_from(server_prompt_ceiling).unwrap_or(u32::MAX);
    let context_tokens = model_context
        .min(agent::AGENT_VALIDATED_CTX)
        .min(prompt_ceiling.saturating_add(reply_tokens));
    if context_tokens <= reply_tokens {
        return Err(format!(
            "agent mode has no prompt room: the effective {context_tokens}-token context does \
             not exceed the {reply_tokens}-token reply allowance"
        ));
    }

    Ok(AgentRuntimeBudget {
        context_tokens,
        reply_tokens,
    })
}

/// Entry point for the `Chat` subcommand. Returns a process exit code (0 = ok,
/// non-zero for the typed unsupported-state backstop) so the caller can exit
/// after this function's `ServerHandle` has torn down any spawned server.
pub fn run_chat(opts: ChatOptions) -> anyhow::Result<i32> {
    init_terminal();
    install_sigint_handler();

    let client = Client::new(opts.addr);
    let server = ServerHandle::ensure(opts.addr, &client)?;
    let spawned = server.spawned();

    let settings = Settings {
        temperature: opts.temperature,
        top_p: opts.top_p,
        top_k: opts.top_k,
        max_tokens: opts.max_tokens,
        seed: opts.seed,
        stream: !opts.no_stream,
        enable_thinking: opts.enable_thinking,
    };
    let mut session = Session::new(client, opts.models_dir, settings, opts.system);

    // --model backstop: load + classify before any UI, so an unsupported GGUF
    // exits with the typed error and no screen takeover. Loading a cold GGUF can
    // take several seconds, so give feedback before the UI takes the screen. A
    // known supported GGUF is labeled with its ledger id (so posture + the agent
    // tool-capable gate match), exactly like the picker.
    if let Some(model) = &opts.model {
        eprintln!("Loading {} …", model.display());
        let label = catalog_label_for(model);
        let posture = label.as_ref().map(|_| "supported");
        match session.load_model_file(model, label.as_deref(), posture)? {
            LoadResult::Loaded => {}
            LoadResult::Unsupported(message) => {
                eprintln!("{message}");
                return Ok(1);
            }
        }
    }

    // Agent mode: a tool-calling loop (line renderer), gated to tool-capable rows.
    if opts.agent {
        if !session.has_model() {
            eprintln!("agent mode needs a model — pass --model <gguf>");
            return Ok(2);
        }
        let Some(health) = session.client().health() else {
            eprintln!("agent mode could not read runtime ceilings from /v1/health");
            return Ok(2);
        };
        let runtime_budget = match derive_agent_runtime_budget(
            session.active_ctx,
            health.max_prompt_tokens,
            health.max_generation_tokens,
            opts.max_tokens,
        ) {
            Ok(budget) => budget,
            Err(error) => {
                eprintln!("{error}");
                return Ok(2);
            }
        };
        if runtime_budget.reply_tokens < opts.max_tokens {
            eprintln!(
                "agent reply allowance: requested {} tokens, capped to the server's {}-token generation ceiling",
                opts.max_tokens, runtime_budget.reply_tokens
            );
        }
        let shell_sandbox = match opts.shell_sandbox.parse::<shell_sandbox::ShellSandbox>() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{e}");
                return Ok(2);
            }
        };
        let benchmark_exec = opts.benchmark_events.is_some();
        let cfg = agent::AgentConfig {
            workdir: opts.workdir.unwrap_or_else(|| PathBuf::from(".")),
            max_steps: opts.max_steps,
            auto_approve: opts.auto_approve,
            yolo: opts.yolo,
            allow_net: opts.allow_net,
            allow_fs: opts.allow_fs,
            shell_timeout: std::time::Duration::from_secs(opts.shell_timeout),
            max_tokens: runtime_budget.reply_tokens,
            temperature: opts.temperature,
            audit: audit::sink_from_config(opts.audit_webhook.as_deref()),
            shell_sandbox,
            tool_profile: if benchmark_exec {
                tools::ToolProfile::BenchmarkShared
            } else {
                tools::ToolProfile::Full
            },
            // Exact prompt-plus-reply budget after intersecting the active
            // model, validated agent lane, and this server process's limits.
            ctx_budget: Some(runtime_budget.context_tokens),
        };
        // MCP servers, if the user opted in. A broken MCP config costs you MCP,
        // not your session, so problems are reported and the agent still runs.
        if opts.allow_mcp {
            match tools::Sandbox::new(&cfg.workdir, cfg.allow_net, cfg.shell_timeout) {
                Ok(sb) => {
                    let native: Vec<String> = tools::specs(cfg.allow_net, shell_sandbox)
                        .into_iter()
                        .map(|t| t.name)
                        .collect();
                    match mcp::configure(
                        &sb,
                        true,
                        agent::is_production(),
                        &native,
                        &opts.trust_mcp_servers,
                        &session::CANCEL,
                    ) {
                        Ok(0) => eprintln!(
                            "--allow-mcp: no MCP tools loaded (no {} at the workspace root?)",
                            mcp::CONFIG_FILE
                        ),
                        Ok(n) => eprintln!("MCP: {n} tool(s) loaded — each is approval-gated"),
                        Err(e) => eprintln!("MCP: {e}"),
                    }
                }
                Err(e) => eprintln!("MCP: workspace unavailable: {e}"),
            }
        } else if !opts.trust_mcp_servers.is_empty() {
            eprintln!(
                "MCP: --trust-mcp-server requires --allow-mcp; no workspace command was started"
            );
        }

        // Headless one-shot: no REPL, tri-state exit, answer on stdout.
        if let Some(goal) = opts.exec_goal.as_deref() {
            let code = agent::run_exec(
                &mut session,
                opts.addr,
                cfg,
                goal,
                opts.benchmark_events.as_deref(),
            );
            mcp::shutdown();
            return code;
        }

        // Full-screen TUI agent on a real terminal (default); the line renderer
        // is the fallback for --plain, pipes, and non-TTY runs (smoke/tests).
        let interactive = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
        let code = if interactive && !opts.plain {
            agent_tui::run(&mut session, opts.addr, cfg)
        } else {
            agent::run_agent(&mut session, opts.addr, cfg)
        };
        mcp::shutdown();
        return code;
    }

    // Full-screen TUI when we have a real terminal on both ends and the user did
    // not ask for plain mode; otherwise the inline REPL.
    let interactive = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
    if interactive && !opts.plain {
        tui::run(&mut session, opts.addr, spawned)?;
    } else {
        inline::run(&mut session, opts.addr, spawned)?;
    }
    Ok(0)
}

/// If `model`'s filename matches a curated-catalog row, return that catalog id
/// (= the ledger row id) so a `--model`-loaded supported GGUF carries its ledger
/// identity (posture + agent tool-capable gate).
fn catalog_label_for(model: &std::path::Path) -> Option<String> {
    let name = model.file_name()?.to_str()?;
    crate::api::curated_catalog()
        .into_iter()
        .find(|item| item.filename == name)
        .map(|item| item.catalog_id.to_string())
}

/// Parsed `camelid agent-eval` flags.
pub struct AgentEvalOptions {
    pub model: PathBuf,
    pub addr: SocketAddr,
    pub load_timeout: u64,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub receipt_dir: PathBuf,
}

/// Entry for the `agent-eval` subcommand: the tool-capability promotion harness.
/// Returns PASS(0) / FAIL(1) / INCONCLUSIVE(3).
pub fn run_agent_eval(opts: AgentEvalOptions) -> anyhow::Result<i32> {
    agent_eval::run(agent_eval::EvalConfig {
        addr: opts.addr,
        model: opts.model,
        load_timeout: opts.load_timeout,
        max_steps: opts.max_steps,
        max_tokens: opts.max_tokens,
        receipt_dir: opts.receipt_dir,
    })
}

/// Parsed `camelid agent-syscap-eval` flags.
pub struct AgentSyscapOptions {
    pub receipt_dir: PathBuf,
}

/// Entry for the `agent-syscap-eval` subcommand: the Phase-1 Windows
/// system-control gate. Returns PASS(0) / FAIL(1) / INCONCLUSIVE(3) and emits a
/// sealed `camelid.agent-syscap-receipt/v1`.
pub fn run_agent_syscap_eval(opts: AgentSyscapOptions) -> anyhow::Result<i32> {
    agent_syscap::run(agent_syscap::SyscapConfig {
        receipt_dir: opts.receipt_dir,
    })
}

/// Entry for the hidden `__subagent` worker subcommand: run one scoped agent loop
/// described by `task_file` and write its result file. Returns 0/1/3.
pub fn run_subagent_worker(task_file: &std::path::Path) -> anyhow::Result<i32> {
    subagent::run_worker(task_file)
}

/// Parsed `camelid agent-orchestration-eval` flags.
pub struct AgentOrchestrationOptions {
    pub receipt_dir: PathBuf,
    pub model: Option<PathBuf>,
    pub addr: SocketAddr,
    pub load_timeout: u64,
}

/// Entry for the `agent-orchestration-eval` subcommand: the orchestration gate.
/// Without `--model` it runs the canned rung-2 mechanics battery; with `--model`
/// it runs the rung-3 real-model round-trip. Returns 0/1/3.
pub fn run_agent_orchestration_eval(opts: AgentOrchestrationOptions) -> anyhow::Result<i32> {
    agent_orchestration::run(agent_orchestration::OrchestrationConfig {
        receipt_dir: opts.receipt_dir,
        model: opts.model,
        addr: opts.addr,
        load_timeout: opts.load_timeout,
    })
}

/// Parsed `camelid agent-orchestration-bench` flags.
pub struct AgentOrchestrationBenchOptions {
    pub receipt_dir: PathBuf,
    pub model: Option<PathBuf>,
    pub addr: SocketAddr,
    pub load_timeout: u64,
}

/// Entry for the `agent-orchestration-bench` subcommand: the rung-4 wall-clock
/// measurement (concurrent vs sequential subagents) → sealed bench receipt.
pub fn run_agent_orchestration_bench(opts: AgentOrchestrationBenchOptions) -> anyhow::Result<i32> {
    agent_bench::run(agent_bench::BenchConfig {
        receipt_dir: opts.receipt_dir,
        model: opts.model,
        addr: opts.addr,
        load_timeout: opts.load_timeout,
    })
}

extern "C" fn on_sigint(_signal: libc::c_int) {
    session::CANCEL.store(true, Ordering::SeqCst);
}

/// Install a SIGINT handler that flips the cancel flag (used by the inline
/// stream loop). The TUI runs in raw mode where Ctrl-C arrives as a key event,
/// so it cancels through its event loop instead.
fn install_sigint_handler() {
    unsafe {
        libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t);
    }
}

/// Prepare the terminal for the line-mode renderers (inline + agent). On Windows
/// this enables ANSI escape processing and a UTF-8 code page so colors and glyphs
/// render the way they do on macOS/Linux; the full-screen TUI already gets this
/// from crossterm. A no-op on Unix, where terminals handle ANSI + UTF-8 natively.
#[cfg(windows)]
fn init_terminal() {
    win_console::init();
}
#[cfg(not(windows))]
fn init_terminal() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_server_ceilings_bound_full_agent_prompt_and_reply() {
        let budget = derive_agent_runtime_budget(Some(8_192), 384, 96, 512).unwrap();
        assert_eq!(budget.reply_tokens, 96);
        assert_eq!(budget.context_tokens, 480);
        assert_eq!(budget.context_tokens - budget.reply_tokens, 384);
    }

    #[test]
    fn model_and_validated_contexts_remain_hard_agent_ceilings() {
        let model_limited = derive_agent_runtime_budget(Some(320), 10_000, 1_000, 64).unwrap();
        assert_eq!(model_limited.context_tokens, 320);
        assert_eq!(model_limited.reply_tokens, 64);

        let validated_limited =
            derive_agent_runtime_budget(Some(32_768), 32_768, 32_768, 128).unwrap();
        assert_eq!(validated_limited.context_tokens, agent::AGENT_VALIDATED_CTX);
    }

    #[test]
    fn missing_or_impossible_runtime_budget_fails_closed() {
        assert!(derive_agent_runtime_budget(None, 384, 96, 64).is_err());
        assert!(derive_agent_runtime_budget(Some(512), 0, 96, 64).is_err());
        assert!(derive_agent_runtime_budget(Some(512), 384, 0, 64).is_err());
        assert!(derive_agent_runtime_budget(Some(64), 384, 96, 96).is_err());
    }
}
