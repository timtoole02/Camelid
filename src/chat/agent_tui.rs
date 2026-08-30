//! Full-screen ratatui front end for agent mode (`camelid chat --agent`).
//!
//! The line renderer ([`super::agent::run_agent`]) stays the fallback for pipes,
//! `--plain`, and non-TTY runs; this is the interactive screen and shares the
//! same agent engine ([`super::agent::run_loop`] with its
//! `ModelDriver`/`Approver`/`Reporter` traits) and the same visual language as
//! the chat TUI ([`super::tui`]): a bordered transcript, an input box, a status
//! line, and a settings sidebar.
//!
//! The bounded plan-act-observe loop runs on a background thread (like the chat
//! TUI's streaming thread) and forwards transcript events over a channel; the
//! redraw loop renders them. A gated tool's approval is a **modal in the redraw
//! loop**: the loop thread sends a `NeedApproval` event carrying a reply channel
//! and blocks until a keypress (`y`/`n`/`a`/`q`) sends the decision back — the
//! "modal approvals in the redraw loop" follow-up noted in `agent.rs` /
//! `DECISIONS.md` D9. The main thread never blocks (it polls events on a
//! timeout), so there is no deadlock.

use std::io::Stdout;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    self, EnableBracketedPaste, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{enable_raw_mode, EnterAlternateScreen};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::{Frame, Terminal};

use super::agent::{
    self, AgentConfig, AgentMsg, Approver, Decision, LiveDriver, LoopEnd, Policy, Reporter,
};
use super::session::{self, Session};
use super::shell_sandbox::ShellSandbox;
use super::theme::Theme;
use super::tools::{Action, Sandbox, ToolOutcome};
use super::{banner, markdown};

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
/// Soft red for tool *error* results (the themes carry no dedicated error role).
const ERR: Color = Color::Indexed(167);

/// Everything the agent loop owns for one session; moved into the per-goal worker
/// thread to run a goal and handed back (via [`Ev::Done`]) when it ends, so the
/// `a` ("always allow") grants in `policy` persist across goals.
struct Engine {
    driver: LiveDriver,
    sandbox: Sandbox,
    cfg: AgentConfig,
    policy: Policy,
    /// The agent transcript, carried across goals so "now do the same for the
    /// other file" refers to something. Also what /save persists and /resume
    /// restores. Seeded per goal through [`agent::seed_history`], which
    /// rebuilds the System message fresh and drops any carried one.
    transcript: Vec<AgentMsg>,
}

/// A transcript entry, rendered to width-aware lines each frame.
enum Entry {
    Goal(String),
    Model(String),
    ToolCall(String),
    ToolResult { ok: bool, body: String },
    Notice(String),
}

/// Events from the worker thread to the redraw loop.
enum Ev {
    /// A live token delta from the model (streamed into the `live` buffer).
    Delta(String),
    Model(String),
    ToolCall(String),
    ToolResult {
        ok: bool,
        body: String,
    },
    Notice(String),
    NeedApproval {
        risk: String,
        detail: String,
        reply: Sender<Decision>,
    },
    Done {
        end: LoopEnd,
        engine: Box<Engine>,
    },
}

/// A gated action waiting on the operator's keypress.
struct PendingApproval {
    risk: String,
    detail: String,
    reply: Sender<Decision>,
}

// --- channel-backed Reporter + Approver (run on the worker thread) ----------

struct ChannelReporter {
    tx: Sender<Ev>,
}
impl Reporter for ChannelReporter {
    fn model_text(&mut self, text: &str) {
        let _ = self.tx.send(Ev::Model(text.to_string()));
    }
    fn tool_call(&mut self, line: &str) {
        let _ = self.tx.send(Ev::ToolCall(line.to_string()));
    }
    fn tool_result(&mut self, _name: &str, outcome: &ToolOutcome) {
        let _ = self.tx.send(Ev::ToolResult {
            ok: !outcome.is_err(),
            body: outcome.text().to_string(),
        });
    }
    fn notice(&mut self, text: &str) {
        let _ = self.tx.send(Ev::Notice(text.to_string()));
    }
}

struct ChannelApprover {
    tx: Sender<Ev>,
}
impl Approver for ChannelApprover {
    fn approve(&mut self, action: &Action, sandbox: &Sandbox) -> Decision {
        let (reply, wait) = std::sync::mpsc::channel();
        let sent = self.tx.send(Ev::NeedApproval {
            risk: action.risk().label().to_string(),
            detail: action.approval_detail(sandbox),
            reply,
        });
        if sent.is_err() {
            return Decision::Abort; // UI gone
        }
        // Block until the redraw loop relays the operator's keypress. The main
        // thread keeps drawing/polling, so this never deadlocks.
        wait.recv().unwrap_or(Decision::Abort)
    }
}

// --- entry ------------------------------------------------------------------

/// Run agent mode in the full-screen TUI. Mirrors [`super::agent::run_agent`]'s
/// preflight (capability gate, approval policy, sandbox, subagent wiring) then
/// drives the loop through the redraw loop. Returns a process exit code.
pub fn run(session: &mut Session, addr: SocketAddr, cfg: AgentConfig) -> anyhow::Result<i32> {
    // Capability gate (constraint 3): tool-capable supported row only.
    if !session.active_tool_capable() {
        let rows = session.tool_capable_rows();
        eprintln!(
            "agent mode requires a tool-capable supported model. The active model{} is not \
             marked tool_capable in the compatibility ledger (/api/capabilities).{}",
            session
                .active_id
                .as_deref()
                .map(|id| format!(" '{id}'"))
                .unwrap_or_default(),
            if rows.is_empty() {
                String::new()
            } else {
                format!(" Tool-capable rows: {}.", rows.join(", "))
            }
        );
        return Ok(2);
    }
    let Some(agent_context_budget) = cfg.ctx_budget else {
        eprintln!("agent mode requires an effective runtime context budget");
        return Ok(2);
    };
    // Approval policy: --auto-approve is refused (fail closed) under production.
    let policy = match agent::resolve_policy(cfg.auto_approve, cfg.yolo, agent::is_production()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return Ok(2);
        }
    };
    let sandbox = Sandbox::new(&cfg.workdir, cfg.allow_net, cfg.shell_timeout)?
        .with_shell_mode(cfg.shell_sandbox)
        .with_fs_unrestricted(cfg.allow_fs);

    // Enable subagent orchestration (children share this serve + inherit gates).
    let _subagent_session =
        super::subagent::configure_scoped(super::subagent::SubagentConfig::for_session(
            addr,
            session.active_id.clone().unwrap_or_default(),
            session.active_gguf_sha256().unwrap_or_default().to_string(),
            session.active_family(),
            agent_context_budget,
            cfg.max_tokens,
            cfg.auto_approve,
            cfg.shell_sandbox,
        ));

    super::checkpoint::clear();

    let mut driver = LiveDriver::new(session, cfg.max_tokens, cfg.temperature);
    driver.set_context_budget(cfg.ctx_budget);
    driver.set_stream_timeout(agent::AGENT_MODEL_STEP_TIMEOUT);
    let engine = Box::new(Engine {
        driver,
        sandbox,
        cfg,
        policy,
        transcript: Vec::new(),
    });

    enable_raw_mode()?;
    // W6: from here the terminal is raw — arm the restore guard before anything
    // else can fail, and route panics through the chaining hook, so no `?` bail
    // or TUI-thread panic can strand the console (cmd.exe has no `reset`).
    super::term_guard::install_panic_hook();
    let guard = super::term_guard::TerminalGuard::arm();
    let mut out = std::io::stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture)?;
    // W5: bracketed paste is Unix-only value (crossterm 0.28 never constructs
    // Event::Paste on Windows) and could in principle fail on an exotic host —
    // non-fatal by design, on its own write so it can never abort setup. The
    // guard above covers the `?`s regardless.
    let _ = execute!(out, EnableBracketedPaste);
    let mut terminal = Terminal::new(CrosstermBackend::new(out))?;

    let mut app = App::new(session, addr, engine);
    let result = app.run(&mut terminal);

    // Normal exit: the guard restores (same sequence the straight-line teardown
    // used, best-effort) so `run`'s result is never masked by a teardown error.
    drop(guard);
    result.map(|()| 0)
}

struct App<'a> {
    session: &'a Session,
    addr: SocketAddr,
    theme: Theme,
    // Static session facts (for the sidebar).
    workspace: String,
    max_steps: usize,
    shell_mode: ShellSandbox,
    allow_net: bool,
    allow_fs: bool,
    auto_approve: bool,
    yolo: bool,
    tool_count: usize,

    input: String,
    cursor: usize,
    scroll: usize,
    last_max_scroll: usize,
    follow: bool,
    sidebar: bool,
    status: String,
    frame: u64,

    transcript: Vec<Entry>,
    /// The in-progress model output, streamed token-by-token; committed to an
    /// `Entry::Model` (final answer) or discarded (it was a tool call) at step end.
    live: String,
    engine: Option<Box<Engine>>,
    rx: Option<Receiver<Ev>>,
    running: bool,
    started: Option<Instant>,
    approval: Option<PendingApproval>,

    in_hist: Vec<String>,
    hist_idx: Option<usize>,
    help: bool,
    quit: bool,
}

impl<'a> App<'a> {
    fn new(session: &'a Session, addr: SocketAddr, engine: Box<Engine>) -> Self {
        let workspace = engine.sandbox.root().display().to_string();
        let shell_mode = engine.cfg.shell_sandbox;
        let allow_net = engine.cfg.allow_net;
        let allow_fs = engine.cfg.allow_fs;
        let auto_approve = engine.cfg.auto_approve;
        let yolo = engine.cfg.yolo;
        let max_steps = engine.cfg.max_steps;
        let tool_count = super::tools::specs(allow_net, engine.sandbox.shell_mode()).len();
        App {
            session,
            addr,
            theme: Theme::Sandstorm,
            workspace,
            max_steps,
            shell_mode,
            allow_net,
            allow_fs,
            auto_approve,
            yolo,
            tool_count,
            input: String::new(),
            cursor: 0,
            scroll: usize::MAX,
            last_max_scroll: 0,
            follow: true,
            sidebar: true,
            status: "describe a goal · /tools /steps /subagents /stop · F1 help".into(),
            frame: 0,
            transcript: Vec::new(),
            live: String::new(),
            engine: Some(engine),
            rx: None,
            running: false,
            started: None,
            approval: None,
            in_hist: Vec::new(),
            hist_idx: None,
            help: false,
            quit: false,
        }
    }

    fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> anyhow::Result<()> {
        while !self.quit {
            self.frame = self.frame.wrapping_add(1);
            terminal.draw(|f| self.draw(f))?;
            self.poll_events();
            let timeout = if self.running { 80 } else { 120 };
            if event::poll(Duration::from_millis(timeout))? {
                match event::read()? {
                    Event::Key(key) if key.kind != KeyEventKind::Release => {
                        // W5: Windows terminals inject a paste as plain
                        // keystrokes (crossterm never yields Event::Paste
                        // there), so a multi-line paste arrives as text plus
                        // real Enter presses — each of which used to fire a
                        // goal. Near-simultaneous arrival is not human typing:
                        // if more input is already queued the instant an Enter
                        // is processed, it is a pasted newline, not a submit.
                        // Measured live (Phase 0): a 3-line paste fired 2
                        // goals. On Unix bracketed paste handles this and the
                        // flag is constantly false — behavior preserved.
                        let paste_burst = cfg!(windows)
                            && key.code == KeyCode::Enter
                            && event::poll(Duration::ZERO).unwrap_or(false);
                        self.on_key(key, paste_burst);
                    }
                    // A paste is text, not keystrokes: newlines inside it must
                    // not submit the goal mid-paste.
                    Event::Paste(text) => {
                        for c in text.chars() {
                            let c = if c == '\r' { '\n' } else { c };
                            self.input.insert(self.cursor, c);
                            self.cursor += c.len_utf8();
                        }
                    }
                    Event::Mouse(m) => match m.kind {
                        MouseEventKind::ScrollUp => self.scroll_up(3),
                        MouseEventKind::ScrollDown => self.scroll_down(3),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        // On quit, cancel any running goal and release a pending approval so the
        // detached worker unwinds instead of blocking on the approval channel.
        session::CANCEL.store(true, Ordering::SeqCst);
        if let Some(p) = self.approval.take() {
            let _ = p.reply.send(Decision::Abort);
        }
        Ok(())
    }

    // ---- worker events ---------------------------------------------------

    fn poll_events(&mut self) {
        // Drain into a buffer first so the `rx` borrow is released before
        // handle_ev (which needs `&mut self`).
        let mut events = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = self.rx.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(ev) => events.push(ev),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        for ev in events {
            self.handle_ev(ev);
        }
        // Disconnected with the loop still marked running means the worker died
        // without sending Done (a Done already clears rx + running).
        if disconnected && self.running {
            self.running = false;
            self.rx = None;
            self.approval = None;
            self.status = "agent stopped unexpectedly".into();
        }
    }

    fn handle_ev(&mut self, ev: Ev) {
        match ev {
            Ev::Delta(d) => {
                self.live.push_str(&d);
                if self.follow {
                    self.scroll = usize::MAX;
                }
            }
            Ev::Model(t) => {
                self.live.clear();
                self.push(Entry::Model(t));
            }
            Ev::ToolCall(l) => {
                // The live buffer held this step's raw tool-call syntax — drop it
                // and show the clean call line instead.
                self.live.clear();
                self.push(Entry::ToolCall(l));
            }
            Ev::ToolResult { ok, body } => self.push(Entry::ToolResult { ok, body }),
            Ev::Notice(t) => self.push(Entry::Notice(t)),
            Ev::NeedApproval {
                risk,
                detail,
                reply,
            } => {
                self.approval = Some(PendingApproval {
                    risk,
                    detail,
                    reply,
                });
            }
            Ev::Done { end, engine } => {
                self.engine = Some(engine);
                self.running = false;
                self.rx = None;
                self.approval = None;
                self.live.clear();
                // A final answer means the goal was met; close out any plan
                // steps the model left in-progress so the sidebar stops
                // showing "N/N done" short of the total on a successful run.
                if matches!(end, LoopEnd::Answered) {
                    super::plan::complete_all();
                }
                self.push(Entry::Notice(end_label(end).to_string()));
                self.status = "describe a goal · /tools /steps /subagents · F1 help".into();
            }
        }
    }

    fn push(&mut self, e: Entry) {
        self.transcript.push(e);
        if self.follow {
            self.scroll = usize::MAX;
        }
    }

    // ---- input / goals ---------------------------------------------------

    fn submit(&mut self) {
        let text = self.input.trim().to_string();
        self.input.clear();
        self.cursor = 0;
        if text.is_empty() {
            return;
        }
        self.in_hist.push(text.clone());
        self.hist_idx = None;
        if let Some(cmd) = text.strip_prefix('/') {
            self.command(cmd);
            return;
        }
        if self.running {
            self.status = "a goal is already running — /stop to cancel".into();
            return;
        }
        self.start_goal(text);
    }

    fn start_goal(&mut self, goal: String) {
        let Some(boxed) = self.engine.take() else {
            self.status = "agent engine unavailable".into();
            return;
        };
        // Move the engine out of its box so the worker borrows disjoint fields.
        let mut engine: Engine = *boxed;
        self.push(Entry::Goal(goal.clone()));
        let (tx, rx) = std::sync::mpsc::channel();
        session::CANCEL.store(false, Ordering::SeqCst);
        std::thread::spawn(move || {
            let tools = super::tools::specs(engine.cfg.allow_net, engine.sandbox.shell_mode());
            // Re-read per goal: the project file may be edited mid-session.
            let project = agent::load_project_context(&engine.sandbox);
            if engine.transcript.is_empty() {
                // A fresh session gets a fresh plan; a continuing one keeps
                // what it was carrying (a /resume restored it).
                super::plan::clear();
            }
            let system =
                agent::system_prompt_with_project(&engine.sandbox, &tools, project.as_ref());
            let mut history = agent::seed_history(&engine.transcript, system, &goal);
            let mut reporter = ChannelReporter { tx: tx.clone() };
            let mut approver = ChannelApprover { tx: tx.clone() };
            // Stream the model's tokens live into the redraw loop's `live` buffer.
            let delta_tx = tx.clone();
            engine.driver.set_delta_sink(Some(Box::new(move |d: &str| {
                let _ = delta_tx.send(Ev::Delta(d.to_string()));
            })));
            let end = agent::run_loop(
                &mut engine.driver,
                &mut approver,
                &mut reporter,
                &engine.sandbox,
                &engine.cfg,
                &session::CANCEL,
                &mut engine.policy,
                &mut history,
            );
            if end == LoopEnd::Aborted {
                super::subagent::cancel_all("parent goal was cancelled");
            }
            engine.driver.set_delta_sink(None);
            engine.transcript = history;
            let _ = tx.send(Ev::Done {
                end,
                engine: Box::new(engine),
            });
        });
        self.rx = Some(rx);
        self.running = true;
        self.started = Some(Instant::now());
        self.follow = true;
        self.scroll = usize::MAX;
        self.status = "running…".into();
    }

    fn command(&mut self, cmd: &str) {
        match cmd.split_whitespace().next().unwrap_or("") {
            "exit" | "quit" => self.quit = true,
            "stop" => {
                if self.running {
                    session::CANCEL.store(true, Ordering::SeqCst);
                    if let Some(p) = self.approval.take() {
                        let _ = p.reply.send(Decision::Abort);
                    }
                    self.status = "stopping…".into();
                } else {
                    self.status = "nothing is running".into();
                }
            }
            "steps" => self.status = format!("step budget: {} per goal", self.max_steps),
            "save" => {
                let id = cmd.split_whitespace().nth(1).unwrap_or("").to_string();
                self.status = match self.engine.as_ref() {
                    None => "busy — /save when idle".into(),
                    Some(e) => {
                        let saved = super::agent_session::SavedAgentSession {
                            id: id.clone(),
                            model_id: self
                                .session
                                .active_id
                                .clone()
                                .unwrap_or_else(|| self.session.active_label.clone()),
                            model_sha256: self
                                .session
                                .active_gguf_sha256()
                                .unwrap_or_default()
                                .to_string(),
                            tool_capable: true,
                            workspace: e.sandbox.root().display().to_string(),
                            transcript: e.transcript.clone(),
                            plan: super::plan::get(),
                            grants: e.policy.granted(),
                        };
                        match super::agent_session::save(&e.sandbox, &saved) {
                            Ok(p) => format!("saved {} → {}", id, e.sandbox.rel(&p)),
                            Err(err) => err,
                        }
                    }
                };
            }
            "resume" => {
                let id = cmd.split_whitespace().nth(1).unwrap_or("").to_string();
                let model = self
                    .session
                    .active_id
                    .clone()
                    .unwrap_or_else(|| self.session.active_label.clone());
                let model_sha256 = self
                    .session
                    .active_gguf_sha256()
                    .unwrap_or_default()
                    .to_string();
                let mut notices: Vec<String> = Vec::new();
                self.status = match self.engine.as_mut() {
                    None => "busy — /resume when idle".into(),
                    Some(e) => match super::agent_session::load(&e.sandbox, &id) {
                        Err(err) => err,
                        Ok(s) => match super::agent_session::check_identity(
                            &s,
                            &model,
                            Some(&model_sha256),
                            true,
                        ) {
                            Err(refusal) => refusal.to_string(),
                            Ok(()) => {
                                // Replayed as context, never re-executed; grants
                                // are NOT restored (a file cannot carry a live
                                // operator's authority) — list them instead.
                                e.transcript = s.transcript.clone();
                                super::plan::set(s.plan.clone());
                                if !s.grants.is_empty() {
                                    notices.push(format!(
                                        "grants are not carried across sessions; previously \
                                         allowed: {} — press 'a' at the next prompt to re-grant",
                                        s.grants.join(", ")
                                    ));
                                }
                                format!(
                                    "resumed {} — {} message(s) replayed as context",
                                    s.id,
                                    s.transcript.len()
                                )
                            }
                        },
                    },
                };
                for n in notices {
                    self.push(Entry::Notice(n));
                }
            }
            "sessions" => {
                let ids = self
                    .engine
                    .as_ref()
                    .map(|e| super::agent_session::list(&e.sandbox))
                    .unwrap_or_default();
                self.status = if ids.is_empty() {
                    "no saved sessions".into()
                } else {
                    ids.join("  ")
                };
            }
            "clear" => {
                if let Some(e) = self.engine.as_mut() {
                    e.transcript.clear();
                }
                super::plan::clear();
                self.status = "context cleared — the next goal starts fresh".into();
            }
            "diff" | "undo" | "checkpoints" => {
                let text = match self.engine.as_ref() {
                    Some(e) => match cmd.split_whitespace().next().unwrap_or("") {
                        "diff" => super::checkpoint::diff(&e.sandbox),
                        "undo" => {
                            let force = cmd.split_whitespace().nth(1) == Some("force");
                            match super::checkpoint::undo(&e.sandbox, force) {
                                Ok(m) => m,
                                Err(err) => err,
                            }
                        }
                        _ => super::checkpoint::summary(),
                    },
                    None => "busy — try again when idle".into(),
                };
                for line in text.lines() {
                    self.push(Entry::Notice(line.to_string()));
                }
            }
            "init" => {
                self.status = match self.engine.as_ref() {
                    Some(e) => match agent::init_project_file(&e.sandbox) {
                        Ok(p) => format!("wrote {}", e.sandbox.rel(&p)),
                        Err(err) => err,
                    },
                    None => "busy — try /init when idle".into(),
                };
            }
            "copy" => {
                let last = self.transcript.iter().rev().find_map(|e| match e {
                    Entry::Model(t) => Some(t.clone()),
                    _ => None,
                });
                self.status = match last {
                    Some(t) if super::clipboard::copy(&t) => "copied the last answer".into(),
                    Some(_) => "could not reach the clipboard".into(),
                    None => "nothing to copy yet".into(),
                };
            }
            "plan" => {
                let steps = super::plan::get();
                self.push(Entry::Notice(format!(
                    "plan ({}):",
                    super::plan::progress(&steps)
                )));
                for line in super::plan::render(&steps).lines() {
                    self.push(Entry::Notice(format!("  {line}")));
                }
            }
            "tools" => self.show_tools(),
            "subagents" => {
                let root = self.engine.as_ref().map(|e| e.sandbox.root().to_path_buf());
                match root {
                    Some(root) => {
                        for line in super::subagent::list_summary(&root).lines() {
                            self.push(Entry::Notice(line.to_string()));
                        }
                    }
                    None => self.status = "busy — try /subagents when idle".into(),
                }
            }
            "theme" => {
                self.theme = self.theme.next();
                self.status = format!("theme: {}", self.theme.name());
            }
            "sidebar" => self.sidebar = !self.sidebar,
            "help" => self.help = true,
            other => {
                debug_assert!(
                    !super::agent::slash_names(true).contains(&other),
                    "SLASH_COMMANDS advertises /{other} but the TUI has no arm for it"
                );
                self.status = format!("unknown command /{other} — F1 for help");
            }
        }
    }

    fn show_tools(&mut self) {
        let granted = self
            .engine
            .as_ref()
            .map(|e| e.policy.granted())
            .unwrap_or_default();
        let specs = super::tools::specs(self.allow_net, self.shell_mode);
        for t in &specs {
            let tag = if !t.risk.needs_approval() {
                " (auto: read-only)"
            } else if granted.contains(&t.name) {
                " (auto: allowed this session)"
            } else {
                ""
            };
            self.push(Entry::Notice(format!(
                "{} [{}]{} — {}",
                t.name,
                t.risk.label(),
                tag,
                t.description
            )));
        }
    }

    // ---- key handling ----------------------------------------------------

    fn on_key(&mut self, key: KeyEvent, paste_burst: bool) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        // The approval modal intercepts every key until it is answered.
        if self.approval.is_some() {
            match key.code {
                KeyCode::Char('y' | 'Y') | KeyCode::Enter => self.resolve_approval(Decision::Once),
                KeyCode::Char('n' | 'N') => self.resolve_approval(Decision::No),
                KeyCode::Char('a' | 'A') => self.resolve_approval(Decision::AlwaysTool),
                KeyCode::Char('q' | 'Q') | KeyCode::Esc => self.resolve_approval(Decision::Abort),
                KeyCode::Char('c') if ctrl => self.resolve_approval(Decision::Abort),
                _ => {}
            }
            return;
        }
        if self.help {
            self.help = false;
            return;
        }
        if ctrl && key.code == KeyCode::Char('d') {
            self.quit = true;
            return;
        }

        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.running {
                    session::CANCEL.store(true, Ordering::SeqCst);
                    self.status = "stopping…".into();
                } else if !self.input.is_empty() {
                    self.input.clear();
                    self.cursor = 0;
                } else {
                    self.status = "Ctrl-D or /exit to quit".into();
                }
            }
            // W5: for hosts that pass Ctrl+V through instead of intercepting it
            // (custom Windows Terminal keybindings, SSH from another box), read
            // the clipboard directly and insert it as text, never keystrokes.
            #[cfg(windows)]
            KeyCode::Char('v') if ctrl => self.paste_from_clipboard(),
            KeyCode::Tab => self.sidebar = !self.sidebar,
            KeyCode::F(1) => self.help = true,
            KeyCode::PageUp => self.scroll_up(10),
            KeyCode::PageDown => self.scroll_down(10),
            KeyCode::Enter if alt => self.insert_char('\n'),
            // W5: an Enter that arrived glued to more queued input is a pasted
            // newline (see the run loop), never a submit. Always false on Unix.
            KeyCode::Enter if paste_burst => self.insert_char('\n'),
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => self.backspace(),
            KeyCode::Left => self.move_cursor(-1),
            KeyCode::Right => self.move_cursor(1),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up => self.history_prev(),
            KeyCode::Down => self.history_next(),
            // W8: the predicate accepts AltGr compositions (CONTROL|ALT) that
            // the old `!ctrl` guard silently swallowed — on a German layout
            // that was every `@ \ { } [ ] €`. CONTROL-alone chords still never
            // insert, and the Ctrl+C/Ctrl+V arms above match first regardless.
            KeyCode::Char(c) if super::term_guard::char_inserts(key.modifiers) => {
                self.insert_char(c)
            }
            _ => {}
        }
    }

    fn resolve_approval(&mut self, d: Decision) {
        if let Some(p) = self.approval.take() {
            let _ = p.reply.send(d);
            if d == Decision::Abort {
                session::CANCEL.store(true, Ordering::SeqCst);
            }
        }
    }

    fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// W5 — Ctrl+V on Windows: insert the clipboard as text. Same rule as the
    /// `Event::Paste` arm: a paste is text, not keystrokes, so newlines are
    /// normalized (`\r\n` and `\r` → `\n`) and never submit.
    #[cfg(windows)]
    fn paste_from_clipboard(&mut self) {
        let Some(text) = super::win_clipboard::read_text() else {
            return;
        };
        for c in text.replace("\r\n", "\n").chars() {
            let c = if c == '\r' { '\n' } else { c };
            self.insert_char(c);
        }
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.input[..self.cursor]
            .chars()
            .next_back()
            .map(char::len_utf8)
            .unwrap_or(1);
        self.cursor -= prev;
        self.input.remove(self.cursor);
    }

    fn move_cursor(&mut self, delta: isize) {
        if delta < 0 {
            if let Some(c) = self.input[..self.cursor].chars().next_back() {
                self.cursor -= c.len_utf8();
            }
        } else if let Some(c) = self.input[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }

    fn history_prev(&mut self) {
        if self.in_hist.is_empty() {
            return;
        }
        let idx = match self.hist_idx {
            Some(0) => 0,
            Some(i) => i - 1,
            None => self.in_hist.len() - 1,
        };
        self.hist_idx = Some(idx);
        self.input = self.in_hist[idx].clone();
        self.cursor = self.input.len();
    }

    fn history_next(&mut self) {
        match self.hist_idx {
            Some(i) if i + 1 < self.in_hist.len() => {
                self.hist_idx = Some(i + 1);
                self.input = self.in_hist[i + 1].clone();
                self.cursor = self.input.len();
            }
            Some(_) => {
                self.hist_idx = None;
                self.input.clear();
                self.cursor = 0;
            }
            None => {}
        }
    }

    fn scroll_up(&mut self, n: usize) {
        if self.scroll == usize::MAX {
            self.scroll = self.last_max_scroll;
        }
        self.scroll = self.scroll.saturating_sub(n);
        self.follow = false;
    }

    fn scroll_down(&mut self, n: usize) {
        if self.scroll == usize::MAX {
            return;
        }
        self.scroll = (self.scroll + n).min(self.last_max_scroll);
        if self.scroll >= self.last_max_scroll {
            self.follow = true;
        }
    }

    // ---- rendering -------------------------------------------------------

    fn draw(&mut self, f: &mut Frame) {
        let area = f.area();
        let body = if self.sidebar && area.width > 70 {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(20), Constraint::Length(34)])
                .split(area);
            self.draw_sidebar(f, cols[1]);
            cols[0]
        } else {
            area
        };
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),
                Constraint::Length(self.input_height()),
                Constraint::Length(1),
            ])
            .split(body);
        self.draw_transcript(f, rows[0]);
        self.draw_input(f, rows[1]);
        self.draw_status(f, rows[2]);
        if self.approval.is_some() {
            self.draw_approval(f, area);
        }
        if self.help {
            self.draw_help(f, area);
        }
    }

    fn input_height(&self) -> u16 {
        let lines = self.input.split('\n').count().max(1) as u16;
        (lines + 2).min(8)
    }

    fn draw_transcript(&mut self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let title = format!(
            " 🐪 Camelid agent — {} · {} ",
            self.session.active_label, self.session.active_posture
        );
        let mut block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(th.primary()))
            .title(Span::styled(
                title,
                Style::default().fg(th.title()).add_modifier(Modifier::BOLD),
            ))
            .title_alignment(Alignment::Left);
        let inner = block.inner(area);
        let width = inner.width.max(1) as usize;
        let lines = self.transcript_lines(width);
        let height = inner.height as usize;
        let max_scroll = lines.len().saturating_sub(height);
        self.last_max_scroll = max_scroll;
        let start = if self.scroll == usize::MAX || self.follow {
            max_scroll
        } else {
            self.scroll.min(max_scroll)
        };
        if start < max_scroll {
            block = block.title(Span::styled(
                format!("  ↓ {} more  ", max_scroll - start),
                Style::default().fg(th.dim()),
            ));
        }
        f.render_widget(block, area);
        let visible: Vec<Line> = lines.into_iter().skip(start).take(height).collect();
        f.render_widget(Paragraph::new(visible), inner);
    }

    fn transcript_lines(&self, width: usize) -> Vec<Line<'static>> {
        let th = self.theme;
        let mut out: Vec<Line<'static>> = Vec::new();
        if self.transcript.is_empty() && !self.running {
            for line in banner::CAMEL_LINES.lines() {
                out.push(Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(th.primary()),
                )));
            }
            out.push(Line::from(""));
            out.push(Line::from(Span::styled(
                "Describe a goal and press Enter. Tools that write, run, or fetch ask first."
                    .to_string(),
                Style::default().fg(th.dim()),
            )));
            return out;
        }
        for entry in &self.transcript {
            match entry {
                Entry::Goal(g) => {
                    out.push(header("▸ goal", th.user()));
                    for piece in wrap(g, width.max(1)) {
                        out.push(Line::from(Span::styled(
                            piece,
                            Style::default().fg(th.user()),
                        )));
                    }
                    out.push(Line::from(""));
                }
                Entry::Model(t) => {
                    out.push(header("▸ agent", th.primary()));
                    out.extend(markdown::render(t, width, th));
                    out.push(Line::from(""));
                }
                Entry::ToolCall(l) => {
                    out.push(Line::from(Span::styled(
                        format!("  ▸ {l}"),
                        Style::default().fg(th.accent()),
                    )));
                }
                Entry::ToolResult { ok, body } => {
                    let (tag, color) = if *ok {
                        ("result", th.dim())
                    } else {
                        ("error", ERR)
                    };
                    out.push(Line::from(Span::styled(
                        format!("  └ {tag}:"),
                        Style::default().fg(color),
                    )));
                    let total = body.lines().count();
                    for line in body.lines().take(12) {
                        out.push(Line::from(Span::styled(
                            format!("    {line}"),
                            Style::default().fg(th.dim()),
                        )));
                    }
                    if total > 12 {
                        out.push(Line::from(Span::styled(
                            format!("    ({} more lines)", total - 12),
                            Style::default().fg(th.dim()),
                        )));
                    }
                    out.push(Line::from(""));
                }
                Entry::Notice(t) => {
                    out.push(Line::from(Span::styled(
                        format!("· {t}"),
                        Style::default().fg(th.dim()),
                    )));
                }
            }
        }
        if self.running {
            // The model's output streams here token-by-token until the step ends
            // (committed to an Entry::Model, or cleared if it was a tool call).
            if !self.live.is_empty() {
                out.push(header("▸ agent", th.primary()));
                out.extend(markdown::render(&self.live, width, th));
            }
            if self.approval.is_none() {
                let spin = SPINNER[(self.frame as usize) % SPINNER.len()];
                let label = if self.live.is_empty() {
                    "thinking…"
                } else {
                    "…"
                };
                out.push(Line::from(Span::styled(
                    format!("{spin} {label}"),
                    Style::default().fg(th.primary()),
                )));
            }
        }
        out
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let border = if self.running { th.dim() } else { th.primary() };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border))
            .title(Span::styled(" › goal ", Style::default().fg(th.title())));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let text: Vec<Line> = self
            .input
            .split('\n')
            .map(|l| Line::from(l.to_string()))
            .collect();
        f.render_widget(Paragraph::new(text), inner);
        if !self.running && self.approval.is_none() {
            let (row, col) = self.cursor_rowcol();
            f.set_cursor_position((inner.x + col as u16, inner.y + row as u16));
        }
    }

    fn cursor_rowcol(&self) -> (usize, usize) {
        let before = &self.input[..self.cursor];
        let row = before.matches('\n').count();
        let col = before
            .rsplit('\n')
            .next()
            .map(|s| s.chars().count())
            .unwrap_or(0);
        (row, col)
    }

    fn draw_status(&self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let mut spans = Vec::new();
        if self.approval.is_some() {
            spans.push(Span::styled(
                "approval needed — [y]es [n]o [a]lways [q]abort  ".to_string(),
                Style::default()
                    .fg(th.accent())
                    .add_modifier(Modifier::BOLD),
            ));
        } else if self.running {
            let spin = SPINNER[(self.frame as usize) % SPINNER.len()];
            let secs = self
                .started
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            spans.push(Span::styled(
                format!("{spin} running · {secs:.0}s  "),
                Style::default().fg(th.primary()),
            ));
            spans.push(Span::styled(
                "^C stop  ".to_string(),
                Style::default().fg(th.dim()),
            ));
        }
        spans.push(Span::styled(
            self.status.clone(),
            Style::default().fg(th.dim()),
        ));
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_sidebar(&self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(th.dim()))
            .title(Span::styled(" agent ", Style::default().fg(th.title())));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let inner = inner.inner(Margin::new(1, 0));
        let shell = match self.shell_mode {
            ShellSandbox::Disabled => "disabled",
            ShellSandbox::Sandboxed => "sandboxed",
            ShellSandbox::Unrestricted => "unrestricted",
        };
        let mut lines = vec![
            kv("model", &self.session.active_label, th),
            kv("posture", &self.session.active_posture, th),
            Line::from(""),
            kv("workspace", &self.workspace, th),
            kv("steps", &self.max_steps.to_string(), th),
            kv("tools", &self.tool_count.to_string(), th),
            kv("run_shell", shell, th),
            kv(
                "files",
                if self.allow_fs {
                    "full disk"
                } else {
                    "workspace"
                },
                th,
            ),
            kv("network", if self.allow_net { "on" } else { "off" }, th),
            kv(
                "approve",
                if self.yolo {
                    "UNATTENDED"
                } else if self.auto_approve {
                    "auto (exec gated)"
                } else {
                    "prompt"
                },
                th,
            ),
            Line::from(""),
            kv("theme", self.theme.name(), th),
            kv("server", &self.addr.to_string(), th),
        ];

        // The plan panel: the agent's own checklist for this goal. Rendered
        // only when it has one, so an agent that never plans costs no space.
        let steps = super::plan::get();
        if !steps.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("plan — {}", super::plan::progress(&steps)),
                Style::default().fg(th.title()).add_modifier(Modifier::BOLD),
            )));
            for s in &steps {
                // Model-authored text: rendered, never interpreted.
                let color = match s.status {
                    super::plan::Status::Done => th.dim(),
                    super::plan::Status::InProgress => th.primary(),
                    super::plan::Status::Pending => th.dim(),
                };
                lines.push(Line::from(Span::styled(
                    format!("{} {}", s.status.marker(), s.text),
                    Style::default().fg(color),
                )));
            }
        }

        f.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_approval(&self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let Some(p) = self.approval.as_ref() else {
            return;
        };
        let rect = centered(area, 72, 50);
        f.render_widget(Clear, rect);
        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                format!("The agent wants to run a {} tool:", p.risk),
                Style::default().fg(th.title()).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];
        // Bound the detail to what the modal can show, with an honest marker —
        // silent clipping would mean approving text the operator never saw.
        let max_detail = (area.height as usize).saturating_sub(10).max(6);
        let total = p.detail.lines().count();
        for raw in p.detail.lines().take(max_detail) {
            lines.push(Line::from(Span::styled(
                format!("  {raw}"),
                Style::default().fg(th.primary()),
            )));
        }
        if total > max_detail {
            lines.push(Line::from(Span::styled(
                format!("  …({} more lines not shown)", total - max_detail),
                Style::default().fg(th.dim()),
            )));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            key_span("[y]", th),
            Span::styled("yes once   ", Style::default().fg(th.dim())),
            key_span("[n]", th),
            Span::styled("no   ", Style::default().fg(th.dim())),
            key_span("[a]", th),
            Span::styled("always this tool   ", Style::default().fg(th.dim())),
            key_span("[q]", th),
            Span::styled("abort", Style::default().fg(th.dim())),
        ]));
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(th.accent()))
            .title(Span::styled(
                " approval needed ",
                Style::default().fg(th.title()).add_modifier(Modifier::BOLD),
            ));
        f.render_widget(Paragraph::new(lines).block(block), rect);
    }

    fn draw_help(&self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let rect = centered(area, 64, 72);
        f.render_widget(Clear, rect);
        let mut lines = vec![
            help_row("Enter", "run the goal · Alt+Enter newline", th),
            help_row("Ctrl-C", "stop the running goal / clear input", th),
            help_row("Ctrl-D", "quit", th),
            help_row("PgUp/PgDn", "scroll · wheel scrolls too", th),
            help_row("Tab", "toggle the sidebar", th),
            help_row("Up/Down", "input history", th),
            Line::from(""),
        ];
        // Derived from the shared table so a new command cannot ship undocumented.
        for c in super::agent::SLASH_COMMANDS {
            lines.push(help_row(&format!("/{}", c.name), c.help, th));
        }
        lines.push(Line::from(""));
        lines.push(help_row(
            "approval",
            "y yes · n no · a always · q abort",
            th,
        ));
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(th.primary()))
            .title(Span::styled(
                " agent help — any key closes ",
                Style::default().fg(th.title()).add_modifier(Modifier::BOLD),
            ));
        f.render_widget(Paragraph::new(lines).block(block), rect);
    }
}

fn end_label(end: LoopEnd) -> &'static str {
    match end {
        LoopEnd::Answered => "done",
        LoopEnd::Aborted => "stopped",
        LoopEnd::StepCapped => "stopped at the step limit",
        LoopEnd::Repeated => "stopped — the model was repeating a failing call",
        LoopEnd::DriverError => "stopped on a model error",
    }
}

fn header(label: &str, color: Color) -> Line<'static> {
    Line::from(Span::styled(
        label.to_string(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ))
}

fn key_span(k: &str, th: Theme) -> Span<'static> {
    Span::styled(
        format!("{k} "),
        Style::default().fg(th.title()).add_modifier(Modifier::BOLD),
    )
}

fn kv(key: &str, value: &str, th: Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{key:<11}"), Style::default().fg(th.dim())),
        Span::styled(value.to_string(), Style::default().fg(th.title())),
    ])
}

fn help_row(key: &str, desc: &str, th: Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key:<14}"), Style::default().fg(th.title())),
        Span::styled(desc.to_string(), Style::default().fg(th.dim())),
    ])
}

fn centered(area: Rect, pct_w: u16, pct_h: u16) -> Rect {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - pct_h) / 2),
            Constraint::Percentage(pct_h),
            Constraint::Percentage((100 - pct_h) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - pct_w) / 2),
            Constraint::Percentage(pct_w),
            Constraint::Percentage((100 - pct_w) / 2),
        ])
        .split(v[1])[1]
}

/// Char-safe word wrap for plain (non-markdown) lines (mirrors `tui::wrap`).
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize;
    for word in text.split(' ') {
        let wlen = word.chars().count();
        if cur_len == 0 {
            place(&mut out, &mut cur, &mut cur_len, word, wlen, width);
        } else if cur_len + 1 + wlen <= width {
            cur.push(' ');
            cur.push_str(word);
            cur_len += 1 + wlen;
        } else {
            out.push(std::mem::take(&mut cur));
            cur_len = 0;
            place(&mut out, &mut cur, &mut cur_len, word, wlen, width);
        }
    }
    out.push(cur);
    out
}

fn place(
    out: &mut Vec<String>,
    cur: &mut String,
    cur_len: &mut usize,
    word: &str,
    wlen: usize,
    width: usize,
) {
    if wlen > width {
        let mut chunk = String::new();
        let mut n = 0;
        for ch in word.chars() {
            if n == width {
                out.push(std::mem::take(&mut chunk));
                n = 0;
            }
            chunk.push(ch);
            n += 1;
        }
        *cur = chunk;
        *cur_len = n;
    } else {
        *cur = word.to_string();
        *cur_len = wlen;
    }
}
