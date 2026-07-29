//! Full-screen ratatui front end for `camelid chat`.
//!
//! The redraw loop runs on the main thread; each generation streams on a
//! background thread that forwards deltas over a channel, so the UI stays
//! responsive and Ctrl-C cancels mid-stream. Everything rides the same audited
//! HTTP/SSE client and the shared [`Session`] core.
//!
//! Features: a `/` command palette, instant switching between models already
//! loaded in the server, Markdown-rendered replies, a live streaming spinner +
//! tok/s, a context gauge, themes, and clipboard copy.

use std::io::Stdout;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph};
use ratatui::{Frame, Terminal};

use super::client::{LoadedInfo, StreamEnd};
use super::markdown;
use super::models::{Availability, PickerRow};
use super::palette;
use super::session::{self, Role, Session};
use super::theme::Theme;
use super::{banner, clipboard};

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

enum StreamMsg {
    Delta(String),
    Done { tokens: u32, capped: bool },
    Cancelled,
    Error(String),
}

enum Mode {
    Normal,
    Picker,
    Help,
}

enum Suspend {
    Pull(String),
    DownloadAndLoad(usize),
}

/// One row of the model browser: a model already loaded (instant switch) or a
/// supported catalog row (load/download).
enum PickEntry {
    Loaded(LoadedInfo, bool),
    Catalog(PickerRow),
}

struct PickerUi {
    entries: Vec<PickEntry>,
    state: ListState,
}

pub fn run(session: &mut Session, addr: SocketAddr, spawned: bool) -> anyhow::Result<()> {
    enable_raw_mode()?;
    // W6: arm the restore guard before anything else can fail, and route panics
    // through the chaining hook — same discipline as the agent TUI. The
    // `run_suspended` teardown/re-entry below is unaffected (restore is
    // idempotent and only the final drop actually restores).
    super::term_guard::install_panic_hook();
    let guard = super::term_guard::TerminalGuard::arm();
    let mut out = std::io::stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(out))?;

    let mut app = App::new(session, addr, spawned);
    let result = app.run(&mut terminal);

    drop(guard);
    result
}

struct App<'a> {
    session: &'a mut Session,
    addr: SocketAddr,
    spawned: bool,
    theme: Theme,
    input: String,
    cursor: usize,
    scroll: usize,
    last_max_scroll: usize,
    follow: bool,
    sidebar: bool,
    mode: Mode,
    picker: PickerUi,
    palette_open: bool,
    palette_sel: usize,
    palette_dismissed: bool,
    status: String,
    frame: u64,
    stream_rx: Option<Receiver<StreamMsg>>,
    streaming: bool,
    live: String,
    live_tokens: u32,
    started: Option<Instant>,
    stats: Option<String>,
    history: Vec<String>,
    hist_idx: Option<usize>,
    pending: Option<Suspend>,
    quit: bool,
}

impl<'a> App<'a> {
    fn new(session: &'a mut Session, addr: SocketAddr, spawned: bool) -> Self {
        let status = format!(
            "server {} at {addr} · type / for commands · F1 help",
            if spawned { "spawned" } else { "attached" }
        );
        App {
            session,
            addr,
            spawned,
            theme: Theme::Sandstorm,
            input: String::new(),
            cursor: 0,
            scroll: 0,
            last_max_scroll: 0,
            follow: true,
            sidebar: true,
            mode: Mode::Normal,
            picker: PickerUi {
                entries: Vec::new(),
                state: ListState::default(),
            },
            palette_open: false,
            palette_sel: 0,
            palette_dismissed: false,
            status,
            frame: 0,
            stream_rx: None,
            streaming: false,
            live: String::new(),
            live_tokens: 0,
            started: None,
            stats: None,
            history: Vec::new(),
            hist_idx: None,
            pending: None,
            quit: false,
        }
    }

    fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> anyhow::Result<()> {
        if !self.session.has_model() {
            self.open_picker();
        }
        while !self.quit {
            self.frame = self.frame.wrapping_add(1);
            terminal.draw(|f| self.draw(f))?;
            self.poll_stream();
            // Poll faster while streaming so the spinner animates smoothly.
            let timeout = if self.streaming { 80 } else { 120 };
            if event::poll(Duration::from_millis(timeout))? {
                match event::read()? {
                    Event::Key(key) if key.kind != KeyEventKind::Release => self.on_key(key),
                    Event::Mouse(m) => match m.kind {
                        MouseEventKind::ScrollUp => self.scroll_up(3),
                        MouseEventKind::ScrollDown => self.scroll_down(3),
                        _ => {}
                    },
                    _ => {}
                }
            }
            if let Some(op) = self.pending.take() {
                self.run_suspended(terminal, op)?;
            }
        }
        Ok(())
    }

    // ---- streaming -------------------------------------------------------

    fn start_generation(&mut self) {
        let request = self.session.build_request(true);
        session::CANCEL.store(false, Ordering::SeqCst);
        let client = self.session.client();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let forward = tx.clone();
            let result = client.chat_stream(&request, &session::CANCEL, move |delta| {
                let _ = forward.send(StreamMsg::Delta(delta.to_string()));
            });
            let _ = match result {
                Ok((StreamEnd::Done, n)) => tx.send(StreamMsg::Done {
                    tokens: n,
                    capped: false,
                }),
                Ok((StreamEnd::Length, n)) => tx.send(StreamMsg::Done {
                    tokens: n,
                    capped: true,
                }),
                Ok((StreamEnd::Cancelled, _)) => tx.send(StreamMsg::Cancelled),
                Err(err) => tx.send(StreamMsg::Error(err.to_string())),
            };
        });
        self.stream_rx = Some(rx);
        self.streaming = true;
        self.live.clear();
        self.live_tokens = 0;
        self.started = Some(Instant::now());
        self.stats = None;
        self.follow = true;
        self.scroll = usize::MAX;
    }

    fn poll_stream(&mut self) {
        if self.stream_rx.is_none() {
            return;
        }
        loop {
            let msg = match self.stream_rx.as_ref().unwrap().try_recv() {
                Ok(msg) => msg,
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.finish(None, false);
                    return;
                }
            };
            match msg {
                StreamMsg::Delta(d) => {
                    self.live.push_str(&d);
                    self.live_tokens += 1;
                    if self.follow {
                        self.scroll = usize::MAX;
                    }
                }
                StreamMsg::Done { tokens, capped } => {
                    self.finish(Some(tokens), capped);
                    return;
                }
                StreamMsg::Cancelled => {
                    self.session.pop_last();
                    self.end_stream();
                    self.status = "interrupted — turn discarded".into();
                    return;
                }
                StreamMsg::Error(e) => {
                    self.session.pop_last();
                    self.end_stream();
                    self.status = format!("generation failed: {e}");
                    return;
                }
            }
        }
    }

    fn end_stream(&mut self) {
        self.streaming = false;
        self.stream_rx = None;
        self.live.clear();
        self.live_tokens = 0;
    }

    fn finish(&mut self, completion: Option<u32>, capped: bool) {
        let text = std::mem::take(&mut self.live);
        let secs = self
            .started
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        self.session.push_assistant(text);
        self.session.last_prompt_tokens = None;
        self.session.last_completion_tokens = completion;
        self.stats = Some(match completion {
            Some(n) if secs > 0.0 => format!("{n} tok · {:.0} tok/s · {secs:.1}s", n as f64 / secs),
            Some(n) => format!("{n} tok · {secs:.1}s"),
            None => format!("{secs:.1}s"),
        });
        self.end_stream();
        if capped {
            self.status =
                "answer cut off at max_tokens — raise it to finish this response".to_string();
        }
    }

    // ---- input / commands ------------------------------------------------

    fn submit(&mut self) {
        let text = self.input.trim().to_string();
        self.input.clear();
        self.cursor = 0;
        self.palette_open = false;
        self.palette_dismissed = false;
        if text.is_empty() {
            return;
        }
        self.history.push(text.clone());
        self.hist_idx = None;
        if let Some(command) = text.strip_prefix('/') {
            self.command(command);
        } else if self.session.has_model() {
            self.session.push_user(text);
            self.start_generation();
        } else {
            self.status = "no model loaded — /models to pick one".into();
        }
    }

    fn command(&mut self, command: &str) {
        let mut parts = command.splitn(2, char::is_whitespace);
        let raw = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        let name = palette::resolve(raw).map(|c| c.name).unwrap_or(raw);
        match name {
            "models" => self.open_picker(),
            "switch" => self.open_picker(),
            "model" => self.select_by_id(arg),
            "set" => {
                let mut kv = arg.splitn(2, char::is_whitespace);
                match (kv.next(), kv.next()) {
                    (Some(k), Some(v)) if !k.is_empty() && !v.trim().is_empty() => {
                        self.status = match self.session.set_param(k, v.trim()) {
                            Ok(msg) => format!("✓ {msg}"),
                            Err(err) => format!("✗ {err}"),
                        };
                    }
                    _ => {
                        self.status = format!(
                            "usage: /set <name> <value> · {}",
                            self.session.settings.summary()
                        )
                    }
                }
            }
            "system" => {
                if arg.is_empty() {
                    self.status = match &self.session.system {
                        Some(t) => format!("system: {t}"),
                        None => "no system prompt set".into(),
                    };
                } else {
                    self.session.system = Some(arg.to_string());
                    self.status = "system prompt set (next turn)".into();
                }
            }
            "reset" => {
                self.session.reset_history();
                self.status = "history cleared".into();
            }
            "retry" => self.retry(),
            "stop" => {
                if self.streaming {
                    session::CANCEL.store(true, Ordering::SeqCst);
                    self.status = "stopping…".into();
                } else {
                    self.status = "nothing is generating".into();
                }
            }
            "copy" => self.copy_last(),
            "theme" => {
                self.theme = if arg.is_empty() {
                    self.theme.next()
                } else {
                    Theme::from_name(arg).unwrap_or_else(|| self.theme.next())
                };
                self.status = format!("theme: {}", self.theme.name());
            }
            "sidebar" => self.sidebar = !self.sidebar,
            "tokens" => {
                let f = |v: Option<u32>| v.map(|n| n.to_string()).unwrap_or_else(|| "n/a".into());
                self.status = format!(
                    "last turn — prompt: {}  completion: {}",
                    f(self.session.last_prompt_tokens),
                    f(self.session.last_completion_tokens)
                );
            }
            "info" => {
                self.sidebar = true;
                self.status = "model + settings in the sidebar".into();
            }
            "save" => {
                let path = std::path::PathBuf::from(if arg.is_empty() {
                    "camelid-session.json"
                } else {
                    arg
                });
                self.status = match self.session.save(&path) {
                    Ok(()) => format!("saved → {}", path.display()),
                    Err(err) => format!("save failed: {err}"),
                };
            }
            "load" => {
                if arg.is_empty() {
                    self.status = "usage: /load <path.json>".into();
                } else {
                    self.status = match self.session.load(&std::path::PathBuf::from(arg)) {
                        Ok(()) => format!("loaded {} turn(s)", self.session.history.len()),
                        Err(err) => format!("load failed: {err}"),
                    };
                }
            }
            "pull" => {
                if arg.is_empty() {
                    self.status = "usage: /pull <alias>".into();
                } else {
                    self.pending = Some(Suspend::Pull(arg.to_string()));
                }
            }
            "agent" => {
                self.status =
                    "agent mode: relaunch with  camelid chat --agent --model <gguf>".into()
            }
            "help" => self.mode = Mode::Help,
            "exit" => self.quit = true,
            other => self.status = format!("unknown command /{other} — type / to browse"),
        }
    }

    fn copy_last(&mut self) {
        let last = self
            .session
            .history
            .iter()
            .rev()
            .find(|t| t.role == Role::Assistant)
            .map(|t| t.content.clone());
        match last {
            Some(text) if clipboard::copy(&text) => {
                self.status = format!("copied {} chars to clipboard", text.len())
            }
            Some(_) => self.status = "clipboard copy not supported by this terminal".into(),
            None => self.status = "no reply to copy yet".into(),
        }
    }

    fn retry(&mut self) {
        if !self.session.has_model() || self.session.last_user_message().is_none() {
            self.status = "nothing to retry".into();
            return;
        }
        if matches!(
            self.session.history.last().map(|t| t.role),
            Some(Role::Assistant)
        ) {
            self.session.pop_last();
        }
        self.start_generation();
    }

    // ---- model browser (loaded switch + catalog) -------------------------

    fn open_picker(&mut self) {
        let mut entries = Vec::new();
        let active = self.session.active_id.clone();
        for info in self.session.loaded_models() {
            let is_active = active.as_deref() == Some(info.id.as_str());
            entries.push(PickEntry::Loaded(info, is_active));
        }
        for row in self.session.supported_rows() {
            entries.push(PickEntry::Catalog(row));
        }
        self.picker
            .state
            .select(if entries.is_empty() { None } else { Some(0) });
        self.picker.entries = entries;
        self.mode = Mode::Picker;
    }

    fn picker_move(&mut self, delta: isize) {
        let len = self.picker.entries.len();
        if len == 0 {
            return;
        }
        let cur = self.picker.state.selected().unwrap_or(0) as isize;
        let next = (cur + delta).rem_euclid(len as isize) as usize;
        self.picker.state.select(Some(next));
    }

    fn picker_choose(&mut self) {
        let Some(index) = self.picker.state.selected() else {
            return;
        };
        match self.picker.entries.get(index) {
            Some(PickEntry::Loaded(info, _)) => {
                let info = info.clone();
                self.session.switch_to_loaded(&info);
                self.status = format!("switched to {} (history reset)", info.id);
                self.mode = Mode::Normal;
            }
            Some(PickEntry::Catalog(row)) => match row.availability {
                Availability::Ready => {
                    let id = row.id.clone();
                    let path = row.local_path(self.session.models_dir());
                    self.mode = Mode::Normal;
                    if let Some(path) = path {
                        self.load_path(&path, &id);
                    }
                }
                Availability::NotDownloaded => {
                    self.mode = Mode::Normal;
                    self.pending = Some(Suspend::DownloadAndLoad(index));
                }
                Availability::NoPullAlias => {
                    self.status = format!("'{}' has no pull alias — use --model", row.id);
                }
            },
            None => {}
        }
    }

    fn select_by_id(&mut self, id: &str) {
        // Prefer an already-loaded model (instant switch), else a catalog row.
        if let Some(info) = self
            .session
            .loaded_models()
            .into_iter()
            .find(|m| m.id == id)
        {
            self.session.switch_to_loaded(&info);
            self.status = format!("switched to {id} (history reset)");
            return;
        }
        let rows = self.session.supported_rows();
        let Some(row) = rows.iter().find(|r| r.id == id) else {
            self.status = format!("'{id}' is not loaded or a supported id — type /models");
            return;
        };
        match row.availability {
            Availability::Ready => {
                if let Some(path) = row.local_path(self.session.models_dir()) {
                    let id = row.id.clone();
                    self.load_path(&path, &id);
                }
            }
            Availability::NotDownloaded => {
                self.status = format!("'{id}' not downloaded — /models to fetch")
            }
            Availability::NoPullAlias => self.status = format!("'{id}' has no pull alias"),
        }
    }

    fn load_path(&mut self, path: &std::path::Path, label: &str) {
        match self
            .session
            .load_model_file(path, Some(label), Some("supported"))
        {
            Ok(session::LoadResult::Loaded) => {
                self.status = format!("active model: {label} (history reset)")
            }
            Ok(session::LoadResult::Unsupported(message)) => self.status = message,
            Err(err) => self.status = format!("load failed: {err}"),
        }
    }

    fn run_suspended(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
        op: Suspend,
    ) -> anyhow::Result<()> {
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;

        match op {
            Suspend::Pull(alias) => {
                if let Err(err) = crate::catalog::run_pull(Some(&alias), self.session.models_dir())
                {
                    self.status = format!("pull failed: {err}");
                } else {
                    self.status = "downloaded — /models to load it".into();
                }
            }
            Suspend::DownloadAndLoad(index) => {
                if let Some(PickEntry::Catalog(row)) = self.picker.entries.get(index) {
                    if let Some(item) = row.catalog.as_ref() {
                        let id = row.id.clone();
                        match crate::catalog::run_pull(
                            Some(item.catalog_id),
                            self.session.models_dir(),
                        ) {
                            Ok(()) => {
                                if let Some(path) = row.local_path(self.session.models_dir()) {
                                    self.load_path(&path.clone(), &id);
                                }
                            }
                            Err(err) => self.status = format!("pull failed: {err}"),
                        }
                    }
                }
            }
        }

        enable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            EnterAlternateScreen,
            EnableMouseCapture
        )?;
        terminal.clear()?;
        Ok(())
    }

    // ---- scrolling -------------------------------------------------------

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

    // ---- key handling ----------------------------------------------------

    fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        if ctrl && key.code == KeyCode::Char('d') {
            self.quit = true;
            return;
        }

        match self.mode {
            Mode::Help => {
                self.mode = Mode::Normal;
                return;
            }
            Mode::Picker => {
                match key.code {
                    KeyCode::Up | KeyCode::Char('k') => self.picker_move(-1),
                    KeyCode::Down | KeyCode::Char('j') => self.picker_move(1),
                    KeyCode::Enter => self.picker_choose(),
                    KeyCode::Esc | KeyCode::Char('q') => self.mode = Mode::Normal,
                    _ => {}
                }
                return;
            }
            Mode::Normal => {}
        }

        // Command palette intercepts navigation while open.
        if self.palette_open {
            match key.code {
                KeyCode::Up => {
                    self.palette_sel = self.palette_sel.saturating_sub(1);
                    return;
                }
                KeyCode::Down => {
                    self.palette_sel += 1;
                    self.clamp_palette();
                    return;
                }
                KeyCode::Tab => {
                    self.complete_palette();
                    return;
                }
                KeyCode::Esc => {
                    self.palette_open = false;
                    self.palette_dismissed = true;
                    return;
                }
                KeyCode::Enter => {
                    self.palette_enter();
                    return;
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.streaming {
                    session::CANCEL.store(true, Ordering::SeqCst);
                } else if !self.input.is_empty() {
                    self.input.clear();
                    self.cursor = 0;
                    self.refresh_palette();
                } else {
                    self.status = "Ctrl-D or /exit to quit".into();
                }
            }
            KeyCode::Char('l') if ctrl => {
                self.session.reset_history();
                self.status = "history cleared".into();
            }
            KeyCode::Tab => self.sidebar = !self.sidebar,
            KeyCode::F(1) => self.mode = Mode::Help,
            KeyCode::PageUp => self.scroll_up(10),
            KeyCode::PageDown => self.scroll_down(10),
            KeyCode::Enter if alt => {
                self.insert_char('\n');
            }
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => self.backspace(),
            KeyCode::Left => self.move_cursor(-1),
            KeyCode::Right => self.move_cursor(1),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up => self.history_prev(),
            KeyCode::Down => self.history_next(),
            // W8: accept AltGr compositions (CONTROL|ALT); see term_guard.
            KeyCode::Char(c) if super::term_guard::char_inserts(key.modifiers) => {
                self.insert_char(c)
            }
            _ => {}
        }
    }

    fn refresh_palette(&mut self) {
        self.palette_open = self.input.starts_with('/') && !self.palette_dismissed;
        self.clamp_palette();
    }

    fn clamp_palette(&mut self) {
        let n = self.palette_matches().len();
        if n == 0 {
            self.palette_sel = 0;
        } else if self.palette_sel >= n {
            self.palette_sel = n - 1;
        }
    }

    fn palette_query(&self) -> String {
        self.input
            .strip_prefix('/')
            .unwrap_or("")
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string()
    }

    fn palette_matches(&self) -> Vec<&'static palette::Cmd> {
        palette::matches(&self.palette_query())
    }

    fn complete_palette(&mut self) {
        if let Some(cmd) = self.palette_matches().get(self.palette_sel) {
            self.input = format!("/{} ", cmd.name);
            self.cursor = self.input.len();
            self.palette_dismissed = false;
            self.refresh_palette();
        }
    }

    fn palette_enter(&mut self) {
        // Did the user already type an argument after the command word?
        let typed = self.input.strip_prefix('/').unwrap_or("");
        let arg_typed = typed
            .split_once(char::is_whitespace)
            .map(|(_, rest)| !rest.trim().is_empty())
            .unwrap_or(false);
        if arg_typed {
            self.submit();
            return;
        }
        match self.palette_matches().get(self.palette_sel) {
            // A command with a required (`<…>`) argument completes and waits;
            // no-arg and optional-arg (`[…]`) commands run immediately.
            Some(cmd) if cmd.args.starts_with('<') => {
                self.input = format!("/{} ", cmd.name);
                self.cursor = self.input.len();
                self.palette_dismissed = false;
            }
            Some(cmd) => {
                self.input = format!("/{}", cmd.name);
                self.submit();
            }
            None => self.submit(),
        }
    }

    fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.palette_dismissed = false;
        self.refresh_palette();
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
        self.refresh_palette();
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
        if self.history.is_empty() {
            return;
        }
        let idx = match self.hist_idx {
            Some(0) => 0,
            Some(i) => i - 1,
            None => self.history.len() - 1,
        };
        self.hist_idx = Some(idx);
        self.input = self.history[idx].clone();
        self.cursor = self.input.len();
        self.refresh_palette();
    }

    fn history_next(&mut self) {
        match self.hist_idx {
            Some(i) if i + 1 < self.history.len() => {
                self.hist_idx = Some(i + 1);
                self.input = self.history[i + 1].clone();
                self.cursor = self.input.len();
            }
            Some(_) => {
                self.hist_idx = None;
                self.input.clear();
                self.cursor = 0;
            }
            None => {}
        }
        self.refresh_palette();
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

        self.draw_chat(f, rows[0]);
        self.draw_input(f, rows[1]);
        self.draw_status(f, rows[2]);
        if self.palette_open {
            self.draw_palette(f, rows[1]);
        }

        match self.mode {
            Mode::Picker => self.draw_picker(f, area),
            Mode::Help => self.draw_help(f, area),
            Mode::Normal => {}
        }
    }

    fn input_height(&self) -> u16 {
        let lines = self.input.split('\n').count().max(1) as u16;
        (lines + 2).min(8)
    }

    fn draw_chat(&mut self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let label = if self.session.has_model() {
            format!(
                " 🐪 Camelid — {} · {} ",
                self.session.active_label, self.session.active_posture
            )
        } else {
            " 🐪 Camelid — no model ".to_string()
        };
        let mut block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(th.primary()))
            .title(Span::styled(
                label,
                Style::default().fg(th.title()).add_modifier(Modifier::BOLD),
            ))
            .title_alignment(Alignment::Left);
        let inner = block.inner(area);

        let width = inner.width.max(1) as usize;
        let lines = self.chat_lines(width);
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

    fn chat_lines(&self, width: usize) -> Vec<Line<'static>> {
        let th = self.theme;
        let mut out: Vec<Line<'static>> = Vec::new();
        if self.session.history.is_empty() && !self.streaming {
            for line in banner::CAMEL_LINES.lines() {
                out.push(Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(th.primary()),
                )));
            }
            out.push(Line::from(""));
            out.push(Line::from(Span::styled(
                "Type a message and press Enter — or / for commands.".to_string(),
                Style::default().fg(th.dim()),
            )));
            return out;
        }
        for turn in &self.session.history {
            push_header(&mut out, turn.role, th);
            match turn.role {
                Role::User => push_plain(&mut out, &turn.content, width, th.user()),
                Role::Assistant => out.extend(markdown::render(&turn.content, width, th)),
            }
            out.push(Line::from(""));
        }
        if self.streaming {
            push_header(&mut out, Role::Assistant, th);
            out.extend(markdown::render(&self.live, width, th));
            let spin = SPINNER[(self.frame as usize) % SPINNER.len()];
            out.push(Line::from(Span::styled(
                spin.to_string(),
                Style::default().fg(th.primary()),
            )));
        }
        out
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let border = if self.streaming {
            th.dim()
        } else {
            th.primary()
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border))
            .title(Span::styled(" › ", Style::default().fg(th.title())));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let text: Vec<Line> = self
            .input
            .split('\n')
            .map(|l| Line::from(l.to_string()))
            .collect();
        f.render_widget(Paragraph::new(text), inner);
        if !self.streaming {
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
        if self.streaming {
            let spin = SPINNER[(self.frame as usize) % SPINNER.len()];
            let secs = self
                .started
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            let tps = if secs > 0.0 {
                self.live_tokens as f64 / secs
            } else {
                0.0
            };
            spans.push(Span::styled(
                format!("{spin} {} tok · {tps:.0} tok/s  ", self.live_tokens),
                Style::default().fg(th.primary()),
            ));
            spans.push(Span::styled("^C stop  ", Style::default().fg(th.dim())));
        } else if let Some(stats) = &self.stats {
            spans.push(Span::styled(
                format!("└ {stats}  "),
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
            .title(Span::styled(" settings ", Style::default().fg(th.title())));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let inner = inner.inner(Margin::new(1, 0));
        let s = &self.session.settings;
        let lines = vec![
            kv("model", &self.session.active_label, th),
            kv("posture", &self.session.active_posture, th),
            kv(
                "loaded",
                &self.session.loaded_models().len().to_string(),
                th,
            ),
            Line::from(""),
            kv("temperature", &format!("{:.2}", s.temperature), th),
            kv("top_p", &opt(s.top_p.map(|v| format!("{v:.2}"))), th),
            kv("top_k", &opt(s.top_k.map(|v| v.to_string())), th),
            kv("max_tokens", &s.max_tokens.to_string(), th),
            kv("seed", &opt(s.seed.map(|v| v.to_string())), th),
            kv("stream", if s.stream { "on" } else { "off" }, th),
            Line::from(""),
            kv(
                "system",
                if self.session.system.is_some() {
                    "set"
                } else {
                    "—"
                },
                th,
            ),
            kv("turns", &self.session.history.len().to_string(), th),
            kv("theme", self.theme.name(), th),
            kv("server", &self.addr.to_string(), th),
            kv("via", if self.spawned { "spawned" } else { "attached" }, th),
        ];
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(inner);
        f.render_widget(Paragraph::new(lines), rows[0]);
        self.draw_context_gauge(f, rows[1]);
    }

    fn draw_context_gauge(&self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let used = self.session.last_prompt_tokens.unwrap_or(0)
            + self.session.last_completion_tokens.unwrap_or(0);
        let label = match self.session.active_ctx {
            Some(ctx) if ctx > 0 => {
                let ratio = (used as f64 / ctx as f64).clamp(0.0, 1.0);
                let gauge = Gauge::default()
                    .block(
                        Block::default()
                            .title(Span::styled(" context ", Style::default().fg(th.dim()))),
                    )
                    .gauge_style(Style::default().fg(th.primary()))
                    .ratio(ratio)
                    .label(format!("{used}/{ctx}"));
                f.render_widget(gauge, area);
                return;
            }
            _ => format!("context {used}/?"),
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                label,
                Style::default().fg(th.dim()),
            ))),
            area,
        );
    }

    fn draw_palette(&self, f: &mut Frame, input_area: Rect) {
        let th = self.theme;
        let matches = self.palette_matches();
        if matches.is_empty() {
            return;
        }
        let rows = (matches.len() as u16 + 2).min(10);
        let y = input_area.y.saturating_sub(rows);
        let rect = Rect {
            x: input_area.x,
            y,
            width: input_area.width,
            height: rows,
        };
        f.render_widget(Clear, rect);
        let items: Vec<ListItem> = matches
            .iter()
            .map(|c| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" /{:<10}", c.name), Style::default().fg(th.title())),
                    Span::styled(format!("{:<16}", c.args), Style::default().fg(th.dim())),
                    Span::styled(c.desc.to_string(), Style::default().fg(th.dim())),
                ]))
            })
            .collect();
        let mut state = ListState::default();
        state.select(Some(self.palette_sel.min(matches.len() - 1)));
        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(th.primary()))
                    .title(Span::styled(
                        " commands · ↑↓ Tab Enter ",
                        Style::default().fg(th.title()),
                    )),
            )
            .highlight_style(
                Style::default()
                    .bg(th.highlight_bg())
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("› ");
        f.render_stateful_widget(list, rect, &mut state);
    }

    fn draw_picker(&mut self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let rect = centered(area, 70, 70);
        f.render_widget(Clear, rect);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(th.primary()))
            .title(Span::styled(
                " models — ● loaded (instant) · ↑↓ Enter · Esc ",
                Style::default().fg(th.title()).add_modifier(Modifier::BOLD),
            ));
        if self.picker.entries.is_empty() {
            f.render_widget(
                Paragraph::new("the server advertises no supported models")
                    .block(block)
                    .style(Style::default().fg(th.dim())),
                rect,
            );
            return;
        }
        let items: Vec<ListItem> = self
            .picker
            .entries
            .iter()
            .map(|entry| match entry {
                PickEntry::Loaded(info, active) => {
                    let dot = if *active { "● " } else { "○ " };
                    Line::from(vec![
                        Span::styled(dot.to_string(), Style::default().fg(th.primary())),
                        Span::styled(format!("{:<30} ", info.id), Style::default().fg(th.title())),
                        Span::styled(info.descriptor(), Style::default().fg(th.dim())),
                    ])
                }
                PickEntry::Catalog(row) => {
                    let tag = match row.availability {
                        Availability::Ready => "[ready]",
                        Availability::NotDownloaded => "[download]",
                        Availability::NoPullAlias => "[no alias]",
                    };
                    Line::from(vec![
                        Span::raw("  "),
                        Span::styled(format!("{:<30} ", row.id), Style::default().fg(th.accent())),
                        Span::styled(format!("{:<6} ", row.quant), Style::default().fg(th.dim())),
                        Span::styled(tag.to_string(), Style::default().fg(th.dim())),
                    ])
                }
            })
            .map(ListItem::new)
            .collect();
        let list = List::new(items)
            .block(block)
            .highlight_style(
                Style::default()
                    .bg(th.highlight_bg())
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("› ");
        f.render_stateful_widget(list, rect, &mut self.picker.state);
    }

    fn draw_help(&self, f: &mut Frame, area: Rect) {
        let th = self.theme;
        let rect = centered(area, 64, 76);
        f.render_widget(Clear, rect);
        let mut lines: Vec<Line> = vec![
            help_row("Enter", "send · Alt+Enter newline", th),
            help_row("/", "open the command palette", th),
            help_row("Ctrl-C", "stop a stream / clear input", th),
            help_row("Ctrl-D", "quit · Ctrl-L clear history", th),
            help_row("PgUp/PgDn", "scroll · wheel scrolls too", th),
            help_row("Tab", "toggle the sidebar", th),
            help_row("Up/Down", "input history", th),
            help_row("F1", "this help", th),
            Line::from(""),
        ];
        for cmd in palette::COMMANDS {
            lines.push(help_row(
                &format!("/{} {}", cmd.name, cmd.args),
                cmd.desc,
                th,
            ));
        }
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(th.primary()))
            .title(Span::styled(
                " keys & commands — any key closes ",
                Style::default().fg(th.title()).add_modifier(Modifier::BOLD),
            ));
        f.render_widget(Paragraph::new(lines).block(block), rect);
    }
}

fn push_header(out: &mut Vec<Line<'static>>, role: Role, th: Theme) {
    let color = match role {
        Role::User => th.user(),
        Role::Assistant => th.primary(),
    };
    out.push(Line::from(Span::styled(
        format!("▸ {}", role.display()),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )));
}

fn push_plain(
    out: &mut Vec<Line<'static>>,
    text: &str,
    width: usize,
    color: ratatui::style::Color,
) {
    for raw in text.split('\n') {
        for piece in wrap(raw, width.max(1)) {
            out.push(Line::from(Span::styled(piece, Style::default().fg(color))));
        }
    }
}

fn help_row(key: &str, desc: &str, th: Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key:<16}"), Style::default().fg(th.title())),
        Span::styled(desc.to_string(), Style::default().fg(th.dim())),
    ])
}

fn kv(key: &str, value: &str, th: Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{key:<12}"), Style::default().fg(th.dim())),
        Span::styled(value.to_string(), Style::default().fg(th.title())),
    ])
}

fn opt(value: Option<String>) -> String {
    value.unwrap_or_else(|| "off".into())
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

/// Char-safe word wrap for plain (non-markdown) lines.
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
