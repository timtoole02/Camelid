// Background lifetime: what closing the main window does, what the tray says about the
// engine, and the preference that chooses between them.
//
// Pure and Tauri-free, so every rule that carries a promise is unit-tested without a window
// system; main.rs only wires these decisions to Tauri. See DECISIONS.md
// "D11 cont. - background lifetime (P7)".

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const LIFETIME_PREFERENCE_FILE: &str = "desktop-lifetime.json";
const LIFETIME_PREFERENCE_TEMP_FILE: &str = "desktop-lifetime.json.tmp";
const LIFETIME_PREFERENCE_VERSION: u32 = 1;

/// How long one `/v1/health` answer counts as current. It spans two probe intervals, so a
/// single slow probe during a long decode does not flip the tray.
pub const HEALTH_FRESH_FOR: Duration = Duration::from_secs(12);
pub const SUPERVISOR_TICK: Duration = Duration::from_secs(1);
pub const HEALTH_PROBE_EVERY: Duration = Duration::from_secs(5);
const FAILURES_BEFORE_NOT_ANSWERING: u32 = 2;
/// Windows truncates a notification-area tooltip at 128 UTF-16 units including the NUL.
const TOOLTIP_MAX_UTF16: usize = 127;
const DETAIL_MAX_CHARS: usize = 80;

// ---------------------------------------------------------------------------------------
// The preference
// ---------------------------------------------------------------------------------------

/// `<app_data_dir>/desktop-lifetime.json`. A native file read by Rust, never browser
/// storage: the webview's origin changes with the sidecar's port on every launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct LifetimePreference {
    pub version: u32,
    pub keep_engine_running_when_window_closes: bool,
    /// The one-time "Camelid is still running" notice has been shown.
    #[serde(default)]
    pub notice_shown: bool,
}

impl Default for LifetimePreference {
    fn default() -> Self {
        Self {
            version: LIFETIME_PREFERENCE_VERSION,
            keep_engine_running_when_window_closes: false,
            notice_shown: false,
        }
    }
}

/// A preference plus the reason, if any, that the file could not be honoured. Every
/// problem resolves to the default, so an unreadable file can only ever mean "close quits".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreferenceRead {
    pub preference: LifetimePreference,
    pub problem: Option<String>,
}

pub fn lifetime_preference_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(LIFETIME_PREFERENCE_FILE)
}

pub fn read_lifetime_preference(app_data_dir: &Path) -> PreferenceRead {
    let path = lifetime_preference_path(app_data_dir);
    let fallback = |problem: String| PreferenceRead {
        preference: LifetimePreference::default(),
        problem: Some(problem),
    };
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return PreferenceRead {
                preference: LifetimePreference::default(),
                problem: None,
            }
        }
        Err(err) => return fallback(format!("could not read {}: {err}", path.display())),
    };
    match serde_json::from_slice::<LifetimePreference>(&bytes) {
        Ok(preference) if preference.version == LIFETIME_PREFERENCE_VERSION => PreferenceRead {
            preference,
            problem: None,
        },
        // A newer build may mean something else by the same field; trusting it could keep
        // an engine alive that nobody asked to keep.
        Ok(preference) => fallback(format!(
            "{} has unsupported version {}",
            path.display(),
            preference.version
        )),
        Err(err) => fallback(format!("{} is invalid: {err}", path.display())),
    }
}

/// Temp file, sync, rename: a crash mid-write leaves either the old preference or the new
/// one, never a truncated file that would read as the default.
pub fn write_lifetime_preference_atomic(
    app_data_dir: &Path,
    preference: &LifetimePreference,
) -> Result<(), String> {
    std::fs::create_dir_all(app_data_dir)
        .map_err(|err| format!("could not create {}: {err}", app_data_dir.display()))?;
    let bytes = serde_json::to_vec_pretty(preference)
        .map_err(|err| format!("could not encode the lifetime preference: {err}"))?;
    let temp = app_data_dir.join(LIFETIME_PREFERENCE_TEMP_FILE);
    let path = lifetime_preference_path(app_data_dir);
    let written = std::fs::File::create(&temp)
        .and_then(|mut file| {
            file.write_all(&bytes)?;
            file.sync_all()
        })
        .map_err(|err| format!("could not write {}: {err}", temp.display()));
    if let Err(err) = written {
        let _ = std::fs::remove_file(&temp);
        return Err(err);
    }
    std::fs::rename(&temp, &path).map_err(|err| {
        let _ = std::fs::remove_file(&temp);
        format!("could not save {}: {err}", path.display())
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToggleOutcome {
    pub effective: bool,
    pub error: Option<String>,
}

/// The in-memory setting changes only after the write succeeds. The menu's check mark flips
/// itself on click, so without this a failed save would show ON while the next close quits.
pub fn apply_toggle(
    current: bool,
    write: impl FnOnce(bool) -> Result<(), String>,
) -> ToggleOutcome {
    let requested = !current;
    match write(requested) {
        Ok(()) => ToggleOutcome {
            effective: requested,
            error: None,
        },
        Err(error) => ToggleOutcome {
            effective: current,
            error: Some(error),
        },
    }
}

pub fn pref_error_line(error: &str) -> String {
    format!(
        "Could not save preference: {}",
        truncate_chars(error.trim(), DETAIL_MAX_CHARS)
    )
}

// ---------------------------------------------------------------------------------------
// What closing the main window does
// ---------------------------------------------------------------------------------------

/// Whether the sidecar dies with the desktop process even when none of the desktop's own
/// shutdown code runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Containment {
    /// Windows kill-on-close job: the OS kills the sidecar when the last job handle closes.
    #[cfg_attr(not(windows), allow(dead_code))]
    JobObject,
    /// `serve --exit-when-stdin-closes`: the OS closes the pipe and the engine exits itself.
    StdinPipe,
    Uncontained,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayPresence {
    Present,
    Missing(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    MacOs,
    Windows,
    Other,
}

impl Platform {
    pub fn current() -> Self {
        if cfg!(target_os = "macos") {
            Platform::MacOs
        } else if cfg!(windows) {
            Platform::Windows
        } else {
            Platform::Other
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseAction {
    HideKeepEngine,
    QuitApp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundUnavailable {
    TrayMissing,
    NoCrashProtection,
}

pub fn background_availability(
    tray: &TrayPresence,
    containment: Containment,
    platform: Platform,
) -> Result<(), BackgroundUnavailable> {
    // A hidden window with no tray leaves nothing that states the engine is running or
    // offers Quit. On Windows not even a taskbar button remains.
    if !matches!(tray, TrayPresence::Present) {
        return Err(BackgroundUnavailable::TrayMissing);
    }
    // Only the job object reaps the sidecar when the desktop dies without running any Rust
    // code. A multi-gigabyte engine is not kept alive behind a hidden window without it.
    if platform == Platform::Windows && containment != Containment::JobObject {
        return Err(BackgroundUnavailable::NoCrashProtection);
    }
    Ok(())
}

pub fn close_action(
    keep_running: bool,
    tray: &TrayPresence,
    containment: Containment,
    platform: Platform,
) -> CloseAction {
    if !keep_running {
        return CloseAction::QuitApp;
    }
    match background_availability(tray, containment, platform) {
        Ok(()) => CloseAction::HideKeepEngine,
        Err(_) => CloseAction::QuitApp,
    }
}

/// The order of the two things a background close does: raise the one-time notice, and hide
/// the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundClose {
    /// Notice first, hide only once it has been answered. A notice raised after the hide has
    /// no window left to appear on: on macOS nothing was displayed at all, so the close was
    /// indistinguishable from a quit while the engine kept serving.
    NoticeThenHide,
    HideNow,
}

/// `notice_shown` is true only once a notice has actually been acknowledged, so a run that
/// ended while one was on screen still owes it.
pub fn background_close(notice_shown: bool) -> BackgroundClose {
    if notice_shown {
        BackgroundClose::HideNow
    } else {
        BackgroundClose::NoticeThenHide
    }
}

/// App Nap is held off only while the engine is kept running behind a hidden window, so
/// the default mode's App Nap behaviour is unchanged.
pub fn wants_background_activity(keep_running: bool, main_visible: bool) -> bool {
    keep_running && !main_visible
}

pub const KEEP_RUNNING_LABEL: &str = "Keep engine running when window closes";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckItemView {
    pub text: String,
    pub checked: bool,
    pub enabled: bool,
}

/// The check mark shows what the next close will actually do, never the raw preference.
pub fn keep_running_item(
    keep_running: bool,
    availability: Result<(), BackgroundUnavailable>,
) -> CheckItemView {
    match availability {
        Ok(()) => CheckItemView {
            text: KEEP_RUNNING_LABEL.to_string(),
            checked: keep_running,
            enabled: true,
        },
        Err(reason) => CheckItemView {
            text: format!(
                "{KEEP_RUNNING_LABEL} (unavailable: {})",
                match reason {
                    BackgroundUnavailable::TrayMissing => "tray icon missing",
                    BackgroundUnavailable::NoCrashProtection => "crash protection not active",
                }
            ),
            checked: false,
            enabled: false,
        },
    }
}

// ---------------------------------------------------------------------------------------
// What the engine is doing, as observed
// ---------------------------------------------------------------------------------------

/// What `/v1/health` says about the model. `loaded_now` and `ok` are deliberately not read:
/// the busy answer (a model transition holding the registry) reports `loaded_now: false`
/// and `active_model_id: null` whatever is loaded, and `ok` is true on both branches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelObservation {
    Ready(String),
    LoadedNotReady(String),
    NotReady,
    Unknown,
}

pub fn parse_health_body(body: &str) -> ModelObservation {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return ModelObservation::Unknown;
    };
    let Some(ready) = value
        .get("generation_ready")
        .and_then(serde_json::Value::as_bool)
    else {
        return ModelObservation::Unknown;
    };
    match value.get("active_model_id") {
        Some(serde_json::Value::String(id)) if !id.is_empty() => {
            if ready {
                ModelObservation::Ready(id.clone())
            } else {
                ModelObservation::LoadedNotReady(id.clone())
            }
        }
        Some(serde_json::Value::Null) if !ready => ModelObservation::NotReady,
        _ => ModelObservation::Unknown,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeResult {
    /// `/v1/health` answered 200.
    Answered(ModelObservation),
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthObservation {
    pub at: Instant,
    pub result: ProbeResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitSummary {
    Code(i32),
    /// Only Unix reports the signal that ended a process.
    #[cfg_attr(not(unix), allow(dead_code))]
    Signal(i32),
    Unknown,
}

/// One observation of the engine slot, taken with a non-blocking `try_wait`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineSnapshot {
    pub epoch: u64,
    pub port: u16,
    pub pid: u32,
    pub exit: Option<ExitSummary>,
    pub last_stderr_line: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineStatus {
    Starting,
    Restarting,
    Running {
        port: u16,
        pid: u32,
    },
    /// Alive, and the last answer is too old to vouch for now, but nothing has failed since.
    Checking {
        port: u16,
    },
    NotAnswering {
        port: u16,
    },
    Stopped {
        exit: ExitSummary,
    },
    FailedToStart {
        title: String,
    },
}

impl EngineStatus {
    /// Stopped and FailedToStart end a generation; only a new generation moves on from them.
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            EngineStatus::Stopped { .. } | EngineStatus::FailedToStart { .. }
        )
    }
}

/// Health results for one engine generation. A result for another generation resets it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProbeHistory {
    epoch: u64,
    last_success: Option<Instant>,
    model: Option<ModelObservation>,
    consecutive_failures: u32,
}

impl ProbeHistory {
    pub fn record(&mut self, epoch: u64, observation: HealthObservation) {
        // A probe that outlived its engine must not wipe the newer engine's history.
        if epoch < self.epoch {
            return;
        }
        if epoch != self.epoch {
            *self = ProbeHistory {
                epoch,
                ..ProbeHistory::default()
            };
        }
        match observation.result {
            ProbeResult::Answered(model) => {
                self.last_success = Some(observation.at);
                self.model = Some(model);
                self.consecutive_failures = 0;
            }
            ProbeResult::Failed => {
                self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            }
        }
    }

    pub fn model_for(&self, epoch: u64) -> Option<&ModelObservation> {
        if epoch == self.epoch {
            self.model.as_ref()
        } else {
            None
        }
    }
}

/// `None` means "nothing new to say": with no engine in the slot the last published status
/// stands, so the gap between a restart's shutdown and its spawn stays "restarting".
pub fn classify(
    engine: Option<&EngineSnapshot>,
    history: &ProbeHistory,
    now: Instant,
) -> Option<EngineStatus> {
    let engine = engine?;
    // An observed exit outranks any health answer, however recent.
    if let Some(exit) = engine.exit {
        return Some(EngineStatus::Stopped { exit });
    }
    let (last_success, failures) = if history.epoch == engine.epoch {
        (history.last_success, history.consecutive_failures)
    } else {
        (None, 0)
    };
    if failures >= FAILURES_BEFORE_NOT_ANSWERING {
        return Some(EngineStatus::NotAnswering { port: engine.port });
    }
    let Some(answered_at) = last_success else {
        return Some(EngineStatus::Starting);
    };
    if now.saturating_duration_since(answered_at) <= HEALTH_FRESH_FOR {
        Some(EngineStatus::Running {
            port: engine.port,
            pid: engine.pid,
        })
    } else {
        // Timer coalescing or App Nap can stretch the probe interval. Staleness is its own
        // state rather than a verdict either way.
        Some(EngineStatus::Checking { port: engine.port })
    }
}

/// The published status, tagged with the engine generation it describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusStore {
    epoch: u64,
    last: EngineStatus,
}

impl Default for StatusStore {
    fn default() -> Self {
        Self {
            epoch: 0,
            last: EngineStatus::Starting,
        }
    }
}

impl StatusStore {
    pub fn status(&self) -> &EngineStatus {
        &self.last
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Publishes `status` for engine generation `epoch`. A result from a replaced engine is
    /// discarded, so a late answer can never republish an old port. Within a generation an
    /// ending is final: once Stopped or FailedToStart is published, no status describing a
    /// live engine replaces it. The supervisor and the tray's refresh publish from separate
    /// threads, and a snapshot taken before the exit can arrive after it.
    pub fn apply(&mut self, epoch: u64, status: EngineStatus) -> bool {
        if epoch != self.epoch {
            return false;
        }
        if self.last.is_final() && !status.is_final() {
            return false;
        }
        let changed = self.last != status;
        self.last = status;
        changed
    }

    /// Moves to a newer engine generation and publishes `status` for it in the same step.
    pub fn begin(&mut self, epoch: u64, status: EngineStatus) {
        if epoch < self.epoch {
            return;
        }
        self.epoch = epoch;
        self.last = status;
    }
}

/// Published before the old engine is shut down, so nothing between the shutdown and the
/// new spawn can show the previous engine as running.
pub fn restart_transition(store: &mut StatusStore, new_epoch: u64) {
    store.begin(new_epoch, EngineStatus::Restarting);
}

// ---------------------------------------------------------------------------------------
// The tray
// ---------------------------------------------------------------------------------------

/// The literal bind address. "(loopback only)" is a fact about the socket, not a claim
/// about who can reach it: a DNS-rebinding page in a local browser can reach a loopback
/// port.
pub fn status_line(status: &EngineStatus) -> String {
    match status {
        EngineStatus::Starting => "Engine starting\u{2026}".to_string(),
        EngineStatus::Restarting => "Engine restarting\u{2026}".to_string(),
        EngineStatus::Running { port, .. } => {
            format!("Engine running on 127.0.0.1:{port} (loopback only)")
        }
        EngineStatus::Checking { port } => format!("Checking engine on 127.0.0.1:{port}\u{2026}"),
        EngineStatus::NotAnswering { port } => {
            format!("Engine not answering on 127.0.0.1:{port}")
        }
        EngineStatus::Stopped { exit } => match exit {
            ExitSummary::Code(code) => format!("Engine stopped (exit code {code})"),
            ExitSummary::Signal(signal) => format!("Engine stopped (killed by signal {signal})"),
            ExitSummary::Unknown => "Engine stopped".to_string(),
        },
        EngineStatus::FailedToStart { title } => format!("Engine failed to start: {title}"),
    }
}

/// Never "No model loaded": the busy health answer cannot tell a transition from an empty
/// registry, and "no model ready" is true in both cases.
pub fn model_line(model: &ModelObservation) -> Option<String> {
    match model {
        ModelObservation::Ready(id) => {
            Some(format!("Model: {}", truncate_chars(id, DETAIL_MAX_CHARS)))
        }
        ModelObservation::LoadedNotReady(id) => Some(format!(
            "Model: {} (not ready)",
            truncate_chars(id, DETAIL_MAX_CHARS)
        )),
        ModelObservation::NotReady => Some("No model ready".to_string()),
        ModelObservation::Unknown => None,
    }
}

pub fn last_error_line(stderr_line: &str) -> Option<String> {
    let line = stderr_line.trim();
    if line.is_empty() {
        return None;
    }
    Some(format!(
        "Last engine message: {}",
        truncate_chars(line, DETAIL_MAX_CHARS)
    ))
}

pub fn tooltip(status: &EngineStatus) -> String {
    truncate_utf16(
        &format!("Camelid: {}", status_line(status)),
        TOOLTIP_MAX_UTF16,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrayView {
    pub status_line: String,
    pub model_line: Option<String>,
    pub last_error_line: Option<String>,
    pub pref_error_line: Option<String>,
    pub keep_running: CheckItemView,
    pub tooltip: String,
}

pub struct TrayInputs<'a> {
    pub status: &'a EngineStatus,
    pub model: Option<&'a ModelObservation>,
    pub last_stderr_line: Option<&'a str>,
    pub pref_error: Option<&'a str>,
    pub keep_running: bool,
    pub availability: Result<(), BackgroundUnavailable>,
}

pub fn tray_view(inputs: TrayInputs<'_>) -> TrayView {
    // A model line describes a running engine; next to "stopped" it would be stale.
    let model_line = match inputs.status {
        EngineStatus::Running { .. } => inputs.model.and_then(model_line),
        _ => None,
    };
    let last_error_line = match inputs.status {
        EngineStatus::Stopped { .. } => inputs.last_stderr_line.and_then(last_error_line),
        _ => None,
    };
    TrayView {
        status_line: status_line(inputs.status),
        model_line,
        last_error_line,
        pref_error_line: inputs.pref_error.map(pref_error_line),
        keep_running: keep_running_item(inputs.keep_running, inputs.availability),
        tooltip: tooltip(inputs.status),
    }
}

pub const MENU_STATUS: &str = "camelid.status";
pub const MENU_MODEL: &str = "camelid.model";
pub const MENU_LAST_ERROR: &str = "camelid.last_error";
pub const MENU_PREF_ERROR: &str = "camelid.pref_error";
pub const MENU_OPEN: &str = "camelid.open";
pub const MENU_SPOTLIGHT: &str = "camelid.spotlight";
pub const MENU_RESTART: &str = "camelid.restart";
pub const MENU_KEEP_RUNNING: &str = "camelid.keep_running";
pub const MENU_QUIT: &str = "camelid.quit";

pub const QUIT_LABEL: &str = "Quit Camelid (stops the engine)";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuAction {
    OpenMain,
    ToggleSpotlight,
    RestartEngine,
    ToggleKeepRunning,
    Quit,
}

pub fn menu_action(id: &str) -> Option<MenuAction> {
    match id {
        MENU_OPEN => Some(MenuAction::OpenMain),
        MENU_SPOTLIGHT => Some(MenuAction::ToggleSpotlight),
        MENU_RESTART => Some(MenuAction::RestartEngine),
        MENU_KEEP_RUNNING => Some(MenuAction::ToggleKeepRunning),
        MENU_QUIT => Some(MenuAction::Quit),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuEntryKind {
    Item,
    Check { checked: bool },
    Separator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuEntry {
    pub id: &'static str,
    pub text: String,
    pub enabled: bool,
    pub kind: MenuEntryKind,
}

/// The menu, top to bottom. main.rs turns each entry into a Tauri item mechanically, so
/// which lines are clickable is decided here, where it is tested.
pub fn menu_entries(view: &TrayView) -> Vec<MenuEntry> {
    let info = |id, text: String| MenuEntry {
        id,
        text,
        enabled: false,
        kind: MenuEntryKind::Item,
    };
    let action = |id, text: &str| MenuEntry {
        id,
        text: text.to_string(),
        enabled: true,
        kind: MenuEntryKind::Item,
    };
    let separator = || MenuEntry {
        id: "",
        text: String::new(),
        enabled: false,
        kind: MenuEntryKind::Separator,
    };
    let mut entries = vec![info(MENU_STATUS, view.status_line.clone())];
    if let Some(line) = &view.model_line {
        entries.push(info(MENU_MODEL, line.clone()));
    }
    if let Some(line) = &view.last_error_line {
        entries.push(info(MENU_LAST_ERROR, line.clone()));
    }
    if let Some(line) = &view.pref_error_line {
        entries.push(info(MENU_PREF_ERROR, line.clone()));
    }
    entries.push(separator());
    entries.push(action(MENU_OPEN, "Open Camelid"));
    entries.push(action(MENU_SPOTLIGHT, "Show Spotlight"));
    entries.push(action(MENU_RESTART, "Restart engine"));
    entries.push(separator());
    entries.push(MenuEntry {
        id: MENU_KEEP_RUNNING,
        text: view.keep_running.text.clone(),
        enabled: view.keep_running.enabled,
        kind: MenuEntryKind::Check {
            checked: view.keep_running.checked,
        },
    });
    entries.push(separator());
    entries.push(action(MENU_QUIT, QUIT_LABEL));
    entries
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerState {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayPointerEvent {
    Click {
        button: PointerButton,
        state: PointerState,
    },
    DoubleClick,
    Enter,
    Move,
    Leave,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayClickAction {
    ToggleSpotlight,
}

/// Left click keeps the Spotlight toggle it has had since v0.7.0; the menu is on the
/// right button.
pub fn tray_click_action(button: PointerButton, state: PointerState) -> Option<TrayClickAction> {
    match (button, state) {
        (PointerButton::Left, PointerState::Up) => Some(TrayClickAction::ToggleSpotlight),
        _ => None,
    }
}

/// Pointing at or clicking the tray requests a re-read of the child's exit status, so a
/// stretched supervisor tick does not have to be waited out. Whether the re-read lands
/// before the OS opens the menu is not established: the event and the menu rebuild both go
/// through Tauri's event loop.
pub fn tray_event_refresh(event: TrayPointerEvent) -> bool {
    matches!(
        event,
        TrayPointerEvent::Enter | TrayPointerEvent::Click { .. } | TrayPointerEvent::DoubleClick
    )
}

pub fn background_notice(platform: Platform, status: &EngineStatus) -> (String, String) {
    let place = match platform {
        Platform::MacOs => "the menu bar",
        Platform::Windows => "the notification area",
        Platform::Other => "the system tray",
    };
    (
        "Camelid is still running".to_string(),
        format!(
            "Camelid is still running in {place}.\n\n{}\n\nTo stop the engine, choose \
             \"{QUIT_LABEL}\" from the Camelid icon there.",
            status_line(status)
        ),
    )
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
    out.push('\u{2026}');
    out
}

fn truncate_utf16(text: &str, max: usize) -> String {
    if text.encode_utf16().count() <= max {
        return text.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for ch in text.chars() {
        // Leave room for the one-unit ellipsis.
        if used + ch.len_utf16() + 1 > max {
            break;
        }
        out.push(ch);
        used += ch.len_utf16();
    }
    out.push('\u{2026}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAC: Platform = Platform::MacOs;
    const WIN: Platform = Platform::Windows;

    /// Every id the tray menu may carry. `every_tray_menu_item_does_what_it_says` also
    /// checks that `menu_entries` produces nothing outside this list.
    const TRAY_MENU_IDS: [&str; 9] = [
        MENU_STATUS,
        MENU_MODEL,
        MENU_LAST_ERROR,
        MENU_PREF_ERROR,
        MENU_OPEN,
        MENU_SPOTLIGHT,
        MENU_RESTART,
        MENU_KEEP_RUNNING,
        MENU_QUIT,
    ];

    fn at(t0: Instant, seconds: u64) -> Instant {
        t0 + Duration::from_secs(seconds)
    }

    fn alive(epoch: u64, port: u16) -> EngineSnapshot {
        EngineSnapshot {
            epoch,
            port,
            pid: 4242,
            exit: None,
            last_stderr_line: None,
        }
    }

    fn answered(when: Instant) -> HealthObservation {
        HealthObservation {
            at: when,
            result: ProbeResult::Answered(ModelObservation::NotReady),
        }
    }

    fn failed(when: Instant) -> HealthObservation {
        HealthObservation {
            at: when,
            result: ProbeResult::Failed,
        }
    }

    /// The literal shape `busy_health_response` serialises while a model transition holds
    /// the registry (src/api/mod.rs).
    const BUSY_HEALTH_BODY: &str = r#"{"ok":true,"engine":"camelid","api_surface":"full","version":"0.7.3","build":"v0.7.3","loaded_now":false,"generation_ready":false,"active_context_length":null,"max_prompt_tokens":131072,"max_generation_tokens":8192,"vision_ready":false,"vision_token_allowance":null,"active_model_id":null,"q8_runtime":{},"execution_plan":null,"backend":"none","model_family":null,"gemma4_available":false,"gemma4_serve_lane":null}"#;

    #[test]
    fn lifetime_preference_defaults_to_closing_the_engine_with_the_window() {
        let dir = tempfile::tempdir().unwrap();
        let read = read_lifetime_preference(dir.path());
        assert!(!read.preference.keep_engine_running_when_window_closes);
        assert_eq!(read.problem, None);
        assert_eq!(read.preference, LifetimePreference::default());
    }

    #[test]
    fn a_corrupt_or_future_lifetime_preference_falls_back_to_quit_on_close() {
        let dir = tempfile::tempdir().unwrap();
        let path = lifetime_preference_path(dir.path());

        std::fs::write(&path, "{not json").unwrap();
        let read = read_lifetime_preference(dir.path());
        assert!(!read.preference.keep_engine_running_when_window_closes);
        assert!(read
            .problem
            .expect("a corrupt file is reported")
            .contains("invalid"));

        std::fs::write(
            &path,
            r#"{"version":2,"keep_engine_running_when_window_closes":true}"#,
        )
        .unwrap();
        let read = read_lifetime_preference(dir.path());
        assert!(!read.preference.keep_engine_running_when_window_closes);
        assert!(read
            .problem
            .expect("an unknown version is reported")
            .contains("unsupported version 2"));
    }

    #[test]
    fn lifetime_preference_round_trips_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let app_data = dir.path().join("app.camelid.desktop");
        for keep in [true, false] {
            let preference = LifetimePreference {
                keep_engine_running_when_window_closes: keep,
                ..LifetimePreference::default()
            };
            write_lifetime_preference_atomic(&app_data, &preference).expect("save");
            let read = read_lifetime_preference(&app_data);
            assert_eq!(read.problem, None);
            assert_eq!(read.preference.keep_engine_running_when_window_closes, keep);
        }
        // Beside models-directory.json and ui-storage-v1.json, in the app-data directory.
        assert_eq!(
            lifetime_preference_path(&app_data),
            app_data.join("desktop-lifetime.json")
        );
        assert!(app_data.join("desktop-lifetime.json").is_file());
        assert!(!app_data.join("desktop-lifetime.json.tmp").exists());
        // A file written before the notice field existed still reads.
        std::fs::write(
            lifetime_preference_path(&app_data),
            r#"{"version":1,"keep_engine_running_when_window_closes":true}"#,
        )
        .unwrap();
        let read = read_lifetime_preference(&app_data);
        assert!(read.preference.keep_engine_running_when_window_closes);
        assert!(!read.preference.notice_shown);
    }

    #[test]
    fn a_failed_preference_write_leaves_the_effective_setting_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        // A regular file where the app-data directory should be: create_dir_all fails on
        // every OS.
        let not_a_dir = dir.path().join("app-data");
        std::fs::write(&not_a_dir, b"file").unwrap();

        let outcome = apply_toggle(false, |requested| {
            write_lifetime_preference_atomic(
                &not_a_dir,
                &LifetimePreference {
                    keep_engine_running_when_window_closes: requested,
                    ..LifetimePreference::default()
                },
            )
        });
        assert!(
            !outcome.effective,
            "a failed save must not turn background mode on"
        );
        let error = outcome.error.expect("the failure is reported");
        assert!(pref_error_line(&error).starts_with("Could not save preference: "));
        assert!(pref_error_line(&"x".repeat(500)).chars().count() <= 27 + DETAIL_MAX_CHARS);

        let saved = apply_toggle(false, |_| Ok(()));
        assert_eq!(
            saved,
            ToggleOutcome {
                effective: true,
                error: None
            }
        );
    }

    #[test]
    fn close_with_background_off_quits_the_app_rather_than_destroying_the_window() {
        let trays = [
            TrayPresence::Present,
            TrayPresence::Missing("no icon".into()),
        ];
        let containments = [
            Containment::JobObject,
            Containment::StdinPipe,
            Containment::Uncontained,
        ];
        for tray in &trays {
            for containment in containments {
                for platform in [MAC, WIN, Platform::Other] {
                    assert_eq!(
                        close_action(false, tray, containment, platform),
                        CloseAction::QuitApp,
                        "{tray:?} {containment:?} {platform:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn close_with_background_on_hides_and_keeps_the_engine() {
        assert_eq!(
            close_action(true, &TrayPresence::Present, Containment::JobObject, WIN),
            CloseAction::HideKeepEngine
        );
        assert_eq!(
            close_action(true, &TrayPresence::Present, Containment::Uncontained, MAC),
            CloseAction::HideKeepEngine
        );
        assert_eq!(
            close_action(true, &TrayPresence::Present, Containment::StdinPipe, MAC),
            CloseAction::HideKeepEngine
        );
    }

    #[test]
    fn windows_background_requires_a_kill_on_close_job() {
        for containment in [Containment::Uncontained, Containment::StdinPipe] {
            assert_eq!(
                close_action(true, &TrayPresence::Present, containment, WIN),
                CloseAction::QuitApp
            );
            let item = keep_running_item(
                true,
                background_availability(&TrayPresence::Present, containment, WIN),
            );
            assert!(!item.checked, "the item must show what the next close does");
            assert!(!item.enabled);
            assert!(item
                .text
                .ends_with(" (unavailable: crash protection not active)"));
        }
    }

    #[test]
    fn background_requires_a_live_tray() {
        assert_eq!(
            close_action(
                true,
                &TrayPresence::Missing("no icon".into()),
                Containment::JobObject,
                WIN
            ),
            CloseAction::QuitApp
        );
        assert_eq!(
            close_action(
                true,
                &TrayPresence::Missing("x".into()),
                Containment::Uncontained,
                MAC
            ),
            CloseAction::QuitApp
        );
        let item = keep_running_item(
            true,
            background_availability(
                &TrayPresence::Missing("x".into()),
                Containment::JobObject,
                WIN,
            ),
        );
        assert!(!item.checked && !item.enabled);
        assert!(item.text.ends_with(" (unavailable: tray icon missing)"));
        // The normal case is unaffected: an available toggle shows the preference as is.
        for keep in [true, false] {
            let item = keep_running_item(keep, Ok(()));
            assert_eq!(item.checked, keep);
            assert!(item.enabled);
            assert_eq!(item.text, KEEP_RUNNING_LABEL);
        }
    }

    #[test]
    fn a_sidecar_that_exited_is_reported_stopped_not_running() {
        let t0 = Instant::now();
        let mut history = ProbeHistory::default();
        history.record(1, answered(t0));
        let mut engine = alive(1, 5000);
        engine.exit = Some(ExitSummary::Code(101));
        let status = classify(Some(&engine), &history, at(t0, 1)).unwrap();
        assert_eq!(
            status,
            EngineStatus::Stopped {
                exit: ExitSummary::Code(101)
            }
        );
        assert_eq!(status_line(&status), "Engine stopped (exit code 101)");
        assert_eq!(
            status_line(&EngineStatus::Stopped {
                exit: ExitSummary::Signal(9)
            }),
            "Engine stopped (killed by signal 9)"
        );
        assert_eq!(
            status_line(&EngineStatus::Stopped {
                exit: ExitSummary::Unknown
            }),
            "Engine stopped"
        );
    }

    #[test]
    fn running_requires_a_fresh_health_answer() {
        let t0 = Instant::now();
        let engine = alive(1, 5000);
        let mut history = ProbeHistory::default();
        assert_eq!(
            classify(Some(&engine), &history, t0),
            Some(EngineStatus::Starting),
            "alive with no answer yet is not running"
        );

        history.record(1, answered(t0));
        let running = Some(EngineStatus::Running {
            port: 5000,
            pid: 4242,
        });
        assert_eq!(classify(Some(&engine), &history, at(t0, 2)), running);

        history.record(1, failed(at(t0, 5)));
        assert_eq!(
            classify(Some(&engine), &history, at(t0, 6)),
            running,
            "one slow probe during a long decode must not flip the tray"
        );
        history.record(1, failed(at(t0, 10)));
        assert_eq!(
            classify(Some(&engine), &history, at(t0, 11)),
            Some(EngineStatus::NotAnswering { port: 5000 })
        );

        // An answer from another engine generation says nothing about this one.
        let mut other = ProbeHistory::default();
        other.record(7, answered(t0));
        assert_eq!(
            classify(Some(&engine), &other, at(t0, 1)),
            Some(EngineStatus::Starting)
        );
    }

    #[test]
    fn a_stale_observation_without_a_failure_reads_checking() {
        let t0 = Instant::now();
        let engine = alive(1, 5000);
        let mut history = ProbeHistory::default();
        history.record(1, answered(t0));
        let status = classify(Some(&engine), &history, at(t0, 13)).unwrap();
        assert_eq!(status, EngineStatus::Checking { port: 5000 });
        assert_eq!(
            status_line(&status),
            "Checking engine on 127.0.0.1:5000\u{2026}"
        );
    }

    #[test]
    fn a_health_result_from_a_replaced_engine_is_discarded() {
        let mut store = StatusStore::default();
        store.begin(2, EngineStatus::Starting);
        let changed = store.apply(1, EngineStatus::Running { port: 5000, pid: 1 });
        assert!(!changed);
        assert_eq!(store.status(), &EngineStatus::Starting);
        assert_eq!(store.epoch(), 2);
    }

    /// The supervisor takes its snapshot, probes for up to 3 s and publishes; the tray's
    /// refresh can observe the exit and publish in between. The late live snapshot must not
    /// turn the stopped engine back into a running one.
    #[test]
    fn an_observed_exit_is_never_replaced_by_a_stale_live_snapshot() {
        let t0 = Instant::now();
        let mut store = StatusStore::default();
        store.begin(1, EngineStatus::Starting);
        let mut history = ProbeHistory::default();
        history.record(1, answered(t0));
        assert!(store.apply(
            1,
            classify(Some(&alive(1, 5000)), &history, at(t0, 1)).unwrap()
        ));

        let mut dead = alive(1, 5000);
        dead.exit = Some(ExitSummary::Signal(9));
        let stopped = classify(Some(&dead), &history, at(t0, 2)).unwrap();
        assert!(store.apply(1, stopped.clone()));

        // The supervisor's probe of the dying engine failed once; its snapshot predates the
        // exit. On its own that snapshot still reads Running.
        history.record(1, failed(at(t0, 3)));
        let late = classify(Some(&alive(1, 5000)), &history, at(t0, 4)).unwrap();
        assert_eq!(
            late,
            EngineStatus::Running {
                port: 5000,
                pid: 4242
            }
        );
        assert!(!store.apply(1, late));
        assert_eq!(store.status(), &stopped, "a dead engine was republished");
        for live in [
            EngineStatus::Starting,
            EngineStatus::Restarting,
            EngineStatus::Checking { port: 5000 },
            EngineStatus::NotAnswering { port: 5000 },
        ] {
            assert!(!store.apply(1, live));
        }
        assert_eq!(store.status(), &stopped);

        // A new generation moves on.
        store.begin(2, EngineStatus::Restarting);
        assert!(store.apply(2, EngineStatus::Running { port: 6000, pid: 7 }));
    }

    #[test]
    fn a_restart_never_publishes_the_previous_port() {
        let old = EngineStatus::Running { port: 5000, pid: 1 };
        let new = EngineStatus::Running { port: 6000, pid: 2 };
        let mut store = StatusStore::default();
        store.begin(1, EngineStatus::Starting);
        store.apply(1, old.clone());
        assert_eq!(store.status(), &old);

        let mut shown = Vec::new();
        restart_transition(&mut store, 2);
        shown.push(store.status().clone());
        // A late answer from the engine being replaced.
        store.apply(1, old.clone());
        shown.push(store.status().clone());
        // Between the shutdown and the spawn there is no engine at all.
        if let Some(status) = classify(None, &ProbeHistory::default(), Instant::now()) {
            store.apply(2, status);
        }
        shown.push(store.status().clone());
        store.apply(2, EngineStatus::Starting);
        shown.push(store.status().clone());
        store.apply(2, new.clone());
        shown.push(store.status().clone());

        assert!(
            !shown.contains(&old),
            "restart republished the old port: {shown:?}"
        );
        assert_eq!(shown[0], EngineStatus::Restarting);
        assert_eq!(shown[2], EngineStatus::Restarting);
        assert_eq!(shown.last(), Some(&new));
    }

    #[test]
    fn a_busy_health_body_never_claims_no_model_is_loaded() {
        let busy = parse_health_body(BUSY_HEALTH_BODY);
        assert_eq!(busy, ModelObservation::NotReady);
        assert_eq!(model_line(&busy).as_deref(), Some("No model ready"));

        for body in [
            "",
            "{",
            "{\"ok\":true",
            "<html>",
            "[]",
            r#"{"active_model_id":null}"#,
        ] {
            assert_eq!(parse_health_body(body), ModelObservation::Unknown, "{body}");
            assert_eq!(model_line(&parse_health_body(body)), None, "{body}");
        }
        // A ready flag with no model named is contradictory; say nothing.
        assert_eq!(
            parse_health_body(r#"{"generation_ready":true,"active_model_id":null}"#),
            ModelObservation::Unknown
        );

        for model in [
            ModelObservation::Ready("m".into()),
            ModelObservation::LoadedNotReady("m".into()),
            ModelObservation::NotReady,
            ModelObservation::Unknown,
        ] {
            assert_ne!(model_line(&model).as_deref(), Some("No model loaded"));
        }
    }

    #[test]
    fn model_line_names_a_ready_model() {
        let ready = parse_health_body(
            r#"{"ok":true,"loaded_now":true,"generation_ready":true,"active_model_id":"Llama-3.2-1B-Instruct-Q8_0"}"#,
        );
        assert_eq!(
            model_line(&ready).as_deref(),
            Some("Model: Llama-3.2-1B-Instruct-Q8_0")
        );
        let loading = parse_health_body(
            r#"{"ok":true,"loaded_now":true,"generation_ready":false,"active_model_id":"Llama-3.2-1B-Instruct-Q8_0"}"#,
        );
        assert_eq!(
            model_line(&loading).as_deref(),
            Some("Model: Llama-3.2-1B-Instruct-Q8_0 (not ready)")
        );
    }

    #[test]
    fn tray_status_names_the_real_bound_address_literally() {
        let running = EngineStatus::Running {
            port: 51675,
            pid: 9,
        };
        assert_eq!(
            status_line(&running),
            "Engine running on 127.0.0.1:51675 (loopback only)"
        );
        for status in all_statuses("x") {
            assert!(!status_line(&status).contains("this computer"));
        }
    }

    fn all_statuses(title: &str) -> Vec<EngineStatus> {
        vec![
            EngineStatus::Starting,
            EngineStatus::Restarting,
            EngineStatus::Running {
                port: 65535,
                pid: u32::MAX,
            },
            EngineStatus::Checking { port: 65535 },
            EngineStatus::NotAnswering { port: 65535 },
            EngineStatus::Stopped {
                exit: ExitSummary::Code(i32::MIN),
            },
            EngineStatus::Stopped {
                exit: ExitSummary::Signal(i32::MAX),
            },
            EngineStatus::Stopped {
                exit: ExitSummary::Unknown,
            },
            EngineStatus::FailedToStart {
                title: title.to_string(),
            },
        ]
    }

    #[test]
    fn tray_tooltip_fits_the_windows_limit() {
        let long_model = ModelObservation::Ready("m".repeat(200));
        let long_stderr = "e".repeat(200);
        for status in all_statuses(&"\u{1F999}".repeat(200)) {
            let view = tray_view(TrayInputs {
                status: &status,
                model: Some(&long_model),
                last_stderr_line: Some(&long_stderr),
                pref_error: Some(&long_stderr),
                keep_running: true,
                availability: Ok(()),
            });
            assert!(
                view.tooltip.encode_utf16().count() <= 127,
                "{status:?}: {}",
                view.tooltip
            );
            assert!(view.tooltip.starts_with("Camelid: "));
            assert!(!view.tooltip.contains("mmmm") && !view.tooltip.contains("eeee"));
        }
    }

    #[test]
    fn tray_left_click_still_toggles_spotlight() {
        assert_eq!(
            tray_click_action(PointerButton::Left, PointerState::Up),
            Some(TrayClickAction::ToggleSpotlight)
        );
        assert_eq!(
            tray_click_action(PointerButton::Right, PointerState::Up),
            None
        );
        assert_eq!(
            tray_click_action(PointerButton::Left, PointerState::Down),
            None
        );
        assert_eq!(
            tray_click_action(PointerButton::Middle, PointerState::Up),
            None
        );
    }

    #[test]
    fn opening_the_tray_refreshes_the_exit_status() {
        assert!(tray_event_refresh(TrayPointerEvent::Enter));
        for button in [
            PointerButton::Left,
            PointerButton::Right,
            PointerButton::Middle,
        ] {
            for state in [PointerState::Up, PointerState::Down] {
                assert!(tray_event_refresh(TrayPointerEvent::Click {
                    button,
                    state
                }));
            }
        }
        assert!(!tray_event_refresh(TrayPointerEvent::Move));
        assert!(!tray_event_refresh(TrayPointerEvent::Leave));
    }

    #[test]
    fn every_tray_menu_item_does_what_it_says() {
        let informational = [MENU_STATUS, MENU_MODEL, MENU_LAST_ERROR, MENU_PREF_ERROR];
        for id in TRAY_MENU_IDS {
            if informational.contains(&id) {
                assert_eq!(menu_action(id), None, "{id} must not be clickable");
            } else {
                assert!(menu_action(id).is_some(), "{id} has no handler");
            }
        }

        let stopped = EngineStatus::Stopped {
            exit: ExitSummary::Signal(9),
        };
        let model = ModelObservation::Ready("m".into());
        let view = tray_view(TrayInputs {
            status: &stopped,
            model: Some(&model),
            last_stderr_line: Some("thread 'main' panicked"),
            pref_error: Some("disk full"),
            keep_running: true,
            availability: Ok(()),
        });
        // A model line next to "stopped" would describe an engine that is gone.
        assert_eq!(view.model_line, None);
        let entries = menu_entries(&view);
        let ids: Vec<&str> = entries
            .iter()
            .filter(|e| e.kind != MenuEntryKind::Separator)
            .map(|e| e.id)
            .collect();
        assert_eq!(
            ids,
            [
                MENU_STATUS,
                MENU_LAST_ERROR,
                MENU_PREF_ERROR,
                MENU_OPEN,
                MENU_SPOTLIGHT,
                MENU_RESTART,
                MENU_KEEP_RUNNING,
                MENU_QUIT
            ]
        );
        for entry in &entries {
            if entry.kind == MenuEntryKind::Separator {
                continue;
            }
            assert!(
                TRAY_MENU_IDS.contains(&entry.id),
                "{} is unlisted",
                entry.id
            );
            if informational.contains(&entry.id) {
                assert!(!entry.enabled, "{} is informational", entry.id);
            } else {
                assert!(entry.enabled, "{} must be clickable", entry.id);
                assert!(menu_action(entry.id).is_some());
            }
        }
        assert!(entries
            .iter()
            .any(|e| e.id == MENU_QUIT && e.text == "Quit Camelid (stops the engine)"));

        let running = EngineStatus::Running { port: 1, pid: 1 };
        let view = tray_view(TrayInputs {
            status: &running,
            model: Some(&model),
            last_stderr_line: Some("old"),
            pref_error: None,
            keep_running: false,
            availability: Ok(()),
        });
        assert_eq!(view.model_line.as_deref(), Some("Model: m"));
        assert_eq!(view.last_error_line, None);
    }

    #[test]
    fn background_activity_is_held_only_while_backgrounded() {
        assert!(wants_background_activity(true, false));
        assert!(!wants_background_activity(true, true));
        assert!(!wants_background_activity(false, false));
        assert!(!wants_background_activity(false, true));
    }

    #[test]
    fn the_first_background_close_raises_the_notice_before_the_window_hides() {
        assert_eq!(background_close(false), BackgroundClose::NoticeThenHide);
    }

    #[test]
    fn a_background_close_hides_at_once_once_the_notice_was_acknowledged() {
        assert_eq!(background_close(true), BackgroundClose::HideNow);
    }

    #[test]
    fn the_background_notice_says_how_to_stop_the_engine() {
        let running = EngineStatus::Running { port: 5000, pid: 1 };
        let (title, body) = background_notice(MAC, &running);
        assert_eq!(title, "Camelid is still running");
        assert!(body.contains("the menu bar"));
        assert!(body.contains("Engine running on 127.0.0.1:5000 (loopback only)"));
        assert!(body.contains("Quit Camelid (stops the engine)"));
        let (_, body) = background_notice(WIN, &EngineStatus::Starting);
        assert!(body.contains("the notification area"));
        assert!(
            !body.contains("127.0.0.1"),
            "no port is claimed before one is observed"
        );
    }
}
