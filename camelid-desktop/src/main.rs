// Camelid Desktop — additive native shell around the camelid engine.
//
// Lifecycle: open the native webview on a bundled splash, spawn `camelid serve` on a
// loopback ephemeral port as a sidecar, health-gate `/v1/health`, then navigate the window
// to the engine's already-embedded UI (UI + API same-origin).
//
// Closing the main window quits and stops the sidecar, unless the user turned on "Keep
// engine running when window closes" in the tray: then the window hides and the tray states
// what the engine is doing, from observation. Every quit path stops the sidecar, including
// one still inside its health gate. A Windows kill-on-close job object, and a stdin pipe the
// engine exits on where it advertises that flag, backstop crashes. See DECISIONS.md D11,
// lifetime.rs and engine.rs.
//
// `windows_subsystem = "windows"` suppresses the console window in release builds; debug
// builds keep the console so engine stderr is visible while developing.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod engine;
mod lifetime;
mod ui_storage;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State, Wry};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

use engine::{EngineHost, StartOutcome};
use lifetime::{
    CloseAction, Containment, EngineSnapshot, EngineStatus, LifetimePreference, MenuAction,
    MenuEntryKind, Platform, PointerButton, PointerState, ProbeHistory, StatusStore,
    TrayClickAction, TrayInputs, TrayPointerEvent, TrayPresence, TrayView,
};
use ui_storage::UiStorageState;

const MODELS_DIRECTORY_PREFERENCE_FILE: &str = "models-directory.json";
const MAIN_WINDOW: &str = "main";
const SPOTLIGHT_WINDOW: &str = "spotlight";
const TRAY_ID: &str = "camelid";

/// What the close handler, the tray and the supervisor share. `keep_running` is the only
/// input the close handler reads, and it changes only after the preference file is saved.
struct Lifetime {
    keep_running: AtomicBool,
    notice_shown: AtomicBool,
    pref_error: Mutex<Option<String>>,
    /// Serialises preference writes, so a toggle and the notice never interleave.
    pref_write: Mutex<()>,
    tray: Mutex<TrayPresence>,
    restarting: AtomicBool,
    observed: Mutex<Observed>,
    #[cfg(target_os = "macos")]
    activity: Mutex<Option<app_nap::Activity>>,
}

impl Default for Lifetime {
    fn default() -> Self {
        Self {
            keep_running: AtomicBool::new(false),
            notice_shown: AtomicBool::new(false),
            pref_error: Mutex::new(None),
            pref_write: Mutex::new(()),
            tray: Mutex::new(TrayPresence::Missing(
                "the tray has not been built yet".to_string(),
            )),
            restarting: AtomicBool::new(false),
            observed: Mutex::new(Observed::default()),
            #[cfg(target_os = "macos")]
            activity: Mutex::new(None),
        }
    }
}

/// The engine as last observed, and the tray view last shown for it.
#[derive(Default)]
struct Observed {
    store: StatusStore,
    history: ProbeHistory,
    last_stderr: Option<(u64, String)>,
    probe_requested: bool,
    shown: Option<TrayView>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A durable startup snapshot which the splash can replay after its JavaScript loads.
#[derive(Clone, serde::Serialize)]
struct StartupSnapshot {
    message: Option<String>,
    error: Option<StartupError>,
}

#[derive(Clone, serde::Serialize)]
struct StartupError {
    title: String,
    guidance: String,
    detail: String,
}

impl StartupSnapshot {
    fn status(message: impl Into<String>) -> Self {
        Self {
            message: Some(message.into()),
            error: None,
        }
    }

    fn error(
        title: impl Into<String>,
        guidance: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            message: None,
            error: Some(StartupError {
                title: title.into(),
                guidance: guidance.into(),
                detail: detail.into(),
            }),
        }
    }
}

impl Default for StartupSnapshot {
    fn default() -> Self {
        // Typographic ellipsis, matching every emitted status and the splash's static text.
        Self::status("Starting engine\u{2026}")
    }
}

/// Events improve responsiveness; this native state prevents early failures being lost before
/// the splash listener has registered.
#[derive(Default)]
struct StartupState(Mutex<StartupSnapshot>);

impl StartupState {
    fn replace(&self, snapshot: StartupSnapshot) {
        let mut guard = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = snapshot;
    }

    fn snapshot(&self) -> StartupSnapshot {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[tauri::command]
fn startup_snapshot(state: State<'_, StartupState>) -> StartupSnapshot {
    state.snapshot()
}

/// Re-run engine startup after the splash reports a failure, so the error pane's
/// Retry button works in place instead of demanding a quit-and-relaunch. Any
/// half-started sidecar, including one still inside its health gate, is reaped first so a
/// retry always begins clean: it lives in the engine slot from the moment it is spawned.
/// Async so this never runs on the main thread; the health-gated startup still
/// moves to a dedicated thread, matching the setup-time launch.
#[tauri::command]
async fn retry_startup(app: tauri::AppHandle) {
    restart_engine(&app);
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct ModelsDirectoryPreference {
    path: PathBuf,
}

#[derive(Debug, serde::Serialize)]
struct ModelsDirectoryChoice {
    path: Option<String>,
    restart_required: bool,
}

fn models_directory_preference_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(MODELS_DIRECTORY_PREFERENCE_FILE)
}

fn validate_models_directory(path: PathBuf) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("the models directory must be an absolute path".to_string());
    }
    let metadata = std::fs::metadata(&path)
        .map_err(|err| format!("could not access {}: {err}", path.display()))?;
    if !metadata.is_dir() {
        return Err(format!("{} is not a directory", path.display()));
    }
    path.canonicalize()
        .map_err(|err| format!("could not resolve {}: {err}", path.display()))
}

fn read_models_directory_preference(app_data_dir: &Path) -> Result<Option<PathBuf>, String> {
    let preference_path = models_directory_preference_path(app_data_dir);
    let bytes = match std::fs::read(&preference_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(format!(
                "could not read {}: {err}",
                preference_path.display()
            ))
        }
    };
    let preference: ModelsDirectoryPreference = serde_json::from_slice(&bytes)
        .map_err(|err| format!("{} is invalid: {err}", preference_path.display()))?;
    validate_models_directory(preference.path).map(Some)
}

fn write_models_directory_preference(
    app_data_dir: &Path,
    selected: PathBuf,
) -> Result<PathBuf, String> {
    let selected = validate_models_directory(selected)?;
    std::fs::create_dir_all(app_data_dir)
        .map_err(|err| format!("could not create {}: {err}", app_data_dir.display()))?;
    let preference_path = models_directory_preference_path(app_data_dir);
    let bytes = serde_json::to_vec_pretty(&ModelsDirectoryPreference {
        path: selected.clone(),
    })
    .map_err(|err| format!("could not encode the models-directory preference: {err}"))?;
    std::fs::write(&preference_path, bytes)
        .map_err(|err| format!("could not save {}: {err}", preference_path.display()))?;
    Ok(selected)
}

fn clear_models_directory_preference(app_data_dir: &Path) -> Result<(), String> {
    let preference_path = models_directory_preference_path(app_data_dir);
    match std::fs::remove_file(&preference_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!(
            "could not remove {}: {err}",
            preference_path.display()
        )),
    }
}

fn default_models_directory(app_data_dir: &Path) -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        Some(app_data_dir.join("models"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app_data_dir;
        None
    }
}

// Async on purpose: Tauri runs synchronous commands on the MAIN thread, and
// `blocking_pick_folder` parks its calling thread until the panel returns while
// the panel's modal session needs the main thread to pump events — a sync
// command therefore deadlocks the app (frozen panel, beachball cursor). An
// async command runs on a worker thread, leaving the main thread free to run
// the dialog.
#[tauri::command]
async fn choose_models_directory(
    app: tauri::AppHandle,
) -> Result<Option<ModelsDirectoryChoice>, String> {
    let Some(selected) = app
        .dialog()
        .file()
        .set_title("Choose Camelid model storage")
        .blocking_pick_folder()
    else {
        return Ok(None);
    };
    let selected = selected
        .into_path()
        .map_err(|err| format!("the selected folder is not a local filesystem path: {err}"))?;
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|err| format!("could not resolve Camelid application data: {err}"))?;
    let selected = write_models_directory_preference(&app_data_dir, selected)?;
    Ok(Some(ModelsDirectoryChoice {
        path: Some(selected.to_string_lossy().into_owned()),
        restart_required: true,
    }))
}

// Async for the same main-thread reason as choose_models_directory: this only
// does small preference-file IO, but it has no business on the UI thread.
#[tauri::command]
async fn reset_models_directory(app: tauri::AppHandle) -> Result<ModelsDirectoryChoice, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|err| format!("could not resolve Camelid application data: {err}"))?;
    clear_models_directory_preference(&app_data_dir)?;
    Ok(ModelsDirectoryChoice {
        path: default_models_directory(&app_data_dir)
            .map(|path| path.to_string_lossy().into_owned()),
        restart_required: true,
    })
}

/// Report real startup progress to the splash. Never emits a "ready" state that isn't backed
/// by a passing health check.
fn emit_status(app: &tauri::AppHandle, message: &str) {
    let snapshot = StartupSnapshot::status(message);
    if let Some(state) = app.try_state::<StartupState>() {
        state.replace(snapshot.clone());
    }
    let _ = app.emit("engine-status", snapshot);
}

/// Surface a structured, actionable failure on the splash, with raw diagnostics retained.
fn emit_error(app: &tauri::AppHandle, title: &str, guidance: &str, detail: &str) {
    let snapshot = StartupSnapshot::error(title, guidance, detail);
    if let Some(state) = app.try_state::<StartupState>() {
        state.replace(snapshot.clone());
    }
    let _ = app.emit("engine-status", snapshot);
}

fn main() {
    let builder = tauri::Builder::default();
    // Windows only, and first, so a second launch hands over to this instance before building
    // anything: relaunching a backgrounded app must not start a second engine and a second
    // model load. The other process's arguments are ignored; any process can send them. On
    // macOS the plugin listens on a fixed path in the shared /tmp, which another account can
    // own or occupy; Dock, Finder and `open -a` relaunches arrive as RunEvent::Reopen instead.
    #[cfg(windows)]
    let builder = builder.plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
        show_main_window(app)
    }));
    builder
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        toggle_spotlight(app);
                    }
                })
                .build(),
        )
        .manage(EngineHost::default())
        .manage(Lifetime::default())
        .manage(StartupState::default())
        .manage(UiStorageState::default())
        .invoke_handler(tauri::generate_handler![
            startup_snapshot,
            retry_startup,
            choose_models_directory,
            reset_models_directory,
            ui_storage::read_ui_storage,
            ui_storage::set_ui_storage_value,
            ui_storage::replace_ui_storage
        ])
        .on_window_event(|window, event| {
            if window.label() != MAIN_WINDOW {
                return;
            }
            // The `spotlight` window is created hidden and is never destroyed, so letting
            // `main` close never empties the window set and never ends the app: since
            // v0.7.0 a close left the desktop and its engine running with no window, which
            // Reopen could not bring back. Every close is therefore an explicit decision.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let app = window.app_handle();
                match decide_close(app) {
                    CloseAction::HideKeepEngine => {
                        api.prevent_close();
                        let _ = window.hide();
                        sync_background_activity(app, false);
                        show_background_notice_once(app);
                    }
                    CloseAction::QuitApp => app.exit(0),
                }
            }
        })
        .setup(|app| {
            let _ = app
                .global_shortcut()
                .register("CommandOrControl+Shift+Space");

            load_lifetime_preference(app.handle());
            let presence = match build_tray(app.handle()) {
                Ok(()) => TrayPresence::Present,
                Err(reason) => TrayPresence::Missing(reason),
            };
            match &presence {
                TrayPresence::Present => eprintln!("[desktop] lifetime: tray present"),
                TrayPresence::Missing(reason) => {
                    eprintln!("[desktop] lifetime: tray unavailable: {reason}")
                }
            }
            *lock(&app.state::<Lifetime>().tray) = presence;

            let epoch = app.state::<EngineHost>().begin_epoch();
            lock(&app.state::<Lifetime>().observed)
                .store
                .begin(epoch, EngineStatus::Starting);
            let handle = app.handle().clone();
            // Start the sidecar off the UI thread so the splash paints immediately.
            std::thread::spawn(move || start_engine(&handle, epoch));
            let handle = app.handle().clone();
            std::thread::Builder::new()
                .name("engine-supervisor".into())
                .spawn(move || supervise(handle))?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building camelid-desktop")
        .run(|app_handle, event| match event {
            // Every quit path arrives here and none is ever vetoed: the tray's Quit, Cmd+Q,
            // AppleScript quit, logout, and a close with background mode off.
            tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit => {
                shutdown_engine(app_handle)
            }
            // The Dock icon, `open -a`, and Finder relaunches of the running app.
            #[cfg(target_os = "macos")]
            tauri::RunEvent::Reopen { .. } => show_main_window(app_handle),
            _ => {}
        });
}

fn decide_close(app: &AppHandle) -> CloseAction {
    let state = app.state::<Lifetime>();
    let keep_running = state.keep_running.load(Ordering::SeqCst);
    let tray = lock(&state.tray).clone();
    let containment = app
        .state::<EngineHost>()
        .containment()
        .unwrap_or(Containment::Uncontained);
    let action = lifetime::close_action(keep_running, &tray, containment, Platform::current());
    if keep_running && action == CloseAction::QuitApp {
        eprintln!(
            "[desktop] lifetime: background refused (tray {tray:?}, containment={containment:?}); quitting"
        );
    }
    action
}

fn load_lifetime_preference(app: &AppHandle) {
    let preference = match app.path().app_data_dir() {
        Ok(dir) => {
            let read = lifetime::read_lifetime_preference(&dir);
            if let Some(problem) = read.problem {
                eprintln!("[desktop] lifetime: {problem}; closing the window will quit");
            }
            read.preference
        }
        Err(err) => {
            eprintln!(
                "[desktop] lifetime: could not resolve the application data directory ({err}); \
                 closing the window will quit"
            );
            LifetimePreference::default()
        }
    };
    let state = app.state::<Lifetime>();
    state.keep_running.store(
        preference.keep_engine_running_when_window_closes,
        Ordering::SeqCst,
    );
    state
        .notice_shown
        .store(preference.notice_shown, Ordering::SeqCst);
}

fn save_lifetime_preference(
    app: &AppHandle,
    preference: &LifetimePreference,
) -> Result<(), String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|err| format!("could not resolve the application data directory: {err}"))?;
    lifetime::write_lifetime_preference_atomic(&dir, preference)
}

fn toggle_keep_running(app: &AppHandle) {
    let state = app.state::<Lifetime>();
    {
        let _serialised = lock(&state.pref_write);
        let current = state.keep_running.load(Ordering::SeqCst);
        let notice_shown = state.notice_shown.load(Ordering::SeqCst);
        let outcome = lifetime::apply_toggle(current, |requested| {
            save_lifetime_preference(
                app,
                &LifetimePreference {
                    keep_engine_running_when_window_closes: requested,
                    notice_shown,
                    ..LifetimePreference::default()
                },
            )
        });
        state
            .keep_running
            .store(outcome.effective, Ordering::SeqCst);
        if let Some(err) = &outcome.error {
            eprintln!("[desktop] lifetime: could not save the preference: {err}");
        }
        *lock(&state.pref_error) = outcome.error;
    }
    let main_visible = app
        .get_webview_window(MAIN_WINDOW)
        .and_then(|window| window.is_visible().ok())
        .unwrap_or(true);
    sync_background_activity(app, main_visible);
    // The check mark flipped itself on click; rebuild so it shows what was actually saved.
    refresh_tray(app, true);
}

/// Once per user: a hidden window with a running engine is easy to mistake for a quit,
/// and Windows hides new notification-area icons by default.
fn show_background_notice_once(app: &AppHandle) {
    let state = app.state::<Lifetime>();
    if state.notice_shown.swap(true, Ordering::SeqCst) {
        return;
    }
    let status = lock(&state.observed).store.status().clone();
    let (title, body) = lifetime::background_notice(Platform::current(), &status);
    app.dialog()
        .message(body)
        .title(title)
        .kind(MessageDialogKind::Info)
        .show(|_| {});
    let _serialised = lock(&state.pref_write);
    let preference = LifetimePreference {
        keep_engine_running_when_window_closes: state.keep_running.load(Ordering::SeqCst),
        notice_shown: true,
        ..LifetimePreference::default()
    };
    if let Err(err) = save_lifetime_preference(app, &preference) {
        eprintln!("[desktop] lifetime: could not record that the notice was shown: {err}");
    }
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN_WINDOW) {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
    sync_background_activity(app, true);
}

fn toggle_spotlight(app: &AppHandle) {
    if let Some(spotlight) = app.get_webview_window(SPOTLIGHT_WINDOW) {
        if spotlight.is_visible().unwrap_or(false) {
            let _ = spotlight.hide();
        } else {
            let _ = spotlight.show();
            let _ = spotlight.set_focus();
        }
    }
}

fn sync_background_activity(app: &AppHandle, main_visible: bool) {
    let state = app.state::<Lifetime>();
    let wanted = lifetime::wants_background_activity(
        state.keep_running.load(Ordering::SeqCst),
        main_visible,
    );
    #[cfg(target_os = "macos")]
    {
        let mut held = lock(&state.activity);
        if wanted && held.is_none() {
            *held = Some(app_nap::Activity::begin());
        } else if !wanted {
            if let Some(activity) = held.take() {
                activity.end();
            }
        }
    }
    // App Nap is macOS-only; nothing else throttles a hidden app's timers.
    #[cfg(not(target_os = "macos"))]
    let _ = wanted;
}

/// Hiding every window is App Nap's trigger, and a napping app's sleeps stretch, which would
/// leave the tray's status stale. The activity is held only while the engine is kept running
/// behind a hidden window; it still allows idle system sleep.
#[cfg(target_os = "macos")]
mod app_nap {
    use objc2::rc::Retained;
    use objc2::runtime::{NSObjectProtocol, ProtocolObject};
    use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};

    pub struct Activity(Retained<ProtocolObject<dyn NSObjectProtocol>>);

    // SAFETY: the token is opaque. It is only ever handed back to
    // -[NSProcessInfo endActivity:], and NSProcessInfo is documented as thread-safe.
    unsafe impl Send for Activity {}

    impl Activity {
        pub fn begin() -> Activity {
            let reason = NSString::from_str("Camelid engine is serving with the window closed");
            Activity(
                NSProcessInfo::processInfo().beginActivityWithOptions_reason(
                    NSActivityOptions::UserInitiatedAllowingIdleSystemSleep,
                    &reason,
                ),
            )
        }

        pub fn end(self) {
            // SAFETY: the token came from beginActivityWithOptions:reason: on the same
            // process-wide NSProcessInfo, and is ended exactly once.
            unsafe { NSProcessInfo::processInfo().endActivity(&self.0) };
        }
    }
}

/// One restart at a time. Restarting is published before the old engine is reaped, so the
/// tray can never show the old port as running in between.
fn restart_engine(app: &AppHandle) {
    let state = app.state::<Lifetime>();
    if state.restarting.swap(true, Ordering::SeqCst) {
        return;
    }
    let host = app.state::<EngineHost>();
    if host.is_quitting() {
        state.restarting.store(false, Ordering::SeqCst);
        return;
    }
    let epoch = host.begin_epoch();
    lifetime::restart_transition(&mut lock(&state.observed).store, epoch);
    refresh_tray(app, false);
    host.reap();
    let snapshot = StartupSnapshot::default();
    if let Some(startup) = app.try_state::<StartupState>() {
        startup.replace(snapshot.clone());
    }
    let _ = app.emit("engine-status", snapshot);
    let handle = app.clone();
    std::thread::spawn(move || {
        start_engine(&handle, epoch);
        handle
            .state::<Lifetime>()
            .restarting
            .store(false, Ordering::SeqCst);
    });
}

fn publish_status(app: &AppHandle, epoch: u64, status: EngineStatus) {
    lock(&app.state::<Lifetime>().observed)
        .store
        .apply(epoch, status);
    refresh_tray(app, false);
}

fn request_health_probe(app: &AppHandle) {
    lock(&app.state::<Lifetime>().observed).probe_requested = true;
}

/// Watches the engine for the tray: `try_wait` every tick, `/v1/health` every few seconds.
/// It reports; it never restarts anything.
fn supervise(app: AppHandle) {
    let mut last_probe: Option<(u64, Instant)> = None;
    loop {
        let host = app.state::<EngineHost>();
        if host.is_quitting() {
            return;
        }
        let snapshot = host.observe();
        if let Some(engine) = snapshot.as_ref().filter(|engine| engine.exit.is_none()) {
            let requested =
                std::mem::take(&mut lock(&app.state::<Lifetime>().observed).probe_requested);
            let due = requested
                || last_probe.is_none_or(|(epoch, at)| {
                    epoch != engine.epoch || at.elapsed() >= lifetime::HEALTH_PROBE_EVERY
                });
            if due {
                let observation = engine::fetch_health(engine.port);
                last_probe = Some((engine.epoch, Instant::now()));
                lock(&app.state::<Lifetime>().observed)
                    .history
                    .record(engine.epoch, observation);
            }
        }
        publish_observed(&app, snapshot.as_ref());
        std::thread::sleep(lifetime::SUPERVISOR_TICK);
    }
}

fn publish_observed(app: &AppHandle, snapshot: Option<&EngineSnapshot>) {
    {
        let state = app.state::<Lifetime>();
        let mut observed = lock(&state.observed);
        if let Some(engine) = snapshot {
            if let Some(line) = &engine.last_stderr_line {
                observed.last_stderr = Some((engine.epoch, line.clone()));
            }
            if let Some(status) = lifetime::classify(snapshot, &observed.history, Instant::now()) {
                observed.store.apply(engine.epoch, status);
            }
        }
    }
    refresh_tray(app, false);
}

/// Rebuilds the tray menu when what it would say has changed. Always on the main thread.
fn refresh_tray(app: &AppHandle, force: bool) {
    let state = app.state::<Lifetime>();
    let tray = lock(&state.tray).clone();
    if tray != TrayPresence::Present {
        return;
    }
    let containment = app
        .state::<EngineHost>()
        .containment()
        .unwrap_or(Containment::Uncontained);
    let availability = lifetime::background_availability(&tray, containment, Platform::current());
    let pref_error = lock(&state.pref_error).clone();
    let keep_running = state.keep_running.load(Ordering::SeqCst);
    let view = {
        let mut observed = lock(&state.observed);
        let epoch = observed.store.epoch();
        let stderr = observed
            .last_stderr
            .as_ref()
            .filter(|(tagged, _)| *tagged == epoch)
            .map(|(_, line)| line.clone());
        let view = lifetime::tray_view(TrayInputs {
            status: observed.store.status(),
            model: observed.history.model_for(epoch),
            last_stderr_line: stderr.as_deref(),
            pref_error: pref_error.as_deref(),
            keep_running,
            availability,
        });
        if !force && observed.shown.as_ref() == Some(&view) {
            return;
        }
        observed.shown = Some(view.clone());
        view
    };
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || apply_tray_view(&handle, &view));
}

fn apply_tray_view(app: &AppHandle, view: &TrayView) {
    let Some(tray) = app.tray_by_id(TRAY_ID) else {
        return;
    };
    match build_menu(app, view) {
        Ok(menu) => {
            if let Err(err) = tray.set_menu(Some(menu)) {
                eprintln!("[desktop] lifetime: could not replace the tray menu: {err}");
            }
        }
        Err(err) => eprintln!("[desktop] lifetime: could not build the tray menu: {err}"),
    }
    let _ = tray.set_tooltip(Some(&view.tooltip));
}

fn build_menu(app: &AppHandle, view: &TrayView) -> tauri::Result<Menu<Wry>> {
    let menu = Menu::new(app)?;
    for entry in lifetime::menu_entries(view) {
        match entry.kind {
            MenuEntryKind::Separator => menu.append(&PredefinedMenuItem::separator(app)?)?,
            MenuEntryKind::Item => menu.append(&MenuItem::with_id(
                app,
                entry.id,
                &entry.text,
                entry.enabled,
                None::<&str>,
            )?)?,
            MenuEntryKind::Check { checked } => menu.append(&CheckMenuItem::with_id(
                app,
                entry.id,
                &entry.text,
                entry.enabled,
                checked,
                None::<&str>,
            )?)?,
        }
    }
    Ok(menu)
}

/// Left click keeps toggling Spotlight; the menu opens on the right button.
fn build_tray(app: &AppHandle) -> Result<(), String> {
    let icon = app
        .default_window_icon()
        .cloned()
        .ok_or_else(|| "the app has no window icon to show".to_string())?;
    let initial = lifetime::tray_view(TrayInputs {
        status: &EngineStatus::Starting,
        model: None,
        last_stderr_line: None,
        pref_error: None,
        keep_running: app.state::<Lifetime>().keep_running.load(Ordering::SeqCst),
        availability: Ok(()),
    });
    let menu = build_menu(app, &initial).map_err(|err| err.to_string())?;
    TrayIconBuilder::with_id(TRAY_ID)
        .icon(icon)
        .tooltip(&initial.tooltip)
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_tray_icon_event(|tray, event| on_tray_event(tray.app_handle(), &event))
        .on_menu_event(|app, event| on_menu_event(app, event.id().as_ref()))
        .build(app)
        .map_err(|err| err.to_string())?;
    Ok(())
}

fn on_tray_event(app: &AppHandle, event: &TrayIconEvent) {
    let Some(pointer) = pointer_event(event) else {
        return;
    };
    if lifetime::tray_event_refresh(pointer) {
        // try_wait never blocks, so this is safe on the event thread, and the menu about to
        // be read reflects an exit the supervisor has not ticked over yet.
        let snapshot = app.state::<EngineHost>().observe();
        request_health_probe(app);
        publish_observed(app, snapshot.as_ref());
    }
    if let TrayPointerEvent::Click { button, state } = pointer {
        if lifetime::tray_click_action(button, state) == Some(TrayClickAction::ToggleSpotlight) {
            toggle_spotlight(app);
        }
    }
}

fn pointer_event(event: &TrayIconEvent) -> Option<TrayPointerEvent> {
    Some(match event {
        TrayIconEvent::Click {
            button,
            button_state,
            ..
        } => TrayPointerEvent::Click {
            button: match button {
                MouseButton::Left => PointerButton::Left,
                MouseButton::Right => PointerButton::Right,
                MouseButton::Middle => PointerButton::Middle,
            },
            state: match button_state {
                MouseButtonState::Up => PointerState::Up,
                MouseButtonState::Down => PointerState::Down,
            },
        },
        TrayIconEvent::DoubleClick { .. } => TrayPointerEvent::DoubleClick,
        TrayIconEvent::Enter { .. } => TrayPointerEvent::Enter,
        TrayIconEvent::Move { .. } => TrayPointerEvent::Move,
        TrayIconEvent::Leave { .. } => TrayPointerEvent::Leave,
        _ => return None,
    })
}

fn on_menu_event(app: &AppHandle, id: &str) {
    match lifetime::menu_action(id) {
        Some(MenuAction::OpenMain) => show_main_window(app),
        Some(MenuAction::ToggleSpotlight) => toggle_spotlight(app),
        Some(MenuAction::RestartEngine) => restart_engine(app),
        Some(MenuAction::ToggleKeepRunning) => toggle_keep_running(app),
        Some(MenuAction::Quit) => app.exit(0),
        None => {}
    }
}

/// Resolve, spawn, and health-gate the sidecar; on success navigate the window to its UI.
/// Everything is tagged with `epoch`: a start that a restart or quit superseded reports
/// nothing, so a stale failure can never replace a newer engine's splash or tray status.
fn start_engine(app: &AppHandle, epoch: u64) {
    let host = app.state::<EngineHost>();
    let fail = |title: &str, guidance: &str, detail: &str| {
        if host.is_current(epoch) {
            emit_error(app, title, guidance, detail);
            publish_status(
                app,
                epoch,
                EngineStatus::FailedToStart {
                    title: title.to_string(),
                },
            );
        }
    };
    emit_status(app, "Locating engine\u{2026}");
    let resource_dir = app.path().resource_dir().ok();
    let engine_path = match engine::resolve_engine_path(resource_dir) {
        Ok(p) => p,
        Err(e) => {
            fail(e.splash_title(), e.splash_guidance(), &e.detail());
            return;
        }
    };

    // A user choice overrides the platform default. The directory is selected
    // before launch and never changed underneath a running engine.
    let models_dir = match app.path().app_data_dir() {
        Ok(app_data_dir) => match read_models_directory_preference(&app_data_dir) {
            Ok(Some(path)) => Some(path),
            Ok(None) => default_models_directory(&app_data_dir),
            Err(err) => {
                eprintln!(
                    "[desktop] custom model storage is unavailable ({err}); using platform default"
                );
                default_models_directory(&app_data_dir)
            }
        },
        Err(e) => {
            fail(
                "Model storage settings are unavailable",
                "Check access to your user application-data folder, then retry.",
                &format!("could not resolve the application data directory: {e}"),
            );
            return;
        }
    };

    // Optional engine flags are passed only when this engine lists them, so a stale engine
    // beside a fresh desktop, or one found on PATH, still starts exactly as before.
    let (features, probe_log) =
        engine::features_or_none(engine::probe_engine_features(&engine_path));
    eprintln!("{probe_log}");

    emit_status(app, "Starting engine\u{2026}");
    publish_status(app, epoch, EngineStatus::Starting);
    match host.start(epoch, &engine_path, models_dir.as_deref(), &features) {
        StartOutcome::Ready { port } => {
            if let Some(containment) = host.containment() {
                eprintln!("[desktop] lifetime: containment={containment:?}");
            }
            request_health_probe(app);
            let url = engine::base_url(port);
            emit_status(app, "Engine ready. Loading\u{2026}");
            if let Some(window) = app.get_webview_window(MAIN_WINDOW) {
                match tauri::Url::parse(&url) {
                    Ok(parsed) => {
                        if let Err(e) = window.navigate(parsed) {
                            emit_error(
                                app,
                                "Engine UI could not load",
                                "Retry. If the problem persists, review the technical details.",
                                &format!("could not load the engine UI: {e}"),
                            );
                        }
                    }
                    Err(e) => emit_error(
                        app,
                        "Engine UI could not load",
                        "Retry. If the problem persists, review the technical details.",
                        &format!("invalid engine URL {url}: {e}"),
                    ),
                }

                if let Some(spotlight) = app.get_webview_window(SPOTLIGHT_WINDOW) {
                    if let Ok(mut parsed_spotlight) = tauri::Url::parse(&url) {
                        parsed_spotlight.set_fragment(Some("spotlight"));
                        let _ = spotlight.navigate(parsed_spotlight);
                    }
                }
            } else {
                emit_error(
                    app,
                    "Desktop window unavailable",
                    "Quit and reopen Camelid Desktop.",
                    "internal error: main window not found",
                );
            }
        }
        StartOutcome::Failed(e) => fail(e.splash_title(), e.splash_guidance(), &e.detail()),
        StartOutcome::Cancelled => {}
    }
}

/// Stops the sidecar on every quit path, including one still inside its health gate, and
/// refuses any start after it. Idempotent: the slot is empty after the first call.
fn shutdown_engine(app_handle: &AppHandle) {
    if let Some(host) = app_handle.try_state::<EngineHost>() {
        if let Some(reaped) = host.shutdown_for_exit() {
            eprintln!(
                "[desktop] engine pid {} stopped{} ({})",
                reaped.pid,
                if reaped.was_pending {
                    " during its health gate"
                } else {
                    ""
                },
                reaped
                    .status
                    .map_or_else(|| "exit status unknown".to_string(), |s| s.to_string())
            );
        }
    }
    if app_handle.try_state::<Lifetime>().is_some() {
        sync_background_activity(app_handle, true);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        clear_models_directory_preference, read_models_directory_preference,
        write_models_directory_preference, StartupSnapshot, StartupState,
    };

    #[test]
    fn startup_error_is_replayable_after_the_listener_would_register() {
        let state = StartupState::default();
        state.replace(StartupSnapshot::error(
            "Camelid engine is missing",
            "Restore camelid.exe and retry.",
            "failed to launch camelid.exe",
        ));

        let snapshot = state.snapshot();
        let error = snapshot.error.expect("early error remains available");
        assert_eq!(error.title, "Camelid engine is missing");
        assert_eq!(error.guidance, "Restore camelid.exe and retry.");
        assert_eq!(error.detail, "failed to launch camelid.exe");
    }

    #[test]
    fn models_directory_preference_round_trips_and_resets() {
        let root = tempfile::tempdir().unwrap();
        let app_data = root.path().join("app-data");
        let selected = root.path().join("model-library");
        std::fs::create_dir(&selected).unwrap();

        let saved =
            write_models_directory_preference(&app_data, selected.clone()).expect("save choice");
        assert_eq!(saved, selected.canonicalize().unwrap());
        assert_eq!(
            read_models_directory_preference(&app_data).unwrap(),
            Some(selected.canonicalize().unwrap())
        );

        clear_models_directory_preference(&app_data).unwrap();
        assert_eq!(read_models_directory_preference(&app_data).unwrap(), None);
    }

    #[test]
    fn models_directory_preference_rejects_files_and_relative_paths() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("not-a-folder");
        std::fs::write(&file, b"fixture").unwrap();
        assert!(write_models_directory_preference(root.path(), file).is_err());
        assert!(
            write_models_directory_preference(root.path(), PathBuf::from("relative/models"))
                .is_err()
        );
    }

    /// Every capability that owns the `main` window must grant start-dragging.
    ///
    /// This is a regression guard for a bug that shipped in v0.7.0 and made the
    /// macOS window impossible to move at all. The window is `titleBarStyle:
    /// "Overlay"` with `hiddenTitle`, so the OS titlebar is ours to draw and the
    /// app's own `data-tauri-drag-region` strip is the window's ONLY drag
    /// handle. That strip fires `startDragging` over IPC — and `core:default`
    /// does not carry the permission for it: `core:window:default` is a
    /// read-only set (sizes, positions, `is-*`, monitors, theme, and
    /// `internal-toggle-maximize`).
    ///
    /// The failure is silent and easy to reintroduce, because the ACL denial
    /// produces no build error and no crash — just a window that will not move,
    /// while double-click-to-zoom keeps working and makes it look like the
    /// titlebar is fine. Assert the grant instead of trusting a review to spot
    /// its absence.
    ///
    /// `spotlight-ui` is deliberately excluded: that window is a centred,
    /// always-on-top overlay that is positioned, never dragged.
    #[test]
    fn main_window_capabilities_grant_start_dragging() {
        const DRAG: &str = "core:window:allow-start-dragging";
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("capabilities");
        let mut checked = 0;

        for entry in std::fs::read_dir(&dir).expect("capabilities directory") {
            let path = entry.expect("capability entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let raw = std::fs::read_to_string(&path).expect("read capability");
            let value: serde_json::Value = serde_json::from_str(&raw).expect("capability is JSON");

            let owns_main = value["windows"]
                .as_array()
                .is_some_and(|w| w.iter().any(|entry| entry == "main"));
            if !owns_main {
                continue;
            }

            let permissions = value["permissions"].as_array().expect("permissions array");
            assert!(
                permissions.iter().any(|p| p == DRAG),
                "{} owns the main window but does not grant {DRAG}; the macOS window \
                 cannot be moved without it",
                path.display()
            );
            checked += 1;
        }

        // Both the splash capability and the remote loopback one own `main`. If
        // this count ever drops, a capability was renamed or removed and the
        // loop above silently stopped checking anything.
        assert_eq!(
            checked, 2,
            "expected exactly two capabilities owning the main window"
        );
    }
}
