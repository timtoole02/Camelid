// Camelid Desktop — additive native shell around the camelid engine.
//
// Lifecycle: open the native webview on a bundled splash, spawn `camelid serve` on a
// loopback ephemeral port as a sidecar, health-gate `/v1/health`, then navigate the window
// to the engine's already-embedded UI (UI + API same-origin). The sidecar is killed on exit;
// a Windows kill-on-close job object backstops crashes. See DECISIONS.md D11 and engine.rs.
//
// `windows_subsystem = "windows"` suppresses the console window in release builds; debug
// builds keep the console so engine stderr is visible while developing.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod engine;
mod ui_storage;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

use engine::Engine;
use ui_storage::UiStorageState;

const MODELS_DIRECTORY_PREFERENCE_FILE: &str = "models-directory.json";

/// Managed state holding the running sidecar so it can be torn down on exit.
#[derive(Default)]
struct EngineState(Mutex<Option<Engine>>);

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
/// half-started sidecar is torn down first so a retry always begins clean.
/// Async so this never runs on the main thread; the health-gated startup still
/// moves to a dedicated thread, matching the setup-time launch.
#[tauri::command]
async fn retry_startup(app: tauri::AppHandle) {
    shutdown_engine(&app);
    let snapshot = StartupSnapshot::default();
    if let Some(state) = app.try_state::<StartupState>() {
        state.replace(snapshot.clone());
    }
    let _ = app.emit("engine-status", snapshot);
    std::thread::spawn(move || start_engine(app));
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
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        if let Some(spotlight) = app.get_webview_window("spotlight") {
                            if spotlight.is_visible().unwrap_or(false) {
                                let _ = spotlight.hide();
                            } else {
                                let _ = spotlight.show();
                                let _ = spotlight.set_focus();
                            }
                        }
                    }
                })
                .build(),
        )
        .manage(EngineState::default())
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
        .setup(|app| {
            let _ = app
                .global_shortcut()
                .register("CommandOrControl+Shift+Space");

            if let Some(icon) = app.default_window_icon() {
                let _ = TrayIconBuilder::new()
                    .icon(icon.clone())
                    .tooltip("Camelid Spotlight")
                    .on_tray_icon_event(|tray, event| {
                        if let TrayIconEvent::Click {
                            button: MouseButton::Left,
                            button_state: MouseButtonState::Up,
                            ..
                        } = event
                        {
                            let app = tray.app_handle();
                            if let Some(spotlight) = app.get_webview_window("spotlight") {
                                if spotlight.is_visible().unwrap_or(false) {
                                    let _ = spotlight.hide();
                                } else {
                                    let _ = spotlight.show();
                                    let _ = spotlight.set_focus();
                                }
                            }
                        }
                    })
                    .build(app);
            }

            let handle = app.handle().clone();
            // Start the sidecar off the UI thread so the splash paints immediately.
            std::thread::spawn(move || start_engine(handle));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building camelid-desktop")
        .run(|app_handle, event| {
            if matches!(
                event,
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
            ) {
                shutdown_engine(app_handle);
            }
        });
}

/// Resolve, spawn, and health-gate the sidecar; on success navigate the window to its UI.
fn start_engine(app: tauri::AppHandle) {
    emit_status(&app, "Locating engine\u{2026}");
    let resource_dir = app.path().resource_dir().ok();
    let engine_path = match engine::resolve_engine_path(resource_dir) {
        Ok(p) => p,
        Err(e) => {
            emit_error(&app, e.splash_title(), e.splash_guidance(), &e.detail());
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
            emit_error(
                &app,
                "Model storage settings are unavailable",
                "Check access to your user application-data folder, then retry.",
                &format!("could not resolve the application data directory: {e}"),
            );
            return;
        }
    };

    emit_status(&app, "Starting engine\u{2026}");
    match engine::spawn(&engine_path, models_dir.as_deref()) {
        Ok(eng) => {
            let url = eng.base_url();
            if let Some(state) = app.try_state::<EngineState>() {
                if let Ok(mut guard) = state.inner().0.lock() {
                    *guard = Some(eng);
                }
            }
            emit_status(&app, "Engine ready. Loading\u{2026}");
            if let Some(window) = app.get_webview_window("main") {
                match tauri::Url::parse(&url) {
                    Ok(parsed) => {
                        if let Err(e) = window.navigate(parsed) {
                            emit_error(
                                &app,
                                "Engine UI could not load",
                                "Retry. If the problem persists, review the technical details.",
                                &format!("could not load the engine UI: {e}"),
                            );
                        }
                    }
                    Err(e) => emit_error(
                        &app,
                        "Engine UI could not load",
                        "Retry. If the problem persists, review the technical details.",
                        &format!("invalid engine URL {url}: {e}"),
                    ),
                }

                if let Some(spotlight) = app.get_webview_window("spotlight") {
                    if let Ok(mut parsed_spotlight) = tauri::Url::parse(&url) {
                        parsed_spotlight.set_fragment(Some("spotlight"));
                        let _ = spotlight.navigate(parsed_spotlight);
                    }
                }
            } else {
                emit_error(
                    &app,
                    "Desktop window unavailable",
                    "Quit and reopen Camelid Desktop.",
                    "internal error: main window not found",
                );
            }
        }
        Err(e) => emit_error(&app, e.splash_title(), e.splash_guidance(), &e.detail()),
    }
}

/// Kill the sidecar cleanly on shutdown. Idempotent: `take()` ensures one shutdown.
fn shutdown_engine(app_handle: &tauri::AppHandle) {
    if let Some(state) = app_handle.try_state::<EngineState>() {
        if let Ok(mut guard) = state.0.lock() {
            if let Some(mut eng) = guard.take() {
                eng.shutdown();
            }
        }
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
}
