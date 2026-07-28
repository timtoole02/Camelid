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

use std::sync::Mutex;
use tauri::{Emitter, Manager, State};

use engine::Engine;

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
        Self::status("Starting engine...")
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
        .manage(EngineState::default())
        .manage(StartupState::default())
        .invoke_handler(tauri::generate_handler![startup_snapshot])
        .setup(|app| {
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

    // A signed macOS app bundle is immutable application code. Keep downloaded GGUFs in
    // the per-user Application Support directory instead of beside the bundled sidecar
    // under `Camelid Desktop.app/Contents/Resources`. Windows deliberately retains its
    // existing per-user installer layout with `models/` beside `camelid.exe`.
    #[cfg(target_os = "macos")]
    let models_dir = match app.path().app_data_dir() {
        Ok(path) => Some(path.join("models")),
        Err(e) => {
            emit_error(
                &app,
                "Model storage is unavailable",
                "Check access to your user Library folder, then retry.",
                &format!("could not resolve the Application Support directory: {e}"),
            );
            return;
        }
    };
    #[cfg(not(target_os = "macos"))]
    let models_dir: Option<std::path::PathBuf> = None;

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
                                "Retry the desktop app. If it persists, review the technical details.",
                                &format!("could not load the engine UI: {e}"),
                            );
                        }
                    }
                    Err(e) => emit_error(
                        &app,
                        "Engine UI could not load",
                        "Retry the desktop app. If it persists, review the technical details.",
                        &format!("invalid engine URL {url}: {e}"),
                    ),
                }
            } else {
                emit_error(
                    &app,
                    "Desktop window unavailable",
                    "Close Camelid Desktop and try again.",
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
    use super::{StartupSnapshot, StartupState};

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
}
