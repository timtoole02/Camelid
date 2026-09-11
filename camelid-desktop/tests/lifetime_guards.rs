//! Background-lifetime source guards (see `../DECISIONS.md`, "D11 cont. - background
//! lifetime (P7)"). Each asserts a shape whose loss would compile, pass every unit test, and
//! only show up as a desktop that cannot be quit, an orphaned multi-gigabyte engine, or a
//! tray that lost its Spotlight click. The behavioural closure for each is a live receipt;
//! these keep the shape from quietly regressing between receipts.
//!
//! Production code only: everything from the first line that is exactly `#[cfg(test)]` on
//! is ignored, because test modules legitimately name the things production must not do.
//! Needles are single lines, so a CRLF checkout cannot make a multi-line anchor silently
//! miss.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn production_source(relative: &str) -> String {
    let path = repo_root().join(relative);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut production = String::new();
    for line in text.lines() {
        if line.trim() == "#[cfg(test)]" {
            break;
        }
        production.push_str(line);
        production.push('\n');
    }
    assert!(
        production.lines().count() > 50,
        "{relative}: suspiciously little production code; did the cut-off move?"
    );
    production
}

fn desktop_sources() -> Vec<(String, String)> {
    let dir = repo_root().join("camelid-desktop/src");
    let mut sources = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("camelid-desktop/src") {
        let path = entry.expect("source entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let relative = format!(
            "camelid-desktop/src/{}",
            path.file_name().unwrap().to_string_lossy()
        );
        let text = std::fs::read_to_string(&path).expect("read source");
        let production: String = text
            .lines()
            .take_while(|line| line.trim() != "#[cfg(test)]")
            .map(|line| format!("{line}\n"))
            .collect();
        sources.push((relative, production));
    }
    assert!(
        sources.len() >= 4,
        "expected main.rs, engine.rs, lifetime.rs and ui_storage.rs"
    );
    sources
}

/// Vetoing an exit would turn Cmd+Q, AppleScript quit, logout and the tray's Quit into
/// no-ops while the engine keeps its model resident, and both macOS upgrade scripts would
/// then give up waiting for the app to exit.
#[test]
fn desktop_never_prevents_app_exit() {
    for (file, source) in desktop_sources() {
        for (number, line) in source.lines().enumerate() {
            assert!(
                !line.contains("prevent_exit"),
                "{file}:{}: an exit veto breaks every quit path: {line}",
                number + 1
            );
        }
    }
}

/// Background lifetime hides the window; it never detaches the engine. A detached sidecar
/// escapes the kill-on-close job, so a desktop crash would orphan it on Windows.
#[test]
fn sidecar_is_never_detached_from_its_job() {
    let engine = production_source("camelid-desktop/src/engine.rs");
    assert!(
        engine.contains("JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE"),
        "the Windows job no longer kills the sidecar when the desktop dies"
    );
    for detach in [
        "BREAKAWAY",
        "DETACHED_PROCESS",
        "CREATE_NEW_PROCESS_GROUP",
        "setsid",
    ] {
        assert!(
            !engine.contains(detach),
            "engine.rs detaches the sidecar ({detach})"
        );
    }
}

/// P7 keeps the engine on loopback. Serving other devices needs its own confirmation, key
/// and transport (P7b), never a quiet bind change here.
#[test]
fn sidecar_still_binds_loopback_only() {
    let engine = production_source("camelid-desktop/src/engine.rs");
    assert!(
        engine.contains("\"127.0.0.1:{port}\""),
        "the sidecar address is no longer the loopback literal"
    );
    for exposure in [
        "0.0.0.0",
        "--allow-unauthenticated-remote",
        "--allow-cleartext-remote",
    ] {
        assert!(
            !engine.contains(exposure),
            "engine.rs exposes the sidecar beyond loopback ({exposure})"
        );
    }
}

/// Source-level only: the live receipt (left-click opens Spotlight, not the menu) is the
/// behavioural closure.
#[test]
fn tray_menu_does_not_steal_the_spotlight_click() {
    let main = production_source("camelid-desktop/src/main.rs");
    assert!(
        main.contains(".show_menu_on_left_click(false)"),
        "the tray menu would open on left click, replacing the Spotlight toggle"
    );
}

/// Any process in the session can send the single-instance plugin arguments and a working
/// directory, so the callback must not act on them.
#[test]
fn single_instance_callback_ignores_foreign_arguments() {
    let main = production_source("camelid-desktop/src/main.rs");
    assert!(
        main.contains("tauri_plugin_single_instance::init(|app, _argv, _cwd|"),
        "the single-instance callback is missing or reads another process's arguments"
    );
}

/// True when a line containing `first` also contains `then`, or its next non-blank line
/// does: rustfmt may break a match arm or a closure after its head.
fn line_then(source: &str, first: &str, then: &str) -> bool {
    let lines: Vec<&str> = source.lines().collect();
    lines.iter().enumerate().any(|(index, line)| {
        line.contains(first)
            && (line.contains(then)
                || lines[index + 1..]
                    .iter()
                    .find(|next| !next.trim().is_empty())
                    .is_some_and(|next| next.contains(then)))
    })
}

/// Source-level. With background mode off, a close must request the exit. Letting it proceed
/// destroys `main`, but the hidden Spotlight window is never destroyed, so the app would never
/// exit: the v0.7.0 regression. The live receipt is the behavioural closure.
#[test]
fn closing_with_background_off_requests_the_exit() {
    let main = production_source("camelid-desktop/src/main.rs");
    assert!(
        main.contains("CloseAction::QuitApp => app.exit(0),"),
        "a close that should quit no longer requests the exit"
    );
}

/// Source-level. Reopen (the Dock icon, `open -a`) and a second launch must bring the hidden
/// window back; without them a backgrounded app can only be reached from the tray.
#[test]
fn reopen_and_a_second_launch_show_the_main_window() {
    let main = production_source("camelid-desktop/src/main.rs");
    assert!(
        main.contains("tauri::RunEvent::Reopen { .. } => show_main_window(app_handle),"),
        "macOS Reopen no longer shows the main window"
    );
    assert!(
        line_then(
            &main,
            "tauri_plugin_single_instance::init(|app, _argv, _cwd|",
            "show_main_window(app)"
        ),
        "a second launch no longer shows the running instance's window"
    );
}

/// On macOS the plugin's socket is a fixed path in the shared /tmp: another account's stale
/// socket makes it launch with no listener, and a socket another account binds first swallows
/// every launch. It is compiled and registered on Windows only, where it keys on a named mutex
/// and a window class local to the session. macOS relaunches arrive as RunEvent::Reopen.
#[test]
fn single_instance_plugin_is_windows_only() {
    let manifest_path = repo_root().join("camelid-desktop/Cargo.toml");
    let manifest =
        std::fs::read_to_string(&manifest_path).expect("read camelid-desktop/Cargo.toml");
    let mut section = String::new();
    let mut found = Vec::new();
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            section = trimmed.to_string();
        } else if trimmed.starts_with("tauri-plugin-single-instance") {
            found.push(section.clone());
        }
    }
    assert_eq!(
        found,
        ["[target.'cfg(windows)'.dependencies]"],
        "tauri-plugin-single-instance must be a Windows-only dependency"
    );

    let main = production_source("camelid-desktop/src/main.rs");
    let lines: Vec<&str> = main.lines().collect();
    let registrations: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.contains("tauri_plugin_single_instance::init("))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(registrations.len(), 1, "expected one registration");
    let attribute = lines[..registrations[0]]
        .iter()
        .rev()
        .map(|line| line.trim())
        .find(|line| !line.is_empty() && !line.starts_with("//"));
    assert_eq!(
        attribute,
        Some("#[cfg(windows)]"),
        "the single-instance plugin is registered outside Windows"
    );
}

/// Source-level. Every exit reaches the one engine shutdown, and the tray's Quit asks for an
/// exit rather than doing something narrower.
#[test]
fn every_quit_path_reaches_the_engine_shutdown() {
    let main = production_source("camelid-desktop/src/main.rs");
    assert!(
        line_then(
            &main,
            "tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit =>",
            "shutdown_engine(app_handle)"
        ),
        "an exit no longer stops the engine"
    );
    assert!(
        main.contains("Some(MenuAction::Quit) => app.exit(0),"),
        "the tray's Quit no longer requests an exit"
    );
}

/// Both macOS upgrade paths quit the app with the same Apple Event logout sends, then wait
/// for the desktop AND the sidecar to be gone. A hard kill would skip the engine shutdown,
/// and dropping the sidecar wait would replace the bundle under a running engine.
#[test]
fn upgrade_scripts_still_quit_the_app_and_wait_on_the_sidecar() {
    for script in [
        "scripts/install-macos-desktop.sh",
        "scripts/get-desktop-macos.sh",
    ] {
        let path = repo_root().join(script);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert!(
            text.contains("tell application \"Camelid Desktop\" to quit"),
            "{script} no longer asks the app to quit"
        );
        assert!(
            text.contains("pgrep -f \"$sidecar_process\""),
            "{script} no longer waits for the sidecar to exit"
        );
        for hard_kill in ["kill -9", "pkill -9"] {
            assert!(
                !text.contains(hard_kill),
                "{script} hard-kills the app ({hard_kill})"
            );
        }
    }
}
