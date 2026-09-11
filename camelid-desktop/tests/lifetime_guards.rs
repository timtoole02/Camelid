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

/// Source-level. Each clickable tray item reaches the handler its label names. The ids are
/// mapped in lifetime.rs, where they are unit-tested; this pins the last hop in main.rs.
#[test]
fn every_tray_menu_action_reaches_its_handler() {
    let main = production_source("camelid-desktop/src/main.rs");
    let arms = [
        "Some(MenuAction::OpenMain) => show_main_window(app),",
        "Some(MenuAction::ToggleSpotlight) => toggle_spotlight(app),",
        "Some(MenuAction::RestartEngine) => restart_engine(app),",
        "Some(MenuAction::ToggleKeepRunning) => toggle_keep_running(app),",
        "Some(MenuAction::Quit) => app.exit(0),",
    ];
    for arm in arms {
        assert!(
            main.contains(arm),
            "tray menu arm missing or rerouted: {arm}"
        );
    }
    assert_eq!(
        main.matches("Some(MenuAction::").count(),
        arms.len(),
        "a menu action is handled somewhere else too"
    );
}

/// The lines of the top-level `fn` whose header contains `header`, up to its closing brace.
fn fn_body<'a>(source: &'a str, header: &str) -> Vec<&'a str> {
    let body: Vec<&str> = source
        .lines()
        .skip_while(|line| !line.contains(header))
        .take_while(|line| *line != "}")
        .collect();
    assert!(!body.is_empty(), "no function matching {header}");
    body
}

/// Source-level, beside the behavioural `a_restart_moves_the_tray_to_the_new_engine_and_reaps
/// _the_old_one`. `begin_epoch` is private, so the app can only open a generation through
/// `begin_start` or `begin_restart`; these pin which one each site uses, and that `start`
/// keeps going through the slot-holding `start_with`.
#[test]
fn every_engine_generation_opens_through_the_host() {
    let main = production_source("camelid-desktop/src/main.rs");
    assert!(
        fn_body(&main, "fn restart_engine(")
            .iter()
            .any(|line| line.contains(".begin_restart(")),
        "restart_engine no longer opens its generation with begin_restart"
    );
    assert!(
        main.contains(".begin_start("),
        "setup no longer opens the first generation with begin_start"
    );
    assert!(
        !main.contains("engine::spawn("),
        "the app starts an engine outside the slot"
    );
    let engine = production_source("camelid-desktop/src/engine.rs");
    assert!(
        line_then(&engine, "    ) -> StartOutcome {", "self.start_with("),
        "EngineHost::start no longer goes through start_with"
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

/// The index of the first non-blank line after `from`.
fn next_nonblank(lines: &[&str], from: usize) -> Option<usize> {
    lines[from + 1..]
        .iter()
        .position(|line| !line.trim().is_empty())
        .map(|offset| from + 1 + offset)
}

/// The lines of the callback the notice dialog is answered with: from its `.show(` line to
/// the line that closes the call at the same indentation.
fn notice_callback_span(main: &str) -> (usize, usize) {
    let lines: Vec<&str> = main.lines().collect();
    let opened: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.contains(".show(move |"))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        opened.len(),
        1,
        "expected exactly one dialog callback in main.rs, found {opened:?}"
    );
    let start = opened[0];
    let indent = lines[start].len() - lines[start].trim_start().len();
    let end = lines
        .iter()
        .enumerate()
        .skip(start + 1)
        .find(|(_, line)| {
            line.trim_start().starts_with('}') && line.len() - line.trim_start().len() == indent
        })
        .map(|(index, _)| index)
        .expect("the dialog callback is never closed");
    (start, end)
}

/// The source with comments and all whitespace removed, and the line each byte came from, so
/// a needle matches whatever rustfmt did with the line breaks while a comment can still name
/// what the code must not do.
fn stripped_code(source: &str) -> (String, Vec<usize>) {
    let mut code = String::new();
    let mut line_of_byte = Vec::new();
    for (number, line) in source.lines().enumerate() {
        for ch in line
            .split("//")
            .next()
            .unwrap_or_default()
            .chars()
            .filter(|ch| !ch.is_whitespace())
        {
            for _ in 0..ch.len_utf8() {
                line_of_byte.push(number);
            }
            code.push(ch);
        }
    }
    (code, line_of_byte)
}

/// Source-level, and the reason the ordering exists at all. Measured twice on macOS 26 with
/// the bundle built from this branch: a background close hid the window and then raised an
/// unparented dialog, which the OS displayed nowhere (no app window onscreen for 6 s after
/// the close, and none after reopening). The user was told nothing while the preference
/// recorded that they had been. The live receipt is the behavioural closure.
#[test]
fn the_background_notice_is_raised_on_the_visible_window_before_it_hides() {
    let main = production_source("camelid-desktop/src/main.rs");
    let lines: Vec<&str> = main.lines().collect();
    let (callback, callback_end) = notice_callback_span(&main);

    // The order is decided by the pure policy, where it is unit-tested.
    assert!(
        main.contains("lifetime::background_close("),
        "a background close no longer asks the policy which order to use"
    );
    assert!(
        line_then(
            &main,
            "BackgroundClose::NoticeThenHide =>",
            "show_background_notice("
        ),
        "the first background close no longer raises the notice"
    );

    // Without a parent the OS has no window to attach the dialog to.
    let built = lines
        .iter()
        .position(|line| line.contains("lifetime::background_notice("))
        .expect("the notice text is built in main.rs");
    assert!(
        lines[built..callback]
            .iter()
            .any(|line| line.contains(".parent(")),
        "the background notice is raised without a parent window"
    );
    assert!(
        fn_body(&main, "fn show_background_notice(")
            .iter()
            .any(|line| line.contains("notice_raised.swap(true")),
        "two quick closes could stack two notices"
    );

    // One place hides the main window, and both of its callers are past the notice: the arm
    // for a notice already answered, and the callback of one being answered now.
    let hides: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.contains(".hide()") && !line.contains("spotlight"))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        hides.len(),
        1,
        "the main window is hidden from more than one place: {hides:?}"
    );
    assert!(
        fn_body(&main, "fn hide_to_background(")
            .iter()
            .any(|line| line.contains(".hide()")),
        "the main window is hidden outside hide_to_background"
    );

    let hide_now = lines
        .iter()
        .position(|line| line.contains("BackgroundClose::HideNow =>"))
        .expect("the arm for a notice that was already answered");
    let (mut from_arm, mut from_callback) = (0, 0);
    for (index, line) in lines.iter().enumerate() {
        if !line.contains("hide_to_background(") || line.contains("fn hide_to_background(") {
            continue;
        }
        if index == hide_now || next_nonblank(&lines, hide_now) == Some(index) {
            from_arm += 1;
        } else if (callback..=callback_end).contains(&index) {
            from_callback += 1;
        } else {
            panic!(
                "camelid-desktop/src/main.rs:{}: the window is hidden outside the notice's \
                 callback, so a first background close hides it before anything is shown: {line}",
                index + 1
            );
        }
    }
    assert_eq!(
        (from_arm, from_callback),
        (1, 1),
        "the window must hide once for an answered notice and once when one is answered"
    );
}

/// Source-level. `notice_shown` is the claim that this user has been told the engine keeps
/// running with the window closed, and it is written once, for ever. Recorded anywhere but
/// the dialog's callback it claims a notice that may never have been displayed — which is
/// what shipped: the flag was saved immediately after a dialog call that showed nothing.
#[test]
fn notice_shown_is_recorded_only_from_the_notice_callback() {
    let main = production_source("camelid-desktop/src/main.rs");
    let (callback, callback_end) = notice_callback_span(&main);
    let (code, line_of_byte) = stripped_code(&main);
    let mut recorded = 0;
    for needle in [
        "notice_shown:true",
        ".notice_shown.store(true",
        ".notice_shown.swap(true",
        ".notice_shown.fetch_or(true",
    ] {
        for (offset, _) in code.match_indices(needle) {
            let line = line_of_byte[offset];
            assert!(
                (callback..=callback_end).contains(&line),
                "camelid-desktop/src/main.rs:{}: notice_shown is recorded outside the notice's \
                 callback ({needle}), so it would claim a notice nobody answered",
                line + 1
            );
            recorded += 1;
        }
    }
    assert_eq!(
        recorded, 2,
        "the callback must both set notice_shown in memory and save it as true"
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
