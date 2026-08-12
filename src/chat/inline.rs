//! Inline (line-mode) front end for `camelid chat`. Scrollback-friendly, works
//! over pipes/SSH/non-TTY, and is the lane the smoke scripts drive. Shares the
//! [`Session`](super::session::Session) core with the full-screen TUI.

use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use super::banner;
use super::client::StreamEnd;
use super::models::{Availability, PickerRow};
use super::session::{Session, CANCEL};
use super::VERSION;

enum Gen {
    Done,
    Cancelled,
    Failed,
}

pub fn run(session: &mut Session, addr: SocketAddr, spawned: bool) -> anyhow::Result<()> {
    println!(
        "{}\n",
        banner::splash(VERSION, &addr.to_string(), &model_line(session))
    );
    println!(
        "{}",
        banner::dim(&format!(
            "server {} at {addr} · type /help for commands",
            if spawned { "spawned" } else { "attached" }
        ))
    );

    let mut rl = DefaultEditor::new()?;
    if !session.has_model() {
        run_picker(session, &mut rl)?;
    }

    loop {
        match rl.readline(&prompt_line(session)) {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(trimmed);
                if let Some(command) = trimmed.strip_prefix('/') {
                    if dispatch(session, &mut rl, command)? {
                        break;
                    }
                } else {
                    user_turn(session, line);
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("{}", banner::dim("(press Ctrl-D or type /exit to quit)"));
            }
            Err(ReadlineError::Eof) => break,
            Err(err) => {
                eprintln!("input error: {err}");
                break;
            }
        }
    }
    Ok(())
}

fn model_line(session: &Session) -> String {
    if session.has_model() {
        format!("{} · {}", session.active_label, session.active_posture)
    } else {
        "no model loaded — use /models".to_string()
    }
}

fn prompt_line(session: &Session) -> String {
    if session.has_model() {
        format!(
            "camelid ({} · {}) › ",
            session.active_label, session.active_posture
        )
    } else {
        "camelid (no model) › ".to_string()
    }
}

/// Returns true when the session should exit.
fn dispatch(session: &mut Session, rl: &mut DefaultEditor, command: &str) -> anyhow::Result<bool> {
    let mut parts = command.splitn(2, char::is_whitespace);
    let raw = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim();
    let name = super::palette::resolve(raw).map(|c| c.name).unwrap_or(raw);
    match name {
        "models" => run_picker(session, rl)?,
        "switch" => switch_loaded(session, rl)?,
        "copy" => copy_last(session),
        "theme" => println!(
            "{}",
            banner::dim("themes are a full-screen TUI feature (run without --plain)")
        ),
        "stop" => println!("{}", banner::dim("nothing to stop in line mode")),
        "agent" => println!(
            "{}",
            banner::dim(
                "agent mode runs as its own sandboxed loop — relaunch with: \
                 camelid chat --agent --model <gguf>  (requires a tool-capable supported row)"
            )
        ),
        "model" => {
            if arg.is_empty() {
                println!("usage: /model <id>   (see /models)");
            } else {
                select_by_id(session, arg);
            }
        }
        "set" => {
            let mut kv = arg.splitn(2, char::is_whitespace);
            match (kv.next(), kv.next()) {
                (Some(k), Some(v)) if !k.is_empty() && !v.trim().is_empty() => {
                    match session.set_param(k, v.trim()) {
                        Ok(msg) => println!("{}", banner::dim(&format!("✓ {msg}"))),
                        Err(err) => println!("{}", banner::dim(&format!("✗ {err}"))),
                    }
                }
                _ => println!(
                    "usage: /set <temperature|top_p|top_k|max_tokens|seed|stream> <value>\n  now: {}",
                    session.settings.summary()
                ),
            }
        }
        "system" => {
            if arg.is_empty() {
                match &session.system {
                    Some(text) => println!("{}", banner::dim(&format!("system: {text}"))),
                    None => println!("{}", banner::dim("no system prompt set")),
                }
            } else {
                session.system = Some(arg.to_string());
                println!(
                    "{}",
                    banner::dim("system prompt set (takes effect next turn)")
                );
            }
        }
        "reset" | "clear" => {
            session.reset_history();
            println!("{}", banner::dim("conversation history cleared"));
        }
        "tokens" => show_tokens(session),
        "info" => show_info(session),
        "save" => {
            let path = PathBuf::from(if arg.is_empty() {
                "camelid-session.json"
            } else {
                arg
            });
            match session.save(&path) {
                Ok(()) => println!("{}", banner::dim(&format!("saved → {}", path.display()))),
                Err(err) => eprintln!("save failed: {err}"),
            }
        }
        "load" => {
            if arg.is_empty() {
                println!("usage: /load <path.json>");
            } else {
                match session.load(&PathBuf::from(arg)) {
                    Ok(()) => println!(
                        "{}",
                        banner::dim(&format!(
                            "loaded {} turn(s) from {arg}",
                            session.history.len()
                        ))
                    ),
                    Err(err) => eprintln!("load failed: {err}"),
                }
            }
        }
        "retry" | "regenerate" => retry(session),
        "pull" => {
            if arg.is_empty() {
                println!("usage: /pull <alias>   (e.g. /pull tinyllama)");
            } else if let Err(err) = crate::catalog::run_pull(Some(arg), session.models_dir()) {
                eprintln!("pull failed: {err}");
            } else {
                println!("{}", banner::dim("downloaded — use /models to load it"));
            }
        }
        "help" => print_help(),
        "exit" | "quit" => return Ok(true),
        other => println!("unknown command /{other} — try /help"),
    }
    Ok(false)
}

fn user_turn(session: &mut Session, input: String) {
    if !session.has_model() {
        println!("no model loaded — use /models to pick one");
        return;
    }
    session.push_user(input);
    if !matches!(generate(session), Gen::Done) {
        session.pop_last();
    }
}

fn retry(session: &mut Session) {
    if !session.has_model() {
        println!("no model loaded — use /models to pick one");
        return;
    }
    if session.last_user_message().is_none() {
        println!("nothing to retry yet");
        return;
    }
    // Drop the previous assistant reply (if any) and regenerate from the same
    // user turn. The user turn is left intact on cancel/failure.
    if matches!(
        session.history.last().map(|t| t.role),
        Some(super::session::Role::Assistant)
    ) {
        session.pop_last();
    }
    let _ = generate(session);
}

/// Stream (or block for) one assistant turn for the current history. Appends the
/// assistant turn only on success; the caller manages the user turn.
fn generate(session: &mut Session) -> Gen {
    let stream = session.settings.stream;
    let request = session.build_request(stream);
    print!("{}", banner::turn_prefix());
    let _ = std::io::stdout().flush();
    let started = Instant::now();

    if !stream {
        return match session.client().chat_blocking(&request) {
            Ok((text, prompt_tokens, completion_tokens)) => {
                println!("{text}");
                session.last_prompt_tokens = prompt_tokens;
                session.last_completion_tokens = completion_tokens;
                session.push_assistant(text);
                print_stats(session, completion_tokens, started);
                Gen::Done
            }
            Err(err) => {
                println!();
                eprintln!("generation failed: {err}");
                Gen::Failed
            }
        };
    }

    CANCEL.store(false, Ordering::SeqCst);
    let mut assistant = String::new();
    let client = session.client();
    let result = client.chat_stream(&request, &CANCEL, |delta| {
        print!("{delta}");
        let _ = std::io::stdout().flush();
        assistant.push_str(delta);
    });
    println!();
    match result {
        Ok((StreamEnd::Length, deltas)) => {
            session.last_prompt_tokens = None;
            session.last_completion_tokens = Some(deltas);
            session.push_assistant(assistant);
            println!(
                "{}",
                banner::dim("[cut off at max_tokens — raise it to finish this answer]")
            );
            print_stats(session, Some(deltas), started);
            Gen::Done
        }
        Ok((StreamEnd::Done, deltas)) => {
            session.last_prompt_tokens = None;
            session.last_completion_tokens = Some(deltas);
            session.push_assistant(assistant);
            print_stats(session, Some(deltas), started);
            Gen::Done
        }
        Ok((StreamEnd::Cancelled, _)) => {
            println!(
                "{}",
                banner::dim("^C — interrupted; this turn was discarded")
            );
            Gen::Cancelled
        }
        Err(err) => {
            eprintln!("generation failed: {err}");
            Gen::Failed
        }
    }
}

fn print_stats(_session: &Session, completion_tokens: Option<u32>, started: Instant) {
    let secs = started.elapsed().as_secs_f64();
    let stat = match completion_tokens {
        Some(n) if secs > 0.0 => format!("└ {n} tok · {:.0} tok/s · {secs:.1}s", n as f64 / secs),
        Some(n) => format!("└ {n} tok · {secs:.1}s"),
        None => format!("└ {secs:.1}s"),
    };
    println!("{}", banner::dim(&stat));
}

fn show_tokens(session: &Session) {
    let fmt = |value: Option<u32>| value.map(|n| n.to_string()).unwrap_or_else(|| "n/a".into());
    println!(
        "{}",
        banner::dim(&format!(
            "last turn — prompt: {}  completion: {}",
            fmt(session.last_prompt_tokens),
            fmt(session.last_completion_tokens),
        ))
    );
}

fn show_info(session: &Session) {
    for (key, value) in session.model_info() {
        println!("{}", banner::dim(&format!("  {key:<15} {value}")));
    }
}

fn run_picker(session: &mut Session, rl: &mut DefaultEditor) -> anyhow::Result<()> {
    let rows = session.supported_rows();
    if rows.is_empty() {
        println!(
            "{}",
            banner::dim("the server advertises no supported models")
        );
        return Ok(());
    }
    println!("Supported models:");
    for (index, row) in rows.iter().enumerate() {
        println!(
            "  {:>2}. {:<34} {:<7} {}",
            index + 1,
            row.id,
            row.quant,
            availability_tag(row.availability),
        );
    }
    let choice = rl.readline("select # (blank to cancel): ")?;
    let choice = choice.trim();
    if choice.is_empty() {
        return Ok(());
    }
    let Some(row) = choice
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_sub(1))
        .and_then(|i| rows.get(i))
    else {
        println!("not a valid selection: {choice:?}");
        return Ok(());
    };
    select_row(session, rl, row)
}

fn switch_loaded(session: &mut Session, rl: &mut DefaultEditor) -> anyhow::Result<()> {
    let loaded = session.loaded_models();
    if loaded.is_empty() {
        println!("{}", banner::dim("no models are loaded yet — use /models"));
        return Ok(());
    }
    println!("Loaded models (instant switch):");
    let active = session.active_id.clone();
    for (i, info) in loaded.iter().enumerate() {
        let dot = if active.as_deref() == Some(info.id.as_str()) {
            "●"
        } else {
            "○"
        };
        println!(
            "  {:>2}. {dot} {:<30} {}",
            i + 1,
            info.id,
            info.descriptor()
        );
    }
    let choice = rl.readline("switch to # (blank to cancel): ")?;
    let choice = choice.trim();
    if choice.is_empty() {
        return Ok(());
    }
    if let Some(info) = choice
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_sub(1))
        .and_then(|i| loaded.get(i))
    {
        session.switch_to_loaded(info);
        println!(
            "{}",
            banner::dim(&format!("switched to {} (history reset)", info.id))
        );
    } else {
        println!("not a valid selection: {choice:?}");
    }
    Ok(())
}

fn copy_last(session: &Session) {
    match session
        .history
        .iter()
        .rev()
        .find(|t| t.role == super::session::Role::Assistant)
    {
        Some(turn) if super::clipboard::copy(&turn.content) => {
            println!(
                "{}",
                banner::dim(&format!("copied {} chars", turn.content.len()))
            )
        }
        Some(_) => println!(
            "{}",
            banner::dim("clipboard copy not supported by this terminal")
        ),
        None => println!("{}", banner::dim("no reply to copy yet")),
    }
}

fn select_by_id(session: &mut Session, id: &str) {
    // Prefer an already-loaded model (instant switch).
    if let Some(info) = session.loaded_models().into_iter().find(|m| m.id == id) {
        session.switch_to_loaded(&info);
        println!(
            "{}",
            banner::dim(&format!("switched to {id} (history reset)"))
        );
        return;
    }
    let rows = session.supported_rows();
    let Some(row) = rows.iter().find(|row| row.id == id) else {
        println!("'{id}' is not loaded or a supported model id — see /models");
        return;
    };
    match row.availability {
        Availability::Ready => load_row(session, row),
        Availability::NotDownloaded => {
            println!("'{id}' is supported but not downloaded — use /models to fetch it")
        }
        Availability::NoPullAlias => {
            println!("'{id}' has no pull alias — provide its GGUF via --model")
        }
    }
}

fn select_row(
    session: &mut Session,
    rl: &mut DefaultEditor,
    row: &PickerRow,
) -> anyhow::Result<()> {
    match row.availability {
        Availability::Ready => load_row(session, row),
        Availability::NoPullAlias => println!(
            "{}",
            banner::dim(&format!(
                "'{}' is supported but has no pull alias — provide its GGUF via --model",
                row.id
            ))
        ),
        Availability::NotDownloaded => {
            let Some(item) = row.catalog.as_ref() else {
                println!("'{}' has no pull catalog entry", row.id);
                return Ok(());
            };
            let answer = rl.readline(&format!(
                "download {} ({:.1} GB)? [Y/n]: ",
                item.name,
                item.size_bytes as f64 / 1e9
            ))?;
            if matches!(answer.trim().to_ascii_lowercase().as_str(), "n" | "no") {
                return Ok(());
            }
            crate::catalog::run_pull(Some(item.catalog_id), session.models_dir())?;
            load_row(session, row);
        }
    }
    Ok(())
}

fn load_row(session: &mut Session, row: &PickerRow) {
    let Some(path) = row.local_path(session.models_dir()) else {
        println!("'{}' has no local GGUF path", row.id);
        return;
    };
    match session.load_model_file(&path, Some(&row.id), Some("supported")) {
        Ok(super::session::LoadResult::Loaded) => {
            println!(
                "{}",
                banner::dim(&format!("active model: {} (history reset)", row.id))
            )
        }
        Ok(super::session::LoadResult::Unsupported(message)) => eprintln!("{message}"),
        Err(err) => eprintln!("load failed: {err}"),
    }
}

fn availability_tag(availability: Availability) -> &'static str {
    match availability {
        Availability::Ready => "[ready]",
        Availability::NotDownloaded => "[supported · not downloaded]",
        Availability::NoPullAlias => "[supported · no pull alias]",
    }
}

fn print_help() {
    println!("commands:");
    for (cmd, help) in [
        ("/models", "browse loaded + downloadable models"),
        ("/switch", "instantly switch between already-loaded models"),
        (
            "/model <id>",
            "switch to a model by id (loaded or supported)",
        ),
        ("/copy", "copy the last reply to the clipboard"),
        (
            "/set <k> <v>",
            "sampling: temperature, top_p, top_k, max_tokens, seed, stream",
        ),
        ("/system <text>", "set the system prompt (next turn)"),
        ("/reset", "clear conversation history, keep the model"),
        ("/retry", "regenerate the last reply"),
        ("/tokens", "last response's prompt/completion token counts"),
        ("/info", "active model + settings detail"),
        (
            "/save [path]",
            "save the session (settings + transcript) to JSON",
        ),
        ("/load <path>", "load a saved session"),
        ("/pull <alias>", "download a supported model"),
        ("/help", "this list"),
        ("/exit", "quit (also Ctrl-D)"),
    ] {
        println!("  {cmd:<16} {help}");
    }
}
