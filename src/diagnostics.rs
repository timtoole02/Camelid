//! Durable crash and lifecycle capture for the shipped engine.
//!
//! Today a packaged install that panics leaves **no trace on the machine it
//! happened on**: the console window is the only sink, and closing it (or losing
//! it with the process) destroys the evidence. A user who asks "why did it
//! stop?" has no way to answer that on their own hardware, which is the wrong
//! answer for a local-first product.
//!
//! This module writes a small, bounded, append-only journal of *process
//! lifecycle facts* — nothing else. The contract:
//!
//! * **The engine owns the file.** Nothing extra is written to stderr for the
//!   desktop to scrape: `camelid-desktop` retains a piped stderr that it drains
//!   only on a startup failure, so a chatty stderr would fill the OS pipe buffer
//!   and block the engine — a hang caused by the diagnostics feature. The one
//!   deliberate exception is a single line at startup naming this file's path,
//!   emitted next to the existing ready banner: without it the journal is
//!   undiscoverable, and one bounded line per run cannot fill a pipe buffer.
//! * **Fail open.** Every path degrades to "no journal", never to a failed
//!   startup or a failed generation. No `unwrap`/`expect` on any write path, and
//!   every caller ignores the result.
//! * **Bounded by construction.** One active file with a size cap and a single
//!   retained predecessor, enforced before every append — not a follow-up
//!   commit. Messages and backtraces are clamped so one pathological record
//!   cannot consume the budget.
//! * **No prompt or generated text, ever.** Records carry process lifecycle
//!   facts and panic payloads only. Nothing here reads a request.
//! * **No new dependencies** and **no phone-home**. The output is a file on the
//!   user's disk that they choose to share.
//!
//! ## Reading the journal
//!
//! One JSON object per line. `panic` records are the load-bearing ones: they say
//! what failed, on which thread, and where.
//!
//! A `panic` record does NOT by itself mean the engine died. Panics here are
//! sometimes caught and handled — `src/cuda.rs` probes the GPU inside
//! `catch_unwind`, and a decode job wraps its body the same way so a panic
//! becomes a failed request rather than a dead process. Records carry
//! `"expected": true` when the panic came from the cudarc loader, which is the
//! routine "no CUDA runtime on this machine, fall back to the CPU" case; those
//! are noise. `"expected": false` is the interesting kind.
//!
//! A `session_exit` records a run that ended by returning from `serve` — today
//! that means a startup or serving failure, and it carries the reason.
//!
//! A `session_start` with no matching `session_exit` means the process did not
//! leave through that path. Be precise about what that does and does not say:
//! `camelid serve` installs no signal handler (there is no `ctrl_c` or
//! `with_graceful_shutdown` anywhere in the engine), so an ordinary Ctrl-C ends
//! the process exactly as abruptly as an out-of-memory kill, and the two are
//! recorded identically — which is to say, not at all. Read a missing exit
//! record as "did not fail on the way out", never as evidence of a kill.
//!
//! Telling a deliberate stop apart from a kill needs a graceful-shutdown path in
//! `serve`, and that is deliberately NOT built here: taking ownership of Ctrl-C
//! would put this diagnostics feature on the critical path of stopping the
//! server, and a diagnostics feature must never be able to break the thing it
//! diagnoses.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Once;

/// Rotate once the active journal passes this size. One megabyte is far more
/// than a lifecycle journal needs; the cap exists so a pathological panic loop
/// on a long-running local server cannot fill the disk.
const MAX_LOG_BYTES: u64 = 1024 * 1024;

/// Name of the active journal inside [`log_dir`].
const LOG_FILE_NAME: &str = "camelid.log";

/// Suffix of the single retained predecessor. Exactly one generation is kept:
/// enough to survive a rotation that happens between a crash and the user
/// looking, without turning into an unbounded log directory.
const ROTATED_SUFFIX: &str = ".1";

/// Upper bound on a recorded message. A panic payload is a `format!` result, so
/// a buggy call site can make it arbitrarily large; one record must never be
/// able to consume the whole file budget.
const MAX_MESSAGE_BYTES: usize = 4 * 1024;

/// Upper bound on a recorded backtrace. Larger than a message because frames are
/// the point when someone has opted in via `RUST_BACKTRACE`, but still bounded.
const MAX_BACKTRACE_BYTES: usize = 16 * 1024;

/// Truncate on a UTF-8 boundary and say so, so a clipped value can never be
/// mistaken for a complete one.
fn clamp(mut text: String, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str("… [truncated]");
    text
}

/// Directory holding the journal. Mirrors the convention already used by the
/// gait store (`crate::gait::gait_dir`): `%LOCALAPPDATA%\Camelid\logs` on
/// Windows, an XDG-ish path elsewhere, and the temp directory as a last resort
/// so the module is functional (not merely compilable) everywhere.
pub fn log_dir() -> PathBuf {
    log_dir_from(
        std::env::var_os("LOCALAPPDATA"),
        std::env::var_os("XDG_STATE_HOME"),
        std::env::var_os("HOME"),
        std::env::temp_dir(),
    )
}

/// The resolution order as a pure function, so the fallback chain is testable
/// without mutating process-global environment state (which races across the
/// test harness's threads).
///
/// Empty values are skipped rather than accepted: an exported-but-empty
/// `LOCALAPPDATA` would otherwise resolve to the *relative* path `Camelid/logs`
/// and scatter journals through whatever directory the app happened to start in.
fn log_dir_from(
    localappdata: Option<std::ffi::OsString>,
    xdg_state_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
    fallback: PathBuf,
) -> PathBuf {
    let usable = |value: Option<std::ffi::OsString>| value.filter(|value| !value.is_empty());
    let base = usable(localappdata)
        .map(PathBuf::from)
        .or_else(|| usable(xdg_state_home).map(PathBuf::from))
        .or_else(|| usable(home).map(|home| PathBuf::from(home).join(".local").join("state")))
        .unwrap_or(fallback);
    base.join("Camelid").join("logs")
}

/// Full path of the active journal. Surfaced to the user at startup so the file
/// is discoverable without reading the source.
pub fn log_path() -> PathBuf {
    log_dir().join(LOG_FILE_NAME)
}

/// Test-only redirection of the sink. A panic hook is process-global, so the
/// only way to exercise the *installed* hook is to move the file it writes to;
/// there is no way to hand it a destination. Absent in any non-test build, so
/// the shipped binary has exactly one journal location.
#[cfg(test)]
static TEST_LOG_PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Resolve the file a record is appended to.
fn active_log_path() -> PathBuf {
    #[cfg(test)]
    if let Some(path) = TEST_LOG_PATH.get() {
        return path.clone();
    }
    log_path()
}

/// Install the panic recorder.
///
/// Call this **after** `quiet_cudarc_loader_panics` so this hook is the outer
/// one: every panic is recorded here first, then handed to the previous hook,
/// which still decides what reaches the console. That ordering is the whole
/// point — cudarc loader panics stay suppressed on screen (a CPU-only start
/// must not read like a crash) but are no longer invisible, which matters
/// because a stranger's GPU box is exactly where they fire.
///
/// Idempotent: repeated calls install nothing further.
pub fn install_panic_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            record_panic(info);
            previous(info);
        }));
    });
}

/// Record the start of a serving session.
///
/// Deliberately not called for one-shot CLI invocations: this record exists to
/// give panics and exit failures a run context (version, address, platform), and
/// a journal full of `camelid --version` entries would bury them.
pub fn record_session_start(version: &str, addr: &str) {
    append_record(
        "session_start",
        vec![
            ("version".to_string(), serde_json::json!(version)),
            ("addr".to_string(), serde_json::json!(addr)),
            ("os".to_string(), serde_json::json!(std::env::consts::OS)),
            (
                "arch".to_string(),
                serde_json::json!(std::env::consts::ARCH),
            ),
        ],
    );
}

/// Record a serving session that ended by returning from `serve`. In practice
/// that is a startup or serving failure, carried in `detail`; see the module
/// docs for why an ordinary Ctrl-C never reaches this.
pub fn record_session_exit(outcome: &str, detail: Option<&str>) {
    let mut fields = vec![("outcome".to_string(), serde_json::json!(outcome))];
    if let Some(detail) = detail {
        fields.push((
            "detail".to_string(),
            serde_json::json!(clamp(detail.to_string(), MAX_MESSAGE_BYTES)),
        ));
    }
    append_record("session_exit", fields);
}

/// Serialize one panic into the journal. Never panics itself: a panic inside a
/// panic hook aborts the process, which would replace a recoverable crash
/// report with a bare abort.
fn record_panic(info: &std::panic::PanicHookInfo<'_>) {
    let payload = info
        .payload()
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| info.payload().downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<non-string panic payload>".to_string());

    let location = info
        .location()
        .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()));

    let thread = std::thread::current()
        .name()
        .unwrap_or("<unnamed>")
        .to_string();

    // A panic hook cannot know whether the unwind will be caught, and several
    // panics here are ROUTINE: src/cuda.rs probes the GPU with five
    // `catch_unwind` calls around `CudaContext::new` and PTX compilation, and
    // the shipped Windows build has CUDA compiled in. On a machine with no CUDA
    // runtime those fire on every startup, are handled, and the engine goes on to
    // run happily on the CPU. Unmarked, they would be the MAJORITY of records on
    // exactly the population this journal exists for, burying the real crash.
    //
    // Detected by panic origin, the same test `quiet_cudarc_loader_panics` uses
    // to keep them off the console.
    let from_cudarc = info
        .location()
        .is_some_and(|loc| loc.file().replace('\\', "/").contains("/cudarc-"));

    let mut fields = vec![
        (
            "message".to_string(),
            serde_json::json!(clamp(payload, MAX_MESSAGE_BYTES)),
        ),
        ("thread".to_string(), serde_json::json!(thread)),
        ("expected".to_string(), serde_json::json!(from_cudarc)),
    ];
    if let Some(location) = location {
        fields.push(("location".to_string(), serde_json::json!(location)));
    }
    // `Backtrace::capture` is a no-op unless the user opted in via
    // RUST_BACKTRACE, so this neither costs anything nor writes a wall of
    // frames into the journal by default.
    let backtrace = std::backtrace::Backtrace::capture();
    if backtrace.status() == std::backtrace::BacktraceStatus::Captured {
        fields.push((
            "backtrace".to_string(),
            serde_json::json!(clamp(backtrace.to_string(), MAX_BACKTRACE_BYTES)),
        ));
    }

    append_record("panic", fields);
}

/// Build one record and append it, discarding any failure. This is the single
/// choke point where the fail-open rule is enforced.
fn append_record(event: &str, fields: Vec<(String, serde_json::Value)>) {
    let _ = append_line(&active_log_path(), &build_record(event, fields));
}

/// Serialize one record. Pure, so the wire shape can be asserted directly \u2014
/// including that hostile text never escapes its line, which is what keeps the
/// file parseable as JSONL.
fn build_record(event: &str, fields: Vec<(String, serde_json::Value)>) -> String {
    let mut record = serde_json::Map::new();
    record.insert(
        "ts".to_string(),
        serde_json::json!(crate::receipt::rfc3339_utc_now()),
    );
    record.insert("event".to_string(), serde_json::json!(event));
    record.insert("pid".to_string(), serde_json::json!(std::process::id()));
    for (key, value) in fields {
        record.insert(key, value);
    }
    serde_json::Value::Object(record).to_string()
}

/// Append one line to `path`, rotating first if the file has reached the cap.
///
/// Separated from record construction so the bounded-growth and append
/// behaviour can be tested directly against a temporary directory, with no
/// process-global environment state and therefore no cross-test races.
fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // A failed rotation must not cost the record. Two processes can race for the
    // rename and one of them loses; the loser still has something worth writing,
    // and one oversized file beats a dropped crash report.
    let _ = rotate_if_full(path);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    // ONE write call, record and newline together. Camelid can legitimately run
    // as several processes at once (`serve` plus a CLI invocation), and an
    // append-mode write is atomic per call — but only per call, so writing the
    // newline separately would let two records interleave around it.
    let mut buffer = String::with_capacity(line.len() + 1);
    buffer.push_str(line);
    buffer.push('\n');
    file.write_all(buffer.as_bytes())?;
    // `File` has no userspace buffer, so this is a no-op today and is kept only
    // so the code stays correct if the writer type ever changes. What actually
    // makes a record survive is that `write_all` hands the bytes to the OS
    // before returning: the page cache outlives the process, which is the
    // failure this module exists for. It is NOT protection against power loss —
    // that would need `sync_data`, and an fsync per record is not worth it here.
    file.flush()
}

/// Move the active journal aside once it reaches the cap. A missing file is not
/// an error — that is simply the first write.
fn rotate_if_full(path: &Path) -> std::io::Result<()> {
    let Ok(metadata) = std::fs::metadata(path) else {
        return Ok(());
    };
    if metadata.len() < MAX_LOG_BYTES {
        return Ok(());
    }
    let mut rotated = path.as_os_str().to_os_string();
    rotated.push(ROTATED_SUFFIX);
    let rotated = PathBuf::from(rotated);
    // Windows `rename` fails when the destination exists, unlike POSIX. Drop the
    // previous generation first so rotation behaves identically on both.
    let _ = std::fs::remove_file(&rotated);
    std::fs::rename(path, rotated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appended_records_accumulate_one_per_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join(LOG_FILE_NAME);

        append_line(&path, "{\"event\":\"first\"}").unwrap();
        append_line(&path, "{\"event\":\"second\"}").unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(
            lines,
            vec!["{\"event\":\"first\"}", "{\"event\":\"second\"}"]
        );
    }

    #[test]
    fn the_journal_is_bounded_and_keeps_exactly_one_predecessor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LOG_FILE_NAME);
        let rotated = dir.path().join(format!("{LOG_FILE_NAME}{ROTATED_SUFFIX}"));

        // Fill past the cap, then append: the oversized file must move aside and
        // the active file must restart from the new record.
        std::fs::write(&path, vec![b'x'; MAX_LOG_BYTES as usize]).unwrap();
        append_line(&path, "after-first-rotation").unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "after-first-rotation\n"
        );
        assert_eq!(std::fs::metadata(&rotated).unwrap().len(), MAX_LOG_BYTES);

        // A second rotation replaces the predecessor rather than accumulating a
        // .2, .3, ... series — this is what makes growth bounded.
        std::fs::write(&path, vec![b'y'; MAX_LOG_BYTES as usize]).unwrap();
        append_line(&path, "after-second-rotation").unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "after-second-rotation\n"
        );
        assert!(std::fs::read_to_string(&rotated)
            .unwrap()
            .starts_with("yyy"));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[test]
    fn a_write_failure_is_reported_rather_than_panicking() {
        let dir = tempfile::tempdir().unwrap();
        // A path whose parent is an existing *file* cannot be created; the
        // fail-open callers rely on this returning Err instead of unwinding.
        let blocker = dir.path().join("not-a-directory");
        std::fs::write(&blocker, b"x").unwrap();
        assert!(append_line(&blocker.join(LOG_FILE_NAME), "ignored").is_err());
    }

    #[test]
    fn the_default_journal_lives_under_a_camelid_owned_directory() {
        let path = log_path();
        assert_eq!(path.file_name().unwrap(), LOG_FILE_NAME);
        assert_eq!(path.parent().unwrap().file_name().unwrap(), "logs");
        assert_eq!(
            path.parent()
                .unwrap()
                .parent()
                .unwrap()
                .file_name()
                .unwrap(),
            "Camelid"
        );
    }

    #[test]
    fn the_directory_falls_back_in_order_and_skips_empty_values() {
        let os = |s: &str| Some(std::ffi::OsString::from(s));
        let temp = PathBuf::from("/tmp-fallback");

        assert_eq!(
            log_dir_from(os("/local"), os("/state"), os("/home"), temp.clone()),
            PathBuf::from("/local").join("Camelid").join("logs")
        );
        assert_eq!(
            log_dir_from(None, os("/state"), os("/home"), temp.clone()),
            PathBuf::from("/state").join("Camelid").join("logs")
        );
        assert_eq!(
            log_dir_from(None, None, os("/home"), temp.clone()),
            PathBuf::from("/home")
                .join(".local")
                .join("state")
                .join("Camelid")
                .join("logs")
        );
        assert_eq!(
            log_dir_from(None, None, None, temp.clone()),
            temp.join("Camelid").join("logs")
        );

        // An exported-but-empty variable must not win: joining onto "" yields a
        // RELATIVE path, which would scatter journals through whatever directory
        // the app started in.
        let resolved = log_dir_from(os(""), os(""), os(""), temp.clone());
        assert_eq!(resolved, temp.join("Camelid").join("logs"));
        assert_ne!(resolved, PathBuf::from("Camelid").join("logs"));
    }

    #[test]
    fn oversized_text_is_clamped_on_a_char_boundary_and_says_so() {
        assert_eq!(clamp("short".to_string(), 64), "short");

        // Multi-byte characters straddling the cap must not produce invalid
        // UTF-8 or panic on a split boundary.
        let wide = "\u{00e9}".repeat(100); // 2 bytes each
        let clamped = clamp(wide, 51);
        assert!(clamped.ends_with("\u{2026} [truncated]"));
        assert_eq!(clamped.trim_end_matches("\u{2026} [truncated]").len(), 50);

        let huge = clamp("x".repeat(5 * 1024 * 1024), MAX_MESSAGE_BYTES);
        assert!(huge.len() < MAX_MESSAGE_BYTES + 32);
    }

    #[test]
    fn hostile_text_cannot_escape_its_line() {
        // A panic message is a format! result and can contain anything. If any of
        // it reached the file raw, one record would become several and the file
        // would stop being parseable as JSONL.
        let nasty = "line one\nline two\r\nnull:\0 quote:\" brace:}{ tab:\t";
        let line = build_record(
            "panic",
            vec![("message".to_string(), serde_json::json!(nasty))],
        );
        assert_eq!(line.lines().count(), 1);
        assert!(!line.contains('\n'));
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["message"], nasty);
        assert_eq!(parsed["event"], "panic");
    }

    #[test]
    fn concurrent_writers_never_interleave_a_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LOG_FILE_NAME);
        const WRITERS: usize = 8;
        const PER_WRITER: usize = 40;

        std::thread::scope(|scope| {
            for writer in 0..WRITERS {
                let path = path.clone();
                scope.spawn(move || {
                    for index in 0..PER_WRITER {
                        let line = build_record(
                            "panic",
                            vec![
                                ("writer".to_string(), serde_json::json!(writer)),
                                ("index".to_string(), serde_json::json!(index)),
                                // Padding makes a torn write obvious rather than
                                // something a short record could hide.
                                ("pad".to_string(), serde_json::json!("p".repeat(512))),
                            ],
                        );
                        append_line(&path, &line).unwrap();
                    }
                });
            }
        });

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), WRITERS * PER_WRITER);
        // Every line must be a whole, parseable record: a split write would show
        // up as a parse failure or a blank line, not as a missing line.
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|err| panic!("torn record {line:?}: {err}"));
            assert_eq!(parsed["pad"].as_str().map(str::len), Some(512));
        }
    }

    #[test]
    fn a_failed_rotation_does_not_cost_the_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LOG_FILE_NAME);
        std::fs::write(&path, vec![b'x'; MAX_LOG_BYTES as usize]).unwrap();

        // Make the rotation target un-renameable by holding it open as a
        // directory, the same shape a lost rename race produces: rotation fails,
        // but the record still has to land.
        let rotated = dir.path().join(format!("{LOG_FILE_NAME}{ROTATED_SUFFIX}"));
        std::fs::create_dir(&rotated).unwrap();
        std::fs::write(rotated.join("occupied"), b"x").unwrap();

        assert!(
            rotate_if_full(&path).is_err(),
            "the rotation must fail here"
        );
        append_line(&path, "survived-a-failed-rotation").unwrap();
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .ends_with("survived-a-failed-rotation\n"));
    }

    /// End-to-end proof that the hook is wired: install it, panic for real, and
    /// read the record back off disk. Also pins the chaining property that makes
    /// the ordering in `main` correct — the previously installed hook must still
    /// run, because that is the hook deciding what reaches the console.
    ///
    /// This is the only test that touches the process-global hook and the
    /// process-global sink override, so it owns both and no other test may set
    /// them.
    #[test]
    fn an_installed_hook_records_a_real_panic_and_still_calls_the_previous_hook() {
        // Suffixed with the pid: the sink override is process-global, so a fixed
        // name would collide if two test processes ever ran on one machine.
        let dir = std::env::temp_dir().join(format!(
            "camelid-diagnostics-hook-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(LOG_FILE_NAME);
        let _ = std::fs::remove_file(&path);
        TEST_LOG_PATH.set(path.clone()).unwrap();

        // A predecessor hook that records only that it ran; the default hook is
        // replaced so the deliberate panic does not print a scary trace.
        static PREVIOUS_RAN: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        std::panic::set_hook(Box::new(|_| {
            PREVIOUS_RAN.store(true, std::sync::atomic::Ordering::SeqCst);
        }));
        install_panic_hook();

        let unwound =
            std::panic::catch_unwind(|| panic!("deliberate diagnostics self-test panic")).is_err();
        assert!(unwound, "the test panic must actually unwind");
        assert!(
            PREVIOUS_RAN.load(std::sync::atomic::Ordering::SeqCst),
            "the previously installed hook must still run"
        );

        let contents = std::fs::read_to_string(&path).unwrap();
        // Find OUR record rather than parsing the whole file as one value. The
        // hook is process-global once installed, so any other test that panics
        // concurrently — `#[should_panic]` cases, the `catch_unwind` probes in
        // catalog/cuda/diffusion — also appends here, and assuming a
        // single-line file would make this test flaky under a parallel run.
        let records: Vec<serde_json::Value> = contents
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        let ours: Vec<&serde_json::Value> = records
            .iter()
            .filter(|record| record["message"] == "deliberate diagnostics self-test panic")
            .collect();
        assert_eq!(ours.len(), 1, "expected exactly one record for our panic");
        let record = ours[0];
        assert_eq!(record["event"], "panic");
        assert_eq!(record["pid"], std::process::id());
        assert!(record["ts"].as_str().is_some_and(|ts| ts.ends_with('Z')));
        assert!(record["location"]
            .as_str()
            .is_some_and(|loc| loc.contains("diagnostics.rs")));
        // This panic is ours, not a GPU probe, so it must be the interesting kind.
        // Getting this backwards would mark every real crash as routine noise.
        assert_eq!(record["expected"], false);

        let _ = std::panic::take_hook();
        let _ = std::fs::remove_file(&path);
    }
}
