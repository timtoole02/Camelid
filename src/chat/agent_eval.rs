//! `camelid agent-eval` — the tool-capability promotion harness.
//!
//! Decides whether a model can drive a clean tool-call round-trip, producing a
//! receipt that justifies flipping `tool_capable` true — never a lucky run.
//! Crucially it distinguishes a real capability **FAIL** from an
//! **INCONCLUSIVE** result (the model didn't load within budget on a contended
//! box), so promotion is never decided by noise. Promotion is only ever earned
//! by a PASS receipt. See `DECISIONS.md`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use super::agent::{self, AgentMsg, LiveDriver, Reporter};
use super::client::{Client, LoadOutcome};
use super::server::ServerHandle;
use super::tools::{Sandbox, ToolOutcome};

/// Exit codes (distinct so scripts can branch on the three outcomes).
pub const EXIT_PASS: i32 = 0;
pub const EXIT_FAIL: i32 = 1;
pub const EXIT_INCONCLUSIVE: i32 = 3;

pub struct EvalConfig {
    pub addr: SocketAddr,
    pub model: PathBuf,
    pub load_timeout: u64,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub receipt_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalOutcome {
    Pass,
    Fail,
    Inconclusive,
}

impl EvalOutcome {
    pub fn label(self) -> &'static str {
        match self {
            EvalOutcome::Pass => "PASS",
            EvalOutcome::Fail => "FAIL",
            EvalOutcome::Inconclusive => "INCONCLUSIVE",
        }
    }
    pub fn exit(self) -> i32 {
        match self {
            EvalOutcome::Pass => EXIT_PASS,
            EvalOutcome::Fail => EXIT_FAIL,
            EvalOutcome::Inconclusive => EXIT_INCONCLUSIVE,
        }
    }
}

/// Records the transcript for the verdict + receipt without any styling.
#[derive(Default)]
struct EvalReporter {
    calls: Vec<String>,
    results: Vec<(String, bool, String)>, // (tool, is_ok, text)
    answer: String,
}

impl Reporter for EvalReporter {
    fn model_text(&mut self, text: &str) {
        self.answer = text.to_string();
    }
    fn tool_call(&mut self, line: &str) {
        eprintln!("  ▸ {line}");
        self.calls.push(line.to_string());
    }
    fn tool_result(&mut self, name: &str, outcome: &ToolOutcome) {
        eprintln!(
            "  └ {} {}",
            name,
            if outcome.is_err() { "(error)" } else { "ok" }
        );
        self.results.push((
            name.to_string(),
            !outcome.is_err(),
            outcome.text().to_string(),
        ));
    }
    fn notice(&mut self, text: &str) {
        eprintln!("· {text}");
    }
}

/// Always-allow for the eval (it runs against a controlled fixture in a temp
/// sandbox; promotion evidence shouldn't depend on interactive approval).
struct AutoApprove;
impl agent::Approver for AutoApprove {
    fn approve(&mut self, _a: &super::tools::Action, _s: &Sandbox) -> agent::Decision {
        agent::Decision::Once
    }
}

/// One fixed case in the battery.
struct EvalCase {
    name: &'static str,
    goal: &'static str,
    /// Returns true if the recorded run satisfies the case.
    check: fn(&EvalReporter) -> bool,
}

const FIXTURE: &str = "alpha\nbeta\ngamma\n"; // 3 lines

/// True if a tool `name` ran cleanly (well-formed args → the sandbox executed it)
/// and its output satisfies `out_ok`.
fn tool_ran_ok(r: &EvalReporter, name: &str, out_ok: impl Fn(&str) -> bool) -> bool {
    r.results
        .iter()
        .any(|(n, ok, out)| n == name && *ok && out_ok(out))
}

/// A genuine multi-tool battery exercising three distinct tools (read_file /
/// list_dir / write_file). Each case requires the RIGHT tool to execute cleanly with
/// a correct result AND a correct final answer — tighter than the prior single-case
/// `answer.contains` heuristic. All must pass for a promotion-eligible certificate.
fn battery() -> Vec<EvalCase> {
    let all = vec![
        EvalCase {
            name: "read_and_count",
            goal: "Read the file notes.txt and tell me how many lines it has. Use the read_file \
                   tool, then give the count.",
            check: |r| {
                // read_file read the real fixture AND the answer states the count.
                tool_ran_ok(r, "read_file", |o| {
                    o.contains("alpha") && o.contains("gamma")
                }) && r.answer.contains('3')
            },
        },
        EvalCase {
            name: "list_dir_find",
            goal:
                "List the entries of the current directory '.' with the list_dir tool, then tell \
                   me the name of the text file you find there.",
            check: |r| {
                tool_ran_ok(r, "list_dir", |o| o.contains("notes.txt"))
                    && r.answer.to_lowercase().contains("notes.txt")
            },
        },
        EvalCase {
            name: "write_greeting",
            goal:
                "Create a file named greeting.txt whose exact contents are: hello there\nUse the \
                   write_file tool ONCE, then reply in words that you created it. Do not call any \
                   further tools and do not read the file back.",
            check: |r| {
                let wrote = r
                    .calls
                    .iter()
                    .any(|c| c.contains("write_file") && c.contains("greeting"))
                    && r.results.iter().any(|(n, ok, _)| n == "write_file" && *ok);
                wrote
                    && (r.answer.to_lowercase().contains("created")
                        || r.answer.to_lowercase().contains("greeting")
                        || r.answer.to_lowercase().contains("hello there"))
            },
        },
    ];
    // Runnable-lane prefill is slow for a 9B (a full multi-tool prompt prefills at
    // ~200s/turn on this CPU host, and the macOS resident Metal lane only improves
    // prefill ~1.4x), so a full multi-case battery overruns the agent client's read
    // budget. `CAMELID_EVAL_CASE=<name>` runs exactly one case per
    // invocation, keeping each run within budget; absent the env, the full battery
    // runs (the fast path for optimized-lane models). Each single-case run still
    // mints a full promotion-eligible receipt for that case.
    if let Ok(name) = std::env::var("CAMELID_EVAL_CASE") {
        if all.iter().any(|c| c.name == name) {
            return all.into_iter().filter(|c| c.name == name).collect();
        }
        // Unknown name: fall through to the full battery rather than a 0-case PASS.
    }
    all
}

pub fn run(cfg: EvalConfig) -> anyhow::Result<i32> {
    let client = Client::new(cfg.addr);
    let _server = ServerHandle::ensure(cfg.addr, &client)?;

    // --- bounded load: a contended box yields INCONCLUSIVE, never FAIL ------
    // Absolute WITHOUT resolving symlinks: the serve lane is selected from the
    // path AS NAMED (a catalog-managed Ghost-MoE pair is detected via the
    // catalog filename's `.cghost` sibling), so canonicalizing a catalog
    // symlink silently reroutes the load onto the bare artifact — measured on
    // the 26B pair: the resolved sparse hot shadow loaded WITHOUT its expert
    // pack and every generation decoded routed experts as zeros (fluent-looking
    // garbage), failing the battery for a reason that was never tool
    // capability. The receipt records exactly the path that was loaded.
    let abs = std::path::absolute(&cfg.model).unwrap_or_else(|_| cfg.model.clone());
    eprintln!("loading {} (timeout {}s)…", abs.display(), cfg.load_timeout);
    let started = Instant::now();
    let (tx, rx) = mpsc::channel();
    let loader = client.clone();
    let path = abs.to_string_lossy().to_string();
    std::thread::spawn(move || {
        let _ = tx.send(loader.load_model(&path, None));
    });
    let loaded = match rx.recv_timeout(Duration::from_secs(cfg.load_timeout)) {
        Ok(Ok(LoadOutcome::Loaded { id })) => id,
        Ok(Ok(LoadOutcome::Unsupported { message })) => {
            return finish(
                &cfg,
                EvalOutcome::Fail,
                &abs,
                None,
                &format!("unsupported: {message}"),
                &[],
            );
        }
        Ok(Err(err)) => {
            // A load the HOST refused is not a verdict on the model. The fit
            // preflight declines when the machine is merely busy ("fits this
            // machine, but only N GB free right now") or too small for this
            // artifact — in both cases the model never ran a single step, so a
            // FAIL receipt would stand as evidence that it cannot drive tools
            // when nothing of the kind was measured. That is the same contended-
            // box case the timeout arm below already treats as INCONCLUSIVE, and
            // this function's own contract promises: "never FAIL".
            let (outcome, note) = classify_load_error(&err.to_string());
            return finish(&cfg, outcome, &abs, None, &note, &[]);
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            return finish(
                &cfg,
                EvalOutcome::Inconclusive,
                &abs,
                None,
                &format!(
                    "model did not load within {}s — box likely contended; re-run on a quiet host",
                    cfg.load_timeout
                ),
                &[],
            );
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            return finish(
                &cfg,
                EvalOutcome::Inconclusive,
                &abs,
                None,
                "loader thread died",
                &[],
            );
        }
    };
    eprintln!(
        "loaded '{loaded}' in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    // The eval is intrinsically auto-approve (it runs a controlled fixture with a
    // non-interactive approver). That bypass must never happen in production, so
    // mirror the agent's fail-closed rule: refuse under CAMELID_PRODUCTION.
    if agent::is_production() {
        return finish(
            &cfg,
            EvalOutcome::Fail,
            &abs,
            Some(&loaded),
            "refused: agent-eval uses auto-approval and CAMELID_PRODUCTION is set; \
             run the promotion harness on a non-production host",
            &[],
        );
    }

    // --- fixture workspace --------------------------------------------------
    let work = std::env::temp_dir().join(format!("camelid-agent-eval-{}", std::process::id()));
    std::fs::create_dir_all(&work)?;
    std::fs::write(work.join("notes.txt"), FIXTURE)?;
    // The eval runs a controlled read-only fixture; run_shell is unrestricted so
    // the harness works on any host (including non-Linux CI) without the
    // kernel-sandbox preconditions. The battery never invokes run_shell.
    let sandbox = Sandbox::new(&work, false, Duration::from_secs(20))?
        .with_shell_mode(super::shell_sandbox::ShellSandbox::Unrestricted);
    let tools = super::tools::specs(false, sandbox.shell_mode());
    let family = family_for(&abs);

    // --- run the battery ----------------------------------------------------
    let cancel = AtomicBool::new(false);
    let mut cases = Vec::new();
    let mut all_pass = true;
    for case in battery() {
        eprintln!("== case: {}", case.name);
        let mut driver = LiveDriver::with(
            client.clone(),
            loaded.clone(),
            family.clone(),
            cfg.max_tokens,
            0.0,
        );
        let mut reporter = EvalReporter::default();
        let mut approver = AutoApprove;
        let mut history = vec![
            AgentMsg::System(agent::system_prompt(&sandbox, &tools)),
            AgentMsg::User(case.goal.to_string()),
        ];
        let cfg_loop = agent::AgentConfig {
            workdir: work.clone(),
            max_steps: cfg.max_steps,
            auto_approve: true,
            yolo: false,
            allow_net: false,
            allow_fs: false,
            shell_timeout: Duration::from_secs(20),
            max_tokens: cfg.max_tokens,
            temperature: 0.0,
            // The promotion harness audits nothing (controlled fixture, no sink).
            audit: Box::new(super::audit::NoopSink),
            shell_sandbox: super::shell_sandbox::ShellSandbox::Unrestricted,
            tool_profile: super::tools::ToolProfile::Full,
            allow_plan: true,
            default_write_path: None,
            // No compaction: a promotion receipt must attest a transcript the
            // harness fully determines. Cases are short and cannot approach the
            // budget anyway (D-DROVER-6).
            ctx_budget: None,
            context_paging: false,
        };
        // Auto-approve posture (write/network auto; run_shell still gated, which
        // the AutoApprove approver allows). Production was already refused above.
        let mut policy = agent::Policy::default();
        policy.set_auto_all(true);
        let end = agent::run_loop(
            &mut driver,
            &mut approver,
            &mut reporter,
            &sandbox,
            &cfg_loop,
            &cancel,
            &mut policy,
            &mut history,
        );
        let passed = (case.check)(&reporter);
        all_pass &= passed;
        cases.push(json!({
            "case": case.name,
            "goal": case.goal,
            "loop_end": format!("{end:?}"),
            "tool_calls": reporter.calls,
            "tool_results": reporter.results.iter().map(|(n,ok,t)| json!({"tool":n,"ok":ok,"output":t})).collect::<Vec<_>>(),
            "final_answer": reporter.answer,
            "passed": passed,
        }));
    }
    let _ = std::fs::remove_dir_all(&work);

    let outcome = if all_pass {
        EvalOutcome::Pass
    } else {
        EvalOutcome::Fail
    };
    finish(
        &cfg,
        outcome,
        &abs,
        Some(&loaded),
        "battery complete",
        &cases,
    )
}

/// Host 1-minute load average for the eval receipt. POSIX-only (`getloadavg`);
/// Windows has no equivalent, so the receipt records `null` (unavailable)
/// rather than a misleading number.
#[cfg(unix)]
fn host_loadavg_1m() -> Option<f64> {
    let mut load = [0f64; 3];
    // SAFETY: getloadavg writes up to 3 doubles into the provided buffer and
    // returns the number of samples written.
    let n = unsafe { libc::getloadavg(load.as_mut_ptr(), 3) };
    if n >= 1 {
        Some(load[0])
    } else {
        None
    }
}

#[cfg(not(unix))]
fn host_loadavg_1m() -> Option<f64> {
    None
}

/// Emit the receipt + the human verdict, return the exit code.
/// Split a failed load into "the host said no" (never a verdict on the model)
/// and a real load failure. The fit preflight's refusals all name the
/// `CAMELID_SKIP_FIT_CHECK` override, which is the stable marker for "this was a
/// host decision", not prose that happens to mention memory.
fn classify_load_error(err: &str) -> (EvalOutcome, String) {
    if err.contains("CAMELID_SKIP_FIT_CHECK") {
        return (
            EvalOutcome::Inconclusive,
            format!("host refused the load, so tool capability was never exercised: {err}"),
        );
    }
    (EvalOutcome::Fail, format!("load error: {err}"))
}

fn finish(
    cfg: &EvalConfig,
    outcome: EvalOutcome,
    gguf: &std::path::Path,
    model_id: Option<&str>,
    note: &str,
    cases: &[Value],
) -> anyhow::Result<i32> {
    let loadavg_1m = host_loadavg_1m();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut receipt = json!({
        "schema": "camelid.agent_eval/v1",
        "outcome": outcome.label(),
        "model_id": model_id,
        "gguf": redacted_path_display(gguf),
        "gguf_bytes": std::fs::metadata(gguf).map(|m| m.len()).ok(),
        "quantization": infer_quant(gguf),
        "note": note,
        "cases": cases,
        "host_loadavg_1m": loadavg_1m,
        "timestamp_unix": ts,
        "promotion_eligible": outcome == EvalOutcome::Pass,
    });
    // Seal the receipt with the shared tamper-evident digest so `camelid
    // verify-receipt` can prove it intact. `receipt_id_over` hashes the canonical
    // body with any `receipt_id` field removed, so inserting it afterwards yields
    // the same digest the verifier recomputes.
    let receipt_id = crate::receipt::receipt_id_over(&receipt);
    receipt
        .as_object_mut()
        .expect("a json! object is always an object")
        .insert("receipt_id".to_string(), Value::from(receipt_id.clone()));
    std::fs::create_dir_all(&cfg.receipt_dir)?;
    let stem = gguf
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());
    let path = cfg
        .receipt_dir
        .join(format!("{stem}-{ts}-{}.json", outcome.label()));
    let mut text = serde_json::to_string_pretty(&receipt)?;
    text.push('\n');
    std::fs::write(&path, text)?;

    eprintln!();
    eprintln!("{} — {note}", outcome.label());
    eprintln!("receipt → {} ({receipt_id})", path.display());
    if outcome == EvalOutcome::Inconclusive {
        eprintln!("(inconclusive does NOT change any tool_capable flag — re-run on a quiet box)");
    }
    if outcome == EvalOutcome::Pass {
        eprintln!("(eligible for promotion: set this row's tool_capable=true in the ledger)");
    }
    // Machine-readable verdict on stdout.
    println!("{}", outcome.label());
    Ok(outcome.exit())
}

/// Receipt-safe path spelling: receipts are committed to the public repo, and
/// the public-scrub CI gate rejects absolute home paths, so the home-directory
/// prefix is recorded as `<home>` (the spelling the existing receipt corpus
/// already uses). Non-home paths are recorded as-is.
fn redacted_path_display(path: &std::path::Path) -> String {
    let display = path.display().to_string();
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| home.to_string_lossy().to_string());
    if let Some(home) = home.filter(|home| !home.is_empty()) {
        if let Some(rest) = display.strip_prefix(&home) {
            return format!("<home>{rest}");
        }
    }
    display
}

fn family_for(gguf: &std::path::Path) -> String {
    let name = gguf
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();
    // Ornith / qwen35 emit the custom `<function=…>` XML tool format, which routes
    // to `parse_ornith`. Check before "qwen" (the model is Qwen3.5-derived but its
    // tool grammar differs from Qwen2/Qwen3 JSON-in-`<tool_call>`).
    if name.contains("ornith") || name.contains("qwen35") || name.contains("qwen3.5") {
        "ornith".into()
    } else if name.contains("qwen") {
        "qwen".into()
    } else if name.contains("mistral") {
        "mistral".into()
    } else if name.contains("gemma-4") || name.contains("gemma4") {
        // Routes the content-text fallback to `parse_gemma4` (the primary path
        // stays the server's structured tool_calls).
        "gemma4".into()
    } else {
        "llama".into()
    }
}

fn infer_quant(gguf: &std::path::Path) -> String {
    let name = gguf
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_uppercase();
    for q in [
        "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "BF16", "F16", "F32",
    ] {
        if name.contains(q) {
            return q.to_string();
        }
    }
    "unknown".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Receipts land in the public repo: the home prefix must record as
    /// `<home>` (the corpus spelling the public-scrub gate accepts), while
    /// non-home paths stay verbatim.
    #[test]
    fn receipt_path_redacts_the_home_prefix() {
        let _guard = crate::test_support::env_lock();
        let prior = std::env::var_os("HOME");
        std::env::set_var("HOME", "/home/someone");
        let home_path = std::path::Path::new("/home/someone/models/m.gguf");
        let outside = std::path::Path::new("/srv/models/m.gguf");
        let redacted = redacted_path_display(home_path);
        let untouched = redacted_path_display(outside);
        match prior {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
        assert_eq!(redacted, "<home>/models/m.gguf");
        assert_eq!(untouched, "/srv/models/m.gguf");
    }

    #[test]
    fn outcome_exit_codes_are_distinct() {
        assert_eq!(EvalOutcome::Pass.exit(), 0);
        assert_eq!(EvalOutcome::Fail.exit(), 1);
        assert_eq!(EvalOutcome::Inconclusive.exit(), 3);
    }

    #[test]
    fn a_host_refused_load_is_inconclusive_not_a_model_verdict() {
        // Measured on this Mac: an 8B artifact was declined because only ~7.9 GB
        // of ~17.2 GB was free. The model never ran a step, yet the receipt said
        // FAIL — which reads as evidence that it cannot drive tools, and could be
        // cited later to keep a capable row un-promoted.
        let busy = "This model (~8.7 GB) fits this machine, but only ~7.9 GB of ~17.2 GB memory \
                    is free right now. Close some applications and retry, or set \
                    CAMELID_SKIP_FIT_CHECK=1 to attempt the load anyway.";
        let (outcome, note) = classify_load_error(busy);
        assert_eq!(outcome, EvalOutcome::Inconclusive);
        assert!(note.contains("never exercised"), "{note}");

        let too_big = "This model (~40.0 GB) is larger than this machine can hold in memory. \
                       Set CAMELID_SKIP_FIT_CHECK=1 to attempt the load anyway.";
        assert_eq!(classify_load_error(too_big).0, EvalOutcome::Inconclusive);

        // A genuine load failure is still a FAIL: the engine tried and could not.
        let broken = "tensor blk.0.attn_k.weight has unknown or removed GGML type Unknown(16)";
        assert_eq!(classify_load_error(broken).0, EvalOutcome::Fail);
    }

    #[test]
    fn check_requires_clean_read_and_correct_count() {
        // No tool ran, just an answer → fail.
        let r = EvalReporter {
            answer: "the file has 3 lines".into(),
            ..Default::default()
        };
        assert!(!(battery()[0].check)(&r));
        // read_file ran ok + correct count → pass.
        let r = EvalReporter {
            answer: "it has 3 lines".into(),
            results: vec![("read_file".into(), true, "alpha\nbeta\ngamma\n".into())],
            ..Default::default()
        };
        assert!((battery()[0].check)(&r));
        // read_file errored (malformed args) → fail even with a lucky answer.
        let r2 = EvalReporter {
            answer: "3".into(),
            results: vec![("read_file".into(), false, "requires path".into())],
            ..Default::default()
        };
        assert!(!(battery()[0].check)(&r2));
    }
}
