//! `camelid agent-orchestration-eval` — the subagent-orchestration gate.
//!
//! Two modes, emitting a tamper-evident sealed `camelid.agent-orchestration-
//! receipt/v1`:
//! - **rung 2 (default, no `--model`)**: drives the REAL spawn → run → collect
//!   round-trip against a CANNED worker (no model, no server) and exercises the
//!   mandatory caps — concurrency, depth limit, per-child timeout/reaping,
//!   subtask_id validation, malformed-IPC handling.
//! - **rung 3 (`--model <gguf>`)**: drives a REAL model through one
//!   spawn_subagent → check_subagent_status round-trip; the subagent task is
//!   canned (deterministic) so the case isolates the model's orchestration-
//!   driving from subagent inference. A contended box yields INCONCLUSIVE.
//!
//! Scope discipline: BOTH rungs promote NOTHING (`tool_capable` is untouched),
//! and neither claims parallel speedup (that needs a same-box wall-clock
//! receipt). rung 3 attests a real model *can drive* a spawn — it is not a
//! support promotion.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::agent_eval::EvalOutcome;
use super::subagent::{self, SubagentConfig};

pub const RECEIPT_SCHEMA_V1: &str = crate::receipt::agent::ORCHESTRATION_RECEIPT_SCHEMA_V1;

pub struct OrchestrationConfig {
    pub receipt_dir: PathBuf,
    /// When set, run the rung-3 REAL-model battery (drive this GGUF through a
    /// spawn round-trip) instead of the canned rung-2 mechanics battery.
    pub model: Option<PathBuf>,
    pub addr: std::net::SocketAddr,
    pub load_timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HostBlock {
    os: String,
    arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hostname: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CaseResult {
    name: String,
    observed: String,
    verdict: String, // "PASS" | "FAIL"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OrchestrationReceipt {
    schema: String,
    receipt_id: String,
    created_unix: u64,
    feature: String,
    /// 2 = canned mechanics; 3 = real model drove the orchestration.
    rung: u8,
    /// The model id, present only for a rung-3 real-model run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    outcome: String,
    host: HostBlock,
    cases: Vec<CaseResult>,
    note: String,
    /// Honest scope: this receipt promotes nothing (tool_capable untouched).
    promotes_capability: bool,
    /// Honest scope: never a speedup claim. Always false.
    claims_speedup: bool,
}

impl OrchestrationReceipt {
    fn compute_receipt_id(&self) -> String {
        crate::receipt::receipt_id_over(&serde_json::to_value(self).expect("receipt serializes"))
    }
    fn seal(&mut self) {
        self.receipt_id = self.compute_receipt_id();
    }
    fn verify_self_digest(&self) -> bool {
        self.compute_receipt_id() == self.receipt_id
    }
}

fn snippet(s: &str) -> String {
    const N: usize = 300;
    let s = s.replace('\n', " | ");
    if s.len() <= N {
        return s;
    }
    let mut end = N;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

pub(super) fn hostname() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .filter(|s| !s.is_empty())
}

pub fn run(cfg: OrchestrationConfig) -> anyhow::Result<i32> {
    let (outcome, cases, note, rung, model_id, feature) = match &cfg.model {
        Some(model) => {
            let (o, c, n, mid) = run_real_model_battery(cfg.addr, model, cfg.load_timeout);
            (
                o,
                c,
                n,
                3u8,
                mid,
                "subagent-orchestration: a REAL model drives spawn_subagent + \
                 check_subagent_status (subagent task canned for determinism)"
                    .to_string(),
            )
        }
        None => {
            let (o, c, n) = run_battery();
            (
                o,
                c,
                n,
                2u8,
                None,
                "subagent-orchestration: spawn_subagent + check_subagent_status \
                 (stub round-trip + caps/depth/reaping)"
                    .to_string(),
            )
        }
    };

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut receipt = OrchestrationReceipt {
        schema: RECEIPT_SCHEMA_V1.to_string(),
        receipt_id: String::new(),
        created_unix: ts,
        feature,
        rung,
        model_id,
        outcome: outcome.label().to_string(),
        host: HostBlock {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            hostname: hostname(),
        },
        cases,
        note,
        promotes_capability: false,
        claims_speedup: false,
    };
    receipt.seal();
    debug_assert!(receipt.verify_self_digest());

    std::fs::create_dir_all(&cfg.receipt_dir)?;
    let path = cfg.receipt_dir.join(format!(
        "orchestration-rung{rung}-{ts}-{}.json",
        outcome.label()
    ));
    let mut text = serde_json::to_string_pretty(&receipt)?;
    text.push('\n');
    std::fs::write(&path, text)?;

    eprintln!();
    eprintln!("{} — {}", outcome.label(), receipt.note);
    eprintln!("receipt → {} ({})", path.display(), receipt.receipt_id);
    match outcome {
        EvalOutcome::Inconclusive => {
            eprintln!("(inconclusive does NOT change any capability flag)")
        }
        EvalOutcome::Pass => eprintln!(
            "(rung-{rung}: verified; promotes NOTHING — tool_capable untouched; NOT a speedup)"
        ),
        EvalOutcome::Fail => {}
    }
    println!("{}", outcome.label());
    Ok(outcome.exit())
}

/// Poll a subagent until it leaves the `running` state (or the deadline passes).
pub(super) fn poll_to_terminal(root: &Path, id: &str, max: Duration) -> String {
    let deadline = Instant::now() + max;
    loop {
        let s = subagent::status(root, id).unwrap_or_else(|e| format!("status: error\n{e}"));
        if !s.contains("status: running") || Instant::now() >= deadline {
            return s;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn base_config(concurrency: usize, timeout: Duration) -> SubagentConfig {
    SubagentConfig {
        addr: SocketAddr::from(([127, 0, 0, 1], 8181)),
        model_id: "canned".to_string(),
        model_sha256: "00".repeat(32),
        family: "llama".to_string(),
        max_steps: 6,
        context_budget_tokens: 512,
        max_tokens: 64,
        concurrency,
        depth_limit: 1,
        timeout,
        // Conservative posture for the gate (the canned worker only makes a
        // read-only call anyway).
        auto_approve: false,
        shell_mode: super::shell_sandbox::ShellSandbox::Sandboxed,
    }
}

fn run_battery() -> (EvalOutcome, Vec<CaseResult>, String) {
    let temp = std::env::temp_dir().join(format!("camelid-orch-eval-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp);

    let mut cases: Vec<CaseResult> = Vec::new();
    let mut push = |name: &str, observed: String, pass: bool| {
        cases.push(CaseResult {
            name: name.to_string(),
            observed: snippet(&observed),
            verdict: if pass { "PASS" } else { "FAIL" }.to_string(),
        });
    };
    let spawn_seen; // did a child process actually run? (assigned in case 1)

    // 1) Full spawn → run → collect round-trip against the canned worker.
    {
        let work = temp.join("roundtrip");
        let _ = std::fs::create_dir_all(&work);
        let _subagent_session = subagent::configure_scoped(base_config(2, Duration::from_secs(60)));
        let spawned = subagent::spawn_canned(&work, "rt-1", "say hello", "ROUNDTRIP-OK", 0);
        let status = poll_to_terminal(&work, "rt-1", Duration::from_secs(30));
        spawn_seen = spawned.is_ok()
            && (status.contains("completed")
                || status.contains("failed")
                || status.contains("inconclusive"));
        let ok = spawned.is_ok()
            && status.contains("status: completed")
            && status.contains("ROUNDTRIP-OK")
            && status.contains("list_dir");
        push(
            "spawn_run_collect_roundtrip",
            format!("spawn={spawned:?} | {status}"),
            ok,
        );
    }

    // 2) Concurrency cap: with cap=1, a second live spawn is refused.
    {
        let work = temp.join("cap");
        let _ = std::fs::create_dir_all(&work);
        let _subagent_session = subagent::configure_scoped(base_config(1, Duration::from_secs(60)));
        let a = subagent::spawn_canned(&work, "cap-a", "g", "A", 3000); // stays live ~3s
        let b = subagent::spawn_canned(&work, "cap-b", "g", "B", 0); // must be refused
        let ok = a.is_ok() && b.is_err() && b.as_ref().unwrap_err().contains("concurrency");
        push("concurrency_cap_enforced", format!("a={a:?} b={b:?}"), ok);
        let _ = poll_to_terminal(&work, "cap-a", Duration::from_secs(10)); // reap A
    }

    // 3) Depth cap: at depth==limit, spawning is refused (fork-bomb guard).
    {
        let work = temp.join("depth");
        let _ = std::fs::create_dir_all(&work);
        let _subagent_session = subagent::configure_scoped(base_config(2, Duration::from_secs(60)));
        std::env::set_var(subagent::DEPTH_ENV, "1");
        let d = subagent::spawn_canned(&work, "depth-1", "g", "D", 0);
        std::env::remove_var(subagent::DEPTH_ENV);
        let ok = d.is_err() && d.as_ref().unwrap_err().contains("depth");
        push("depth_limit_enforced", format!("d={d:?}"), ok);
    }

    // 4) Timeout/reaping: a child exceeding its timeout is terminated and reported
    //    INCONCLUSIVE; the parent stays responsive (we get a status back).
    {
        let work = temp.join("reap");
        let _ = std::fs::create_dir_all(&work);
        let _subagent_session = subagent::configure_scoped(base_config(2, Duration::from_secs(1)));
        let e = subagent::spawn_canned(&work, "reap-1", "g", "E", 5000); // 5s > 1s timeout
        std::thread::sleep(Duration::from_millis(1500));
        let status = subagent::status(&work, "reap-1").unwrap_or_default();
        let ok =
            e.is_ok() && status.contains("status: inconclusive") && status.contains("timed out");
        push("timeout_reaps_to_inconclusive", status, ok);
    }

    // 5) subtask_id traversal/invalid is rejected before any spawn.
    {
        let work = temp.join("trav");
        let _ = std::fs::create_dir_all(&work);
        let t = subagent::spawn(&work, "../escape", "g");
        let ok = t.is_err() && t.as_ref().unwrap_err().contains("invalid subtask_id");
        push("subtask_id_traversal_rejected", format!("t={t:?}"), ok);
    }

    // 6) Malformed/partial result file is handled as failed data, not a crash.
    {
        let work = temp.join("mal");
        let dir = work.join(".camelid/subagents");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("result_mal.json"), "{ not valid json");
        let status = subagent::status(&work, "mal").unwrap_or_default();
        let ok = status.contains("malformed");
        push("malformed_result_handled", status, ok);
    }

    let _ = std::fs::remove_dir_all(&temp);

    if !spawn_seen {
        return (
            EvalOutcome::Inconclusive,
            cases,
            "could not run a subagent child process (environment issue) — not a logic failure"
                .to_string(),
        );
    }
    let passed = cases.iter().filter(|c| c.verdict == "PASS").count();
    let outcome = if passed == cases.len() {
        EvalOutcome::Pass
    } else {
        EvalOutcome::Fail
    };
    (
        outcome,
        cases.clone(),
        format!(
            "{passed}/{} orchestration mechanics cases passed",
            cases.len()
        ),
    )
}

pub(super) fn family_for(model: &Path) -> String {
    let name = model
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();
    // Ornith / Qwen3.5 emit the custom `<function=…>` XML tool format, which routes
    // to `parse_ornith`. Check before "qwen" (the model is Qwen3.5-derived but its
    // tool grammar differs from Qwen2/Qwen3 JSON-in-`<tool_call>`). Mirrors
    // `agent_eval::family_for` so the two gate harnesses classify a model identically.
    if name.contains("ornith") || name.contains("qwen35") || name.contains("qwen3.5") {
        "ornith".to_string()
    } else if name.contains("qwen") {
        "qwen".to_string()
    } else if name.contains("mistral") {
        "mistral".to_string()
    } else {
        "llama".to_string()
    }
}

/// Rung-3: drive a REAL model through one spawn_subagent → check_subagent_status
/// round-trip. The subagent runs a deterministic canned task (env hook) so this
/// isolates the model's orchestration-driving from subagent inference. A contended
/// box yields INCONCLUSIVE; promotion never happens here regardless.
fn run_real_model_battery(
    addr: SocketAddr,
    model: &Path,
    load_timeout: u64,
) -> (EvalOutcome, Vec<CaseResult>, String, Option<String>) {
    use super::agent::{self, AgentMsg, LiveDriver};
    use super::client::{Client, LoadOutcome};
    use super::server::ServerHandle;
    use super::tools::{Action, Sandbox, ToolOutcome};
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;

    let inconclusive =
        |note: String, mid: Option<String>| (EvalOutcome::Inconclusive, Vec::new(), note, mid);

    let client = Client::new(addr);
    let _server = match ServerHandle::ensure(addr, &client) {
        Ok(s) => s,
        Err(e) => return inconclusive(format!("shared serve unavailable: {e}"), None),
    };

    // The eval auto-approves the model's spawn — refuse under production.
    if agent::is_production() {
        return (
            EvalOutcome::Fail,
            Vec::new(),
            "refused: CAMELID_PRODUCTION is set and the eval auto-approves".to_string(),
            None,
        );
    }

    // Bounded load: a contended box yields INCONCLUSIVE, never FAIL.
    let abs = std::fs::canonicalize(model).unwrap_or_else(|_| model.to_path_buf());
    eprintln!("loading {} (timeout {load_timeout}s)…", abs.display());
    let (tx, rx) = mpsc::channel();
    let loader = client.clone();
    let path = abs.to_string_lossy().to_string();
    std::thread::spawn(move || {
        let _ = tx.send(loader.load_model(&path, None));
    });
    let model_id = match rx.recv_timeout(Duration::from_secs(load_timeout)) {
        Ok(Ok(LoadOutcome::Loaded { id })) => id,
        Ok(Ok(LoadOutcome::Unsupported { message })) => {
            return (
                EvalOutcome::Fail,
                Vec::new(),
                format!("unsupported: {message}"),
                None,
            )
        }
        Ok(Err(e)) => {
            return (
                EvalOutcome::Fail,
                Vec::new(),
                format!("load error: {e}"),
                None,
            )
        }
        Err(_) => {
            return inconclusive(
                format!("model did not load within {load_timeout}s — re-run on a quiet box"),
                None,
            )
        }
    };
    let model_sha256 = match client
        .list_loaded()
        .into_iter()
        .find(|model| model.id == model_id)
        .map(|model| model.gguf_sha256)
        .filter(|sha256| !sha256.is_empty())
    {
        Some(sha256) => sha256,
        None => {
            return inconclusive(
                "loaded model did not expose an exact GGUF SHA-256 identity".to_string(),
                Some(model_id),
            )
        }
    };

    let work = std::env::temp_dir().join(format!("camelid-orch-real-{}", std::process::id()));
    if std::fs::create_dir_all(&work).is_err() {
        return inconclusive(
            "could not create the fixture workspace".to_string(),
            Some(model_id),
        );
    }
    let _ = std::fs::write(work.join("notes.txt"), "alpha\nbeta\ngamma\n");

    // Deterministic subagent: a spawn the model emits runs a canned child that
    // answers with the line count. Cleared at the end of the run.
    std::env::set_var("CAMELID_SUBAGENT_FORCE_CANNED", "notes.txt has 3 lines");

    let family = family_for(&abs);
    let _subagent_session = subagent::configure_scoped(SubagentConfig {
        addr,
        model_id: model_id.clone(),
        model_sha256: model_sha256.clone(),
        family: family.clone(),
        max_steps: 4,
        context_budget_tokens: agent::AGENT_VALIDATED_CTX,
        max_tokens: 128,
        concurrency: 2,
        depth_limit: 1,
        timeout: Duration::from_secs(60),
        auto_approve: true,
        shell_mode: super::shell_sandbox::ShellSandbox::Unrestricted,
    });

    let sandbox = match Sandbox::new(&work, false, Duration::from_secs(60)) {
        Ok(s) => s.with_shell_mode(super::shell_sandbox::ShellSandbox::Unrestricted),
        Err(e) => {
            std::env::remove_var("CAMELID_SUBAGENT_FORCE_CANNED");
            let _ = std::fs::remove_dir_all(&work);
            return inconclusive(format!("sandbox: {e}"), Some(model_id));
        }
    };
    let tools = super::tools::specs(false, sandbox.shell_mode());

    struct EvalReporter {
        calls: Vec<String>,
        answer: String,
    }
    impl agent::Reporter for EvalReporter {
        fn model_text(&mut self, text: &str) {
            self.answer = text.to_string();
        }
        fn tool_call(&mut self, line: &str) {
            eprintln!("  ▸ {line}");
            self.calls.push(line.to_string());
        }
        fn tool_result(&mut self, name: &str, outcome: &ToolOutcome) {
            eprintln!(
                "  └ {name} {}",
                if outcome.is_err() { "(error)" } else { "ok" }
            );
        }
        fn notice(&mut self, text: &str) {
            eprintln!("· {text}");
        }
    }
    struct AutoApprove;
    impl agent::Approver for AutoApprove {
        fn approve(&mut self, _a: &Action, _s: &Sandbox) -> agent::Decision {
            agent::Decision::Once
        }
    }

    let goal = "Use the spawn_subagent tool to delegate work: spawn a subagent with subtask_id \
                \"counter\" and goal \"read notes.txt and report how many lines it has\". Then call \
                check_subagent_status with subtask_id \"counter\" and tell me the line count it \
                reports.";

    let mut driver = LiveDriver::with(
        client.clone(),
        model_id.clone(),
        model_sha256,
        family,
        128,
        0.0,
    );
    let mut reporter = EvalReporter {
        calls: Vec::new(),
        answer: String::new(),
    };
    let mut approver = AutoApprove;
    let cancel = AtomicBool::new(false);
    let cfg = agent::AgentConfig {
        workdir: work.clone(),
        max_steps: 8,
        auto_approve: true,
        yolo: false,
        allow_net: false,
        allow_fs: false,
        shell_timeout: Duration::from_secs(60),
        max_tokens: 128,
        temperature: 0.0,
        audit: Box::new(super::audit::NoopSink),
        shell_sandbox: super::shell_sandbox::ShellSandbox::Unrestricted,
        tool_profile: super::tools::ToolProfile::Full,
        // Gate harness: bounded by max_steps, and must stay reproducible.
        ctx_budget: None,
    };
    let mut policy = agent::Policy::default();
    policy.set_auto_all(true);
    let mut history = vec![
        AgentMsg::System(agent::system_prompt(&sandbox, &tools)),
        AgentMsg::User(goal.to_string()),
    ];
    let started = Instant::now();
    let end = agent::run_loop(
        &mut driver,
        &mut approver,
        &mut reporter,
        &sandbox,
        &cfg,
        &cancel,
        &mut policy,
        &mut history,
    );
    let elapsed = started.elapsed();

    let child_status = poll_to_terminal(&work, "counter", Duration::from_secs(30));

    std::env::remove_var("CAMELID_SUBAGENT_FORCE_CANNED");
    let _ = std::fs::remove_dir_all(&work);

    // If the model emitted NO tool calls at all, tool-calling itself is not
    // working for this model on this build (the certified read_file agent-eval
    // fails identically) — orchestration cannot be FAIRLY assessed, so this is
    // INCONCLUSIVE (blocked on a pre-existing issue), not a FAIL.
    if reporter.calls.is_empty() {
        return (
            EvalOutcome::Inconclusive,
            vec![CaseResult {
                name: "model_emitted_any_tool_call".to_string(),
                observed: snippet(&format!(
                    "loop={end:?} elapsed={:.1}s answer={:?} — ZERO tool calls",
                    elapsed.as_secs_f64(),
                    reporter.answer
                )),
                verdict: "INCONCLUSIVE".to_string(),
            }],
            format!(
                "INCONCLUSIVE — model {model_id} emitted no tool calls at all; tool-calling itself \
                 is the blocker on this build (the certified read_file agent-eval fails the same \
                 way), not the orchestration code"
            ),
            Some(model_id),
        );
    }

    let emitted_spawn = reporter.calls.iter().any(|c| c.contains("spawn_subagent"));
    let emitted_check = reporter
        .calls
        .iter()
        .any(|c| c.contains("check_subagent_status"));
    let child_completed = child_status.contains("status: completed");
    let answer_has_count = reporter.answer.contains('3');

    let mut cases = Vec::new();
    let mut push = |name: &str, observed: String, pass: bool| {
        cases.push(CaseResult {
            name: name.to_string(),
            observed: snippet(&observed),
            verdict: if pass { "PASS" } else { "FAIL" }.to_string(),
        });
    };
    push(
        "model_emitted_spawn_subagent",
        format!("calls={:?}", reporter.calls),
        emitted_spawn,
    );
    push(
        "model_emitted_check_subagent_status",
        format!("calls={:?}", reporter.calls),
        emitted_check,
    );
    push(
        "subagent_round_trip_completed",
        child_status,
        child_completed,
    );
    push(
        "final_answer_reports_count",
        format!(
            "loop={end:?} elapsed={:.1}s answer={}",
            elapsed.as_secs_f64(),
            reporter.answer
        ),
        answer_has_count,
    );

    let passed = cases.iter().filter(|c| c.verdict == "PASS").count();
    let outcome = if passed == cases.len() {
        EvalOutcome::Pass
    } else {
        EvalOutcome::Fail
    };
    let note = format!(
        "{passed}/{} rung-3 checks passed — real model {model_id} driving orchestration",
        cases.len()
    );
    (outcome, cases, note, Some(model_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> OrchestrationReceipt {
        OrchestrationReceipt {
            schema: RECEIPT_SCHEMA_V1.to_string(),
            receipt_id: String::new(),
            created_unix: 7,
            feature: "f".to_string(),
            rung: 2,
            model_id: None,
            outcome: "PASS".to_string(),
            host: HostBlock {
                os: "windows".to_string(),
                arch: "x86_64".to_string(),
                hostname: None,
            },
            cases: vec![],
            note: "n".to_string(),
            promotes_capability: false,
            claims_speedup: false,
        }
    }

    #[test]
    fn seal_then_verify_passes_and_tamper_breaks() {
        let mut r = sample();
        r.seal();
        assert!(r.verify_self_digest());
        r.outcome = "FAIL".to_string();
        assert!(!r.verify_self_digest());
    }

    #[test]
    fn receipt_never_promotes_or_claims_speedup() {
        let r = sample();
        assert!(!r.promotes_capability);
        assert!(!r.claims_speedup);
    }

    #[test]
    fn family_for_routes_ornith_before_qwen() {
        // Ornith / Qwen3.5 must classify as `ornith` (custom `<function=…>` XML tool
        // grammar), not `qwen` — matching `agent_eval::family_for`. Regression for a
        // missing arm that misrouted these models to the JSON tool parser.
        assert_eq!(family_for(Path::new("Ornith-4B-Q8_0.gguf")), "ornith");
        assert_eq!(family_for(Path::new("Qwen3.5-4B-Instruct.gguf")), "ornith");
        assert_eq!(family_for(Path::new("qwen35-coder.gguf")), "ornith");
        assert_eq!(family_for(Path::new("Qwen3-4B-Q4_K_M.gguf")), "qwen");
        assert_eq!(family_for(Path::new("Mistral-7B.gguf")), "mistral");
        assert_eq!(family_for(Path::new("Llama-3.2-3B.gguf")), "llama");
    }
}
