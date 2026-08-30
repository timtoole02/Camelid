//! Standalone verifier for the sealed agent-family receipts, exposed through
//! `camelid verify-receipt`.
//!
//! The agent gates each emit a **tamper-evident** receipt sealed with the shared
//! [`receipt_id`](super::receipt_id_over) digest:
//!
//! - `agent-syscap-eval` → `camelid.agent-syscap-receipt/v1`
//! - `agent-orchestration-eval` → `camelid.agent-orchestration-receipt/v1`
//! - `agent-orchestration-bench` → `camelid.agent-orchestration-bench/v1`
//! - `agent-eval` → `camelid.agent_eval/v2` (the tool-capable promotion gate;
//!   legacy v1 remains verifiable under its stricter complete-battery rule)
//!
//! Unlike a parity receipt there is no model to re-run: verification is a
//! self-contained **tamper-evidence + honest-scope** check.
//!
//! 1. *Tamper-evidence* — the document's canonical body (every field except
//!    `receipt_id`, recursively key-sorted and compact) must hash to the stored
//!    `receipt_id`. Any added, removed, or altered field breaks the match.
//! 2. *Honest-scope* — a well-formed but over-claiming receipt is rejected. The
//!    syscap / orchestration / bench gates promote no capability (and
//!    orchestration claims no speedup); the eval gate is the one
//!    promotion-bearing receipt. New receipts distinguish a complete
//!    `full_battery` from diagnostic `single_case` runs; only a passing complete
//!    battery may set `promotion_eligible=true`. Sealed legacy receipts retain
//!    their original PASS/eligibility consistency rule.
//!
//! Verifying a receipt attests only that the document is intact and internally
//! honest; it changes no support-ledger row. Agent-eval receipts minted before
//! sealing carry no `receipt_id`; those are reported as unsealed legacy receipts
//! (their tamper-evidence cannot be established), never as verified.
//!
//! *What the seal does and does not prove.* The `receipt_id` is an **unkeyed**
//! SHA-256, not a signature. It makes accidental corruption and naive hand-edits
//! of a committed receipt evident — the digest stops matching — and honest-scope
//! catches a casual field flip the digest alone would miss (e.g. editing
//! `outcome` without resealing). It is **not** forgery-resistant: anyone who can
//! run `camelid` can mint a fresh, fully-consistent sealed receipt, so a
//! determined forger who reseals is not detected here. Git history and review
//! remain the integrity anchor for what a receipt attests.

use std::path::Path;

use serde_json::{Map, Value};

use super::receipt_id_over;

/// Schema stamped into an `agent-syscap-eval` receipt.
pub const SYSCAP_RECEIPT_SCHEMA_V1: &str = "camelid.agent-syscap-receipt/v1";
/// Schema stamped into an `agent-orchestration-eval` receipt.
pub const ORCHESTRATION_RECEIPT_SCHEMA_V1: &str = "camelid.agent-orchestration-receipt/v1";
/// Schema stamped into an `agent-orchestration-bench` receipt.
pub const ORCHESTRATION_BENCH_SCHEMA_V1: &str = "camelid.agent-orchestration-bench/v1";
/// Schema stamped into an `agent-eval` receipt (the tool-capable promotion gate).
pub const EVAL_RECEIPT_SCHEMA_V1: &str = "camelid.agent_eval/v1";
/// Scope- and artifact-bound agent-eval receipt. Unlike legacy v1, v2 requires
/// an explicit evaluation scope plus the exact GGUF filename and SHA-256.
pub const EVAL_RECEIPT_SCHEMA_V2: &str = "camelid.agent_eval/v2";

/// Every agent-family schema this verifier understands.
pub const RECOGNIZED_SCHEMAS: [&str; 5] = [
    SYSCAP_RECEIPT_SCHEMA_V1,
    ORCHESTRATION_RECEIPT_SCHEMA_V1,
    ORCHESTRATION_BENCH_SCHEMA_V1,
    EVAL_RECEIPT_SCHEMA_V1,
    EVAL_RECEIPT_SCHEMA_V2,
];

/// True when `schema` names an agent-family receipt this module can verify.
/// The CLI uses this to route a receipt to this verifier instead of the parity
/// verifier.
pub fn is_agent_schema(schema: &str) -> bool {
    RECOGNIZED_SCHEMAS.contains(&schema)
}

/// Which agent gate produced a receipt. Each family has a distinct body shape
/// and a distinct honest-scope contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentReceiptClass {
    /// `camelid.agent-syscap-receipt/v1`.
    Syscap,
    /// `camelid.agent-orchestration-receipt/v1`.
    Orchestration,
    /// `camelid.agent-orchestration-bench/v1`.
    OrchestrationBench,
    /// `camelid.agent_eval/v1` — the tool-capable promotion gate.
    Eval,
}

impl AgentReceiptClass {
    fn from_schema(schema: &str) -> Option<Self> {
        match schema {
            SYSCAP_RECEIPT_SCHEMA_V1 => Some(Self::Syscap),
            ORCHESTRATION_RECEIPT_SCHEMA_V1 => Some(Self::Orchestration),
            ORCHESTRATION_BENCH_SCHEMA_V1 => Some(Self::OrchestrationBench),
            EVAL_RECEIPT_SCHEMA_V1 | EVAL_RECEIPT_SCHEMA_V2 => Some(Self::Eval),
            _ => None,
        }
    }

    /// The field carrying the receipt's own top-line result: `outcome` for the
    /// eval gates (PASS/FAIL/INCONCLUSIVE), `verdict` for the wall-clock bench.
    fn result_field(self) -> &'static str {
        match self {
            Self::Syscap | Self::Orchestration | Self::Eval => "outcome",
            Self::OrchestrationBench => "verdict",
        }
    }

    /// Verify the family's honest-scope invariants and return a human-readable
    /// summary of what was checked. The digest is already confirmed by the time
    /// this runs, so these guards additionally reject a *freshly sealed* but
    /// over-claiming receipt (one the digest alone would accept).
    fn check_honest_scope(self, obj: &Map<String, Value>) -> Result<String, AgentVerifyError> {
        match self {
            Self::Syscap => {
                require_false(obj, "promotes_capability")?;
                Ok("promotes_capability=false".to_string())
            }
            Self::Orchestration => {
                require_false(obj, "promotes_capability")?;
                require_false(obj, "claims_speedup")?;
                Ok("promotes_capability=false, claims_speedup=false".to_string())
            }
            Self::OrchestrationBench => {
                // The bench measures wall-clock and never carries
                // `promotes_capability`. Its honest-scope field is
                // `speedup_claimed_for`: a string naming the one measured
                // workload a speedup may be attributed to (or "none").
                let claimed = obj
                    .get("speedup_claimed_for")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AgentVerifyError {
                        phase: "honest-scope",
                        reason: "bench receipt is missing the string `speedup_claimed_for` \
                                 scope field"
                            .to_string(),
                    })?;
                Ok(format!("speedup_claimed_for={claimed:?}"))
            }
            Self::Eval => {
                let schema = obj
                    .get("schema")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if schema == EVAL_RECEIPT_SCHEMA_V1 {
                    let cases = obj.get("cases").and_then(Value::as_array).ok_or_else(|| {
                        AgentVerifyError {
                            phase: "honest-scope",
                            reason: "legacy v1 agent_eval receipt is missing `cases`".to_string(),
                        }
                    })?;
                    if !eval_case_set_status(cases).0 {
                        return Err(AgentVerifyError {
                            phase: "honest-scope",
                            reason: "legacy v1 agent_eval receipts are accepted only with each pinned case exactly once"
                                .to_string(),
                        });
                    }
                }
                if schema == EVAL_RECEIPT_SCHEMA_V2 {
                    let filename = obj
                        .get("gguf_filename")
                        .and_then(Value::as_str)
                        .filter(|filename| !filename.trim().is_empty())
                        .ok_or_else(|| AgentVerifyError {
                            phase: "honest-scope",
                            reason: "v2 agent_eval receipt is missing `gguf_filename`".to_string(),
                        })?;
                    let sha256 = obj
                        .get("gguf_sha256")
                        .and_then(Value::as_str)
                        .filter(|sha256| is_sha256_hex(sha256))
                        .ok_or_else(|| AgentVerifyError {
                            phase: "honest-scope",
                            reason: "v2 agent_eval receipt requires a lowercase hexadecimal `gguf_sha256`"
                                .to_string(),
                        })?;
                    if obj
                        .get("evaluation_scope")
                        .and_then(Value::as_str)
                        .is_none()
                    {
                        return Err(AgentVerifyError {
                            phase: "honest-scope",
                            reason: "v2 agent_eval receipt requires an explicit `evaluation_scope`; it cannot fall back to legacy promotion rules"
                                .to_string(),
                        });
                    }
                    let _artifact_identity = (filename, sha256);
                }
                // The eval gate is the ONE agent receipt that may justify a
                // promotion: a PASS is `promotion_eligible`. Honest-scope here is
                // internal consistency — `promotion_eligible` must equal
                // `outcome == "PASS"`. A receipt claiming eligibility without a
                // PASS (or denying it after one) is dishonest.
                let outcome =
                    obj.get("outcome")
                        .and_then(Value::as_str)
                        .ok_or_else(|| AgentVerifyError {
                            phase: "honest-scope",
                            reason: "agent_eval receipt is missing the string `outcome` field"
                                .to_string(),
                        })?;
                let eligible = obj
                    .get("promotion_eligible")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| AgentVerifyError {
                        phase: "honest-scope",
                        reason: "agent_eval receipt is missing the boolean \
                                 `promotion_eligible` field"
                            .to_string(),
                    })?;
                if let Some(scope) = obj.get("evaluation_scope").and_then(Value::as_str) {
                    match scope {
                        "full_battery" => {
                            let cases =
                                obj.get("cases").and_then(Value::as_array).ok_or_else(|| {
                                    AgentVerifyError {
                                        phase: "honest-scope",
                                        reason:
                                            "full-battery agent_eval receipt is missing `cases`"
                                                .to_string(),
                                    }
                                })?;
                            let (complete, all_passed) = eval_case_set_status(cases);
                            if !complete
                                || (outcome == "PASS") != all_passed
                                || eligible != (outcome == "PASS" && all_passed)
                            {
                                return Err(AgentVerifyError {
                                    phase: "honest-scope",
                                    reason: format!(
                                        "full_battery requires each pinned case exactly once, a \
                                         PASS only when all cases passed, and matching promotion \
                                         eligibility; got promotion_eligible={eligible}, \
                                         outcome={outcome:?}"
                                    ),
                                });
                            }
                        }
                        "single_case" => {
                            if eligible {
                                return Err(AgentVerifyError {
                                    phase: "honest-scope",
                                    reason: format!(
                                        "evaluation_scope={scope:?} is diagnostic only and cannot \
                                         be promotion_eligible"
                                    ),
                                });
                            }
                            let cases = obj.get("cases").and_then(Value::as_array);
                            let one_valid_case = cases.is_some_and(|cases| {
                                cases.len() == 1
                                    && cases[0].get("case").and_then(Value::as_str).is_some_and(
                                        |name| {
                                            ["read_and_count", "list_dir_find", "write_greeting"]
                                                .contains(&name)
                                        },
                                    )
                            });
                            if !one_valid_case {
                                return Err(AgentVerifyError {
                                    phase: "honest-scope",
                                    reason: "single_case scope requires exactly one pinned case"
                                        .to_string(),
                                });
                            }
                            let passed = cases
                                .and_then(|cases| cases[0].get("passed"))
                                .and_then(Value::as_bool);
                            if passed != Some(outcome == "PASS") {
                                return Err(AgentVerifyError {
                                    phase: "honest-scope",
                                    reason: format!(
                                        "single_case outcome {outcome:?} is inconsistent with \
                                         passed={passed:?}"
                                    ),
                                });
                            }
                        }
                        "incomplete_or_not_run" => {
                            if eligible {
                                return Err(AgentVerifyError {
                                    phase: "honest-scope",
                                    reason: "incomplete agent_eval evidence cannot be promotion_eligible"
                                        .to_string(),
                                });
                            }
                        }
                        other => {
                            return Err(AgentVerifyError {
                                phase: "honest-scope",
                                reason: format!("unknown agent_eval evaluation_scope={other:?}"),
                            })
                        }
                    }
                    return Ok(format!(
                        "evaluation_scope={scope:?}, promotion_eligible={eligible}, \
                         outcome={outcome:?}"
                    ));
                }
                // Legacy v1 receipts had no scope field. Because the self-digest
                // is deliberately unkeyed, accepting `PASS => eligible` here
                // lets a one-case diagnostic strip its scope, flip eligibility,
                // and reseal. Legacy promotion is therefore accepted only when
                // the receipt itself contains the exact complete pinned battery.
                let cases =
                    obj.get("cases")
                        .and_then(Value::as_array)
                        .ok_or_else(|| AgentVerifyError {
                            phase: "honest-scope",
                            reason: "legacy v1 agent_eval receipt is missing `cases`".to_string(),
                        })?;
                let (complete, all_passed) = eval_case_set_status(cases);
                if schema != EVAL_RECEIPT_SCHEMA_V1
                    || !complete
                    || (outcome == "PASS") != all_passed
                    || eligible != (outcome == "PASS" && all_passed)
                {
                    return Err(AgentVerifyError {
                        phase: "honest-scope",
                        reason: format!(
                            "legacy v1 agent_eval promotion requires each pinned case exactly \
                             once, a PASS only when all passed, and matching eligibility; got \
                             promotion_eligible={eligible}, outcome={outcome:?}"
                        ),
                    });
                }
                Ok(format!(
                    "legacy_full_battery=true, promotion_eligible={eligible}, outcome={outcome:?}"
                ))
            }
        }
    }
}

fn eval_case_set_status(cases: &[Value]) -> (bool, bool) {
    const REQUIRED: [&str; 3] = ["read_and_count", "list_dir_find", "write_greeting"];
    let complete = cases.len() == REQUIRED.len()
        && REQUIRED.iter().all(|name| {
            cases
                .iter()
                .filter(|case| case.get("case").and_then(Value::as_str) == Some(*name))
                .count()
                == 1
        });
    let all_passed = complete
        && cases
            .iter()
            .all(|case| case.get("passed").and_then(Value::as_bool) == Some(true));
    (complete, all_passed)
}

/// A boolean scope field that every honest receipt of the family pins to
/// `false` (e.g. `promotes_capability`, `claims_speedup`).
fn require_false(obj: &Map<String, Value>, field: &str) -> Result<(), AgentVerifyError> {
    match obj.get(field).and_then(Value::as_bool) {
        Some(false) => Ok(()),
        Some(true) => Err(AgentVerifyError {
            phase: "honest-scope",
            reason: format!(
                "receipt sets {field}=true; an agent gate promotes no capability and this \
                 field must be false"
            ),
        }),
        None => Err(AgentVerifyError {
            phase: "honest-scope",
            reason: format!("receipt is missing the boolean `{field}` scope field"),
        }),
    }
}

/// The outcome of verifying an agent-family receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentVerifyOutcome {
    /// Schema recognized, digest intact, honest-scope invariants hold.
    Verified,
    /// A verification step failed (unrecognized schema, tampered/malformed body,
    /// or an over-claiming scope field).
    NotVerified,
}

impl AgentVerifyOutcome {
    /// Process exit code: 0 verified, 1 not verified. (No divergence/lossy axes
    /// apply to an agent receipt, so the code set is deliberately just 0/1.)
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Verified => 0,
            Self::NotVerified => 1,
        }
    }
}

/// The verification step that failed, and why. `phase` is a stable short label
/// echoed in the final verdict line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentVerifyError {
    pub phase: &'static str,
    pub reason: String,
}

/// What a verified receipt attests, for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentReceiptSummary {
    pub schema: String,
    pub class: AgentReceiptClass,
    pub receipt_id: String,
    /// The receipt's own top-line result (`outcome` or `verdict`), verbatim.
    pub result: String,
    /// Human-readable summary of the honest-scope fields that were checked.
    pub scope_note: String,
    /// `os/arch` (optionally with `(hostname)`) from the receipt's `host` block,
    /// or `None` when the receipt has no host block (e.g. `agent_eval`).
    pub host: Option<String>,
}

/// Pure verification: no I/O, no printing. Returns what the receipt attests, or
/// the first failing step. Exhaustively unit-tested.
pub fn check(value: &Value) -> Result<AgentReceiptSummary, AgentVerifyError> {
    let obj = value.as_object().ok_or_else(|| AgentVerifyError {
        phase: "parse",
        reason: "receipt is not a JSON object".to_string(),
    })?;

    // Schema — must be one this verifier understands.
    let schema = obj
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentVerifyError {
            phase: "schema",
            reason: "receipt has no string `schema` field".to_string(),
        })?;
    let class = AgentReceiptClass::from_schema(schema).ok_or_else(|| AgentVerifyError {
        phase: "schema",
        reason: format!(
            "unrecognized schema {schema:?}; this verifier understands {RECOGNIZED_SCHEMAS:?}"
        ),
    })?;

    // Receipt id — must be a 64-char lowercase-hex SHA-256. Agent-eval receipts
    // minted before sealing carry none; report those as unsealed legacy receipts
    // rather than as a malformed body.
    let receipt_id = match obj.get("receipt_id").and_then(Value::as_str) {
        Some(id) => id,
        None => {
            let reason = if class == AgentReceiptClass::Eval {
                "unsealed legacy agent_eval receipt (no `receipt_id`): it predates receipt \
                 sealing, so its tamper-evidence cannot be established — re-run agent-eval \
                 to mint a sealed receipt"
                    .to_string()
            } else {
                "receipt has no string `receipt_id` field".to_string()
            };
            return Err(AgentVerifyError {
                phase: "self-digest",
                reason,
            });
        }
    };
    if !is_sha256_hex(receipt_id) {
        return Err(AgentVerifyError {
            phase: "self-digest",
            reason: format!(
                "receipt_id {receipt_id:?} is not a 64-character lowercase-hex SHA-256"
            ),
        });
    }

    // Strict self-digest over the exact document body.
    let computed = receipt_id_over(value);
    if computed != receipt_id {
        return Err(AgentVerifyError {
            phase: "self-digest",
            reason: format!(
                "receipt_id does not match the canonical body (stored {receipt_id}, computed \
                 {computed}); the receipt is tampered or malformed"
            ),
        });
    }

    // Honest-scope — reject a well-formed but over-claiming receipt.
    let scope_note = class.check_honest_scope(obj)?;

    Ok(AgentReceiptSummary {
        schema: schema.to_string(),
        class,
        receipt_id: receipt_id.to_string(),
        result: obj
            .get(class.result_field())
            .and_then(Value::as_str)
            .unwrap_or("(none)")
            .to_string(),
        scope_note,
        host: host_line(obj),
    })
}

/// Verify a parsed receipt, printing one PASS/FAIL/NOTE line per step and a
/// single final verdict, and return the outcome.
pub fn verify_value(value: &Value) -> AgentVerifyOutcome {
    match check(value) {
        Ok(summary) => {
            println!(
                "PASS schema: recognized agent-family receipt ({})",
                summary.schema
            );
            println!(
                "PASS self-digest: receipt_id matches the canonical body ({})",
                summary.receipt_id
            );
            println!("PASS honest-scope: {}", summary.scope_note);
            match &summary.host {
                Some(host) => println!(
                    "NOTE receipt: {}={}; host {host}",
                    summary.class.result_field(),
                    summary.result,
                ),
                None => println!(
                    "NOTE receipt: {}={}",
                    summary.class.result_field(),
                    summary.result,
                ),
            }
            println!("AGENT RECEIPT VERIFIED (tamper-evident; changes no support-ledger row)");
            AgentVerifyOutcome::Verified
        }
        Err(err) => {
            println!("FAIL {}: {}", err.phase, err.reason);
            println!("AGENT RECEIPT NOT VERIFIED ({})", err.phase);
            AgentVerifyOutcome::NotVerified
        }
    }
}

/// Read a receipt file, parse it, and verify it. Read/parse failures are
/// reported as a `NotVerified` outcome (never a panic).
pub fn run(path: &Path) -> AgentVerifyOutcome {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            println!("FAIL parse: could not read {}: {err}", path.display());
            println!("AGENT RECEIPT NOT VERIFIED (parse)");
            return AgentVerifyOutcome::NotVerified;
        }
    };
    let value: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(err) => {
            println!("FAIL parse: receipt does not parse as JSON: {err}");
            println!("AGENT RECEIPT NOT VERIFIED (parse)");
            return AgentVerifyOutcome::NotVerified;
        }
    };
    verify_value(&value)
}

/// A 64-character lowercase-hex string, as produced by [`super::sha256_hex`].
fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// `os/arch` (optionally with `(hostname)`) from the receipt's `host` block, or
/// `None` when the receipt carries no host block. A present-but-partial block
/// renders `?` for the missing piece.
fn host_line(obj: &Map<String, Value>) -> Option<String> {
    let host = obj.get("host").and_then(Value::as_object)?;
    let field = |key: &str| host.get(key).and_then(Value::as_str).unwrap_or("?");
    let os = field("os");
    let arch = field("arch");
    Some(match host.get("hostname").and_then(Value::as_str) {
        Some(name) => format!("{os}/{arch} ({name})"),
        None => format!("{os}/{arch}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Stamp `receipt_id` over the body exactly as the emitters do, so the test
    /// fixtures are sealed by the same primitive the verifier checks.
    fn seal(mut body: Value) -> Value {
        let id = receipt_id_over(&body);
        body.as_object_mut()
            .expect("body is an object")
            .insert("receipt_id".to_string(), Value::String(id));
        body
    }

    fn syscap_body() -> Value {
        json!({
            "schema": SYSCAP_RECEIPT_SCHEMA_V1,
            "receipt_id": "",
            "created_unix": 1_784_747_759u64,
            "feature": "windows-system-control: run_windows_command (Exec) + inspect_system (Read)",
            "outcome": "PASS",
            "host": { "os": "windows", "arch": "x86_64", "hostname": "TEST-BOX" },
            "cases": [
                { "name": "echo", "input": "echo hi", "observed": "hi", "verdict": "PASS" }
            ],
            "note": "syscap tools verified under the gate",
            "promotes_capability": false
        })
    }

    fn orchestration_body() -> Value {
        json!({
            "schema": ORCHESTRATION_RECEIPT_SCHEMA_V1,
            "receipt_id": "",
            "created_unix": 1_784_747_760u64,
            "feature": "subagent-orchestration: spawn_subagent + check_subagent_status",
            "rung": 2,
            "outcome": "PASS",
            "host": { "os": "linux", "arch": "aarch64" },
            "cases": [ { "name": "spawn", "observed": "collected", "verdict": "PASS" } ],
            "note": "stub round-trip + caps/depth/reaping",
            "promotes_capability": false,
            "claims_speedup": false
        })
    }

    fn bench_body() -> Value {
        json!({
            "schema": ORCHESTRATION_BENCH_SCHEMA_V1,
            "receipt_id": "",
            "created_unix": 1_784_747_761u64,
            "rung": 4,
            "host": { "os": "windows", "arch": "x86_64", "hostname": "BENCH-BOX" },
            "workloads": [
                {
                    "name": "io_bound_sleep",
                    "subagents": 4,
                    "sequential_ms": 400,
                    "concurrent_ms": 120,
                    "speedup": 3.3333333333333335,
                    "note": "sleeps overlap"
                }
            ],
            "verdict": "io-bound concurrency helped; inference-bound not measured",
            "speedup_claimed_for": "io_bound_sleep",
            "notes": []
        })
    }

    /// Mirrors the fields `agent_eval::finish` emits (a PASS is promotion-eligible).
    fn eval_body() -> Value {
        json!({
            "schema": EVAL_RECEIPT_SCHEMA_V2,
            "receipt_id": "",
            "outcome": "PASS",
            "model_id": "qwen3-4b",
            "gguf": "C:\\models\\Qwen3-4B-Q8_0.gguf",
            "gguf_filename": "Qwen3-4B-Q8_0.gguf",
            "gguf_sha256": "ab".repeat(32),
            "gguf_bytes": 4_000_000_000u64,
            "quantization": "Q8_0",
            "note": "full 3-case battery",
            "cases": [
                {"case":"read_and_count", "passed":true},
                {"case":"list_dir_find", "passed":true},
                {"case":"write_greeting", "passed":true}
            ],
            "host_loadavg_1m": null,
            "timestamp_unix": 1_784_747_762u64,
            "evaluation_scope": "full_battery",
            "promotion_eligible": true
        })
    }

    #[test]
    fn recognizes_exactly_the_agent_schemas() {
        assert!(is_agent_schema(SYSCAP_RECEIPT_SCHEMA_V1));
        assert!(is_agent_schema(ORCHESTRATION_RECEIPT_SCHEMA_V1));
        assert!(is_agent_schema(ORCHESTRATION_BENCH_SCHEMA_V1));
        assert!(is_agent_schema(EVAL_RECEIPT_SCHEMA_V1));
        assert!(is_agent_schema(EVAL_RECEIPT_SCHEMA_V2));
        assert!(!is_agent_schema("camelid.parity-receipt/v1"));
        // The eval schema uses an underscore; the hyphenated spelling is not it.
        assert!(!is_agent_schema("camelid.agent-eval/v1"));
        assert!(!is_agent_schema(""));
    }

    #[test]
    fn verifies_a_sealed_eval_receipt() {
        let receipt = seal(eval_body());
        let summary = check(&receipt).expect("verifies");
        assert_eq!(summary.class, AgentReceiptClass::Eval);
        assert_eq!(summary.result, "PASS");
        assert!(summary
            .scope_note
            .contains("evaluation_scope=\"full_battery\""));
        assert!(summary.scope_note.contains("promotion_eligible=true"));
        assert_eq!(summary.host, None);
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::Verified);
    }

    #[test]
    fn a_sealed_eval_claiming_eligibility_without_pass_fails_honest_scope() {
        // Seal WITH the inconsistency so the digest is intact; only the
        // honest-scope guard should reject it.
        let mut body = eval_body();
        body["outcome"] = json!("FAIL");
        // promotion_eligible stays true — inconsistent with a non-PASS outcome.
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "honest-scope");
        assert!(err.reason.contains("promotion_eligible=true"));
        assert!(err.reason.contains("outcome=\"FAIL\""));
    }

    #[test]
    fn scoped_single_case_pass_is_diagnostic_not_promotion_eligible() {
        let mut body = eval_body();
        body["evaluation_scope"] = json!("single_case");
        body["promotion_eligible"] = json!(false);
        body["cases"] = json!([{"case":"read_and_count", "passed":true}]);
        let receipt = seal(body);
        let summary = check(&receipt).expect("diagnostic PASS remains a valid receipt");
        assert!(summary.scope_note.contains("single_case"));
        assert!(summary.scope_note.contains("promotion_eligible=false"));
    }

    #[test]
    fn v2_single_case_cannot_strip_scope_reseal_and_promote() {
        let mut body = eval_body();
        body["evaluation_scope"] = json!("single_case");
        body["promotion_eligible"] = json!(false);
        body["cases"] = json!([{"case":"read_and_count", "passed":true}]);
        let diagnostic = seal(body.clone());
        assert!(check(&diagnostic).is_ok());

        body.as_object_mut().unwrap().remove("evaluation_scope");
        body["promotion_eligible"] = json!(true);
        let resealed = seal(body);
        let err = check(&resealed).expect_err("v2 may never use the legacy scope fallback");
        assert_eq!(err.phase, "honest-scope");
        assert!(err.reason.contains("explicit `evaluation_scope`"));
    }

    #[test]
    fn v2_eval_requires_exact_artifact_identity() {
        for field in ["gguf_filename", "gguf_sha256"] {
            let mut body = eval_body();
            body.as_object_mut().unwrap().remove(field);
            let receipt = seal(body);
            let err = check(&receipt).expect_err("v2 artifact identity is mandatory");
            assert_eq!(err.phase, "honest-scope");
            assert!(err.reason.contains(field), "{}", err.reason);
        }
    }

    #[test]
    fn legacy_v1_without_scope_requires_the_exact_full_battery() {
        let mut body = eval_body();
        body["schema"] = json!(EVAL_RECEIPT_SCHEMA_V1);
        body.as_object_mut().unwrap().remove("evaluation_scope");
        body.as_object_mut().unwrap().remove("gguf_filename");
        body.as_object_mut().unwrap().remove("gguf_sha256");
        let receipt = seal(body.clone());
        let summary = check(&receipt).expect("complete legacy battery remains verifiable");
        assert!(summary.scope_note.contains("legacy_full_battery=true"));

        body["cases"] = json!([{"case":"read_and_count", "passed":true}]);
        let resealed = seal(body);
        let err = check(&resealed).expect_err("one-case legacy receipt cannot promote");
        assert!(err.reason.contains("each pinned case exactly once"));
    }

    #[test]
    fn scoped_full_battery_requires_all_pinned_passing_cases() {
        let mut body = eval_body();
        body["evaluation_scope"] = json!("full_battery");
        body["cases"] = json!([
            {"case":"read_and_count", "passed":true},
            {"case":"list_dir_find", "passed":true}
        ]);
        let receipt = seal(body);
        let err = check(&receipt).expect_err("an incomplete battery must not promote");
        assert_eq!(err.phase, "honest-scope");
        assert!(err.reason.contains("each pinned case exactly once"));
    }

    #[test]
    fn scoped_full_battery_pass_can_promote() {
        let mut body = eval_body();
        body["evaluation_scope"] = json!("full_battery");
        body["cases"] = json!([
            {"case":"read_and_count", "passed":true},
            {"case":"list_dir_find", "passed":true},
            {"case":"write_greeting", "passed":true}
        ]);
        let receipt = seal(body);
        let summary = check(&receipt).expect("complete battery verifies");
        assert!(summary.scope_note.contains("full_battery"));
        assert!(summary.scope_note.contains("promotion_eligible=true"));
    }

    #[test]
    fn scoped_full_battery_failure_is_valid_but_cannot_promote() {
        let mut body = eval_body();
        body["outcome"] = json!("FAIL");
        body["promotion_eligible"] = json!(false);
        body["evaluation_scope"] = json!("full_battery");
        body["cases"] = json!([
            {"case":"read_and_count", "passed":true},
            {"case":"list_dir_find", "passed":false},
            {"case":"write_greeting", "passed":true}
        ]);
        let receipt = seal(body);
        let summary = check(&receipt).expect("a complete failing battery is honest evidence");
        assert_eq!(summary.result, "FAIL");
        assert!(summary.scope_note.contains("promotion_eligible=false"));
    }

    #[test]
    fn an_unsealed_legacy_eval_receipt_is_reported_as_such() {
        // The 9 committed agent_eval receipts predate sealing: no `receipt_id`.
        // They must be reported as unsealed legacy, not as a malformed body.
        let mut legacy = eval_body();
        legacy["schema"] = json!(EVAL_RECEIPT_SCHEMA_V1);
        legacy.as_object_mut().unwrap().remove("evaluation_scope");
        legacy.as_object_mut().unwrap().remove("gguf_filename");
        legacy.as_object_mut().unwrap().remove("gguf_sha256");
        legacy.as_object_mut().unwrap().remove("receipt_id");
        let err = check(&legacy).expect_err("must fail");
        assert_eq!(err.phase, "self-digest");
        assert!(err.reason.contains("unsealed legacy"));
        assert_eq!(verify_value(&legacy), AgentVerifyOutcome::NotVerified);
    }

    #[test]
    fn verifies_a_sealed_inconclusive_eval_receipt() {
        // The noisy-box path: INCONCLUSIVE is not promotion-eligible, and a
        // consistent receipt (eligible=false) must verify.
        let mut body = eval_body();
        body["outcome"] = json!("INCONCLUSIVE");
        body["promotion_eligible"] = json!(false);
        body["evaluation_scope"] = json!("incomplete_or_not_run");
        body["cases"] = json!([]);
        let receipt = seal(body);
        let summary = check(&receipt).expect("verifies");
        assert_eq!(summary.result, "INCONCLUSIVE");
        assert!(summary
            .scope_note
            .contains("evaluation_scope=\"incomplete_or_not_run\""));
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::Verified);
    }

    #[test]
    fn a_sealed_eval_with_a_float_loadavg_round_trips() {
        // POSIX hosts record `host_loadavg_1m` as a float. Confirm the digest is
        // stable across the f64 serialize/parse round-trip the on-disk path takes.
        let mut body = eval_body();
        body["host_loadavg_1m"] = json!(0.42);
        let receipt = seal(body);
        assert_eq!(check(&receipt).expect("verifies").result, "PASS");
        let text = serde_json::to_string_pretty(&receipt).unwrap();
        let reparsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(verify_value(&reparsed), AgentVerifyOutcome::Verified);
    }

    #[test]
    fn verifies_a_sealed_syscap_receipt() {
        let receipt = seal(syscap_body());
        let summary = check(&receipt).expect("verifies");
        assert_eq!(summary.class, AgentReceiptClass::Syscap);
        assert_eq!(summary.result, "PASS");
        assert_eq!(summary.scope_note, "promotes_capability=false");
        assert_eq!(summary.host.as_deref(), Some("windows/x86_64 (TEST-BOX)"));
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::Verified);
    }

    #[test]
    fn verifies_a_sealed_orchestration_receipt() {
        let receipt = seal(orchestration_body());
        let summary = check(&receipt).expect("verifies");
        assert_eq!(summary.class, AgentReceiptClass::Orchestration);
        assert_eq!(
            summary.scope_note,
            "promotes_capability=false, claims_speedup=false"
        );
        // No hostname in this fixture.
        assert_eq!(summary.host.as_deref(), Some("linux/aarch64"));
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::Verified);
    }

    #[test]
    fn verifies_a_sealed_bench_receipt() {
        let receipt = seal(bench_body());
        let summary = check(&receipt).expect("verifies");
        assert_eq!(summary.class, AgentReceiptClass::OrchestrationBench);
        assert_eq!(
            summary.result,
            "io-bound concurrency helped; inference-bound not measured"
        );
        assert_eq!(summary.scope_note, "speedup_claimed_for=\"io_bound_sleep\"");
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::Verified);
    }

    #[test]
    fn a_tampered_field_fails_the_self_digest() {
        let mut receipt = seal(syscap_body());
        // Flip the outcome after sealing; the stored receipt_id no longer matches.
        receipt["outcome"] = json!("FAIL");
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "self-digest");
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::NotVerified);
    }

    #[test]
    fn an_injected_extra_field_fails_the_self_digest() {
        // Strictness: the digest covers the whole document, so an added field that
        // no emitter sealed is caught (the struct-based check would silently drop
        // an unknown field — this Value-based check does not).
        let mut receipt = seal(syscap_body());
        receipt
            .as_object_mut()
            .unwrap()
            .insert("injected".to_string(), json!(true));
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "self-digest");
    }

    #[test]
    fn a_removed_field_fails_the_self_digest() {
        let mut receipt = seal(syscap_body());
        receipt.as_object_mut().unwrap().remove("note");
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "self-digest");
    }

    #[test]
    fn a_sealed_but_overclaiming_syscap_receipt_fails_honest_scope() {
        // Seal WITH the over-claim so the digest is intact; only the honest-scope
        // guard should reject it.
        let mut body = syscap_body();
        body["promotes_capability"] = json!(true);
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "honest-scope");
        assert!(err.reason.contains("promotes_capability=true"));
    }

    #[test]
    fn a_sealed_orchestration_speedup_claim_fails_honest_scope() {
        let mut body = orchestration_body();
        body["claims_speedup"] = json!(true);
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "honest-scope");
        assert!(err.reason.contains("claims_speedup=true"));
    }

    #[test]
    fn a_missing_scope_field_fails_honest_scope() {
        let mut body = syscap_body();
        body.as_object_mut().unwrap().remove("promotes_capability");
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "honest-scope");
    }

    #[test]
    fn a_bench_missing_its_scope_field_fails_honest_scope() {
        let mut body = bench_body();
        body.as_object_mut().unwrap().remove("speedup_claimed_for");
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "honest-scope");
    }

    #[test]
    fn an_unrecognized_schema_is_rejected() {
        let mut body = syscap_body();
        body["schema"] = json!("camelid.parity-receipt/v1");
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "schema");
    }

    #[test]
    fn a_malformed_receipt_id_is_rejected_before_hashing() {
        for bad in [
            "",
            "tooshort",
            &"a".repeat(63),
            &"a".repeat(65),
            &"A".repeat(64), // uppercase is not our lowercase-hex form
            &format!("{}z", "a".repeat(63)),
        ] {
            let mut receipt = syscap_body();
            receipt["receipt_id"] = json!(bad);
            let err = check(&receipt).expect_err("must fail");
            assert_eq!(err.phase, "self-digest", "id {bad:?} should be rejected");
        }
    }

    #[test]
    fn a_non_object_is_rejected() {
        let err = check(&json!(["not", "an", "object"])).expect_err("must fail");
        assert_eq!(err.phase, "parse");
    }

    #[test]
    fn a_missing_schema_is_rejected() {
        let mut body = syscap_body();
        body.as_object_mut().unwrap().remove("schema");
        let err = check(&body).expect_err("must fail");
        assert_eq!(err.phase, "schema");
    }

    #[test]
    fn run_reads_and_verifies_a_receipt_file() {
        let dir = std::env::temp_dir().join(format!("camelid-agent-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");

        let good = dir.join("good.json");
        std::fs::write(
            &good,
            serde_json::to_vec_pretty(&seal(syscap_body())).unwrap(),
        )
        .expect("write");
        assert_eq!(run(&good), AgentVerifyOutcome::Verified);

        let mut tampered_value = seal(syscap_body());
        tampered_value["outcome"] = json!("FAIL");
        let tampered = dir.join("tampered.json");
        std::fs::write(
            &tampered,
            serde_json::to_vec_pretty(&tampered_value).unwrap(),
        )
        .expect("write");
        assert_eq!(run(&tampered), AgentVerifyOutcome::NotVerified);

        let missing = dir.join("does-not-exist.json");
        assert_eq!(run(&missing), AgentVerifyOutcome::NotVerified);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
