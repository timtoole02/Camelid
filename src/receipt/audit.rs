//! Repo-wide integrity audit for the committed **sealed** receipts.
//!
//! [`super::verify`] and [`super::agent`] prove ONE receipt intact; this walks a
//! directory tree and checks EVERY sealed receipt's self-digest, so a corrupted
//! or hand-edited committed receipt cannot pass review unnoticed. It is the
//! mechanical companion to the CI gate - the automatic layer under "git history
//! + review" that the receipt scheme otherwise leans on entirely.
//!
//! Scope, deliberately narrow so the audit never false-fails:
//! - Only receipts whose `schema` is one of the families sealed with the shared
//!   [`receipt_id`](super::receipt_id_over) convention (parity, distributed, and
//!   the agent family) are checked. Every other JSON file - the many unsealed
//!   evidence / summary / prompt-pack schemas - is skipped.
//! - A sealed-family receipt that carries no `receipt_id` (some parity evidence
//!   items, and pre-sealing `agent_eval` receipts) is reported as *unsealed*,
//!   not failed: this gate checks the INTEGRITY of sealed receipts, not that
//!   sealing happened - which it cannot tell from a legitimately unsealed one.
//! - Only a receipt whose stored `receipt_id` does not match its recomputed
//!   digest is a failure. That is exactly the accidental-corruption / naive-edit
//!   case the seal exists to catch; it is not forgery-resistance (see
//!   [`super::agent`]).
//!
//! ## Tracked-debt baseline
//!
//! When first run against `qa/`, this audit surfaced [`BASELINE`] receipts whose
//! seals were already broken before it existed - some by privacy scrubs that
//! redacted paths/IPs after sealing without re-sealing, some sealed under an
//! earlier serialization. Those are grandfathered by their exact
//! `(stored, computed)` digest pair so the gate can enforce integrity going
//! forward without rewriting historical provenance. The match is content-keyed,
//! so any further edit to a baselined receipt changes its `computed` digest, no
//! longer matches, and is reported as a real failure - the debt is suppressed,
//! not the future.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::receipt_id_over;

/// One pre-existing broken seal, grandfathered by its exact digest pair. See the
/// module "Tracked-debt baseline" note. `path` is documentation only; matching
/// is on `(stored, computed)`, which a SHA-256 pair makes unique and
/// move-independent.
pub struct BaselineEntry {
    pub path: &'static str,
    pub stored: &'static str,
    pub computed: &'static str,
    pub reason: &'static str,
}

/// Sealed receipts already broken when this audit was introduced (2026-07-23),
/// grandfathered so the gate enforces integrity forward without touching
/// historical provenance. Any change to one of these files changes its
/// `computed` digest, drops the match, and turns it back into a hard failure.
///
/// The `computed` values are outputs of the current `receipt_id_over`
/// serialization. A `serde_json` change to number/float formatting would
/// recompute them - and would equally break every committed receipt carrying a
/// float, since the whole receipt scheme relies on that serialization being
/// stable. Treat a `serde_json` bump (it is a loose `1` dependency) as a
/// receipt-format event, and re-derive these values if it lands.
pub const BASELINE: &[BaselineEntry] = &[
    BaselineEntry {
        path: "qa/agent-orchestration/bench-rung4-1782702313.json",
        stored: "7c26a31bde23e41020565031beddfc74e959897cc08bd19b973d8399c06e3388",
        computed: "1e757f79691e7b1f236cca109f589c22ad932310bfdb886e0836c18e1a134054",
        reason: "sealed once and never edited (single commit); the stored digest predates the current canonical serialization and is not reproduced by it - historical sealing drift, not a post-seal edit",
    },
    BaselineEntry {
        path: "qa/agent-syscap/syscap-1782688037-PASS.json",
        stored: "06ac92dd39fb246525488ede277e4c11c166d3b4b308e2c83385eac580bfeaa4",
        computed: "cfea544cfb3c49ce44649edf8e86be2ead9e6676c09540558f7cb1e82f426fb4",
        reason: "privacy scrub replaced the Windows home path with `<home>` after sealing (commit d2825069); never re-sealed",
    },
    BaselineEntry {
        path: "qa/distributed/hetero-mac-pi-tinyllama-q8.json",
        stored: "11bbe0e14f38030bf4c3f55b221c79875aaabdc60b39c816e686754f3645ec9d",
        computed: "236fe8fc23ed619a14f8c7e0356e2c6b86de701a7c83dd7af4911509ba59fced",
        reason: "privacy scrub redacted LAN IPs / SSH-key path after sealing (commit cfaed035); never re-sealed",
    },
    BaselineEntry {
        path: "qa/distributed/two-mac-tinyllama-q8.json",
        stored: "33b79d8d0b99c729946de405009d4f293e364736d72b3985d47b3ae3587483be",
        computed: "4e6276cb6e5fbca61bf1ba26242f31140de41a6ff32c07f2e76a04f2823e2004",
        reason: "privacy scrub redacted LAN IPs / SSH-key path after sealing (commit cfaed035); never re-sealed",
    },
];

/// `.json` files under `qa/` that are intentionally NOT JSON - misnamed
/// artifacts, not receipts - excluded from the unparseable-failure set. Matched
/// by path suffix (separator- and EOL-robust; these are fixed non-receipt
/// artifacts). Keep this minimal: a NEW unparseable `.json` is a failure,
/// because a receipt corrupted into invalid JSON (e.g. a leftover merge-conflict
/// marker) is the exact threat this gate must not skip.
const UNPARSEABLE_ALLOWLIST: &[(&str, &str)] = &[
    (
        "qa/evidence-bundles/backend-q8-stream-diagnostics-loop-20260518T2222Z-head-7bfba9ac68e1/artifacts/same-host-plan.json",
        "benchmark plan in key=value text, not JSON (misnamed .json)",
    ),
    (
        "qa/evidence-bundles/mixtral-8x7b-v0.1-q8-current-head-sev1-20260514T124203Z-head-61e6e972a294/api/unload.json",
        "empty (0-byte) API-unload capture, not a receipt",
    ),
];

/// The documented reason a `.json` is allowed to be non-JSON, or `None` if an
/// unparseable file at `path` is a real failure. Path-suffix match, separator-
/// normalized so it holds regardless of how the audited directory was named.
fn unparseable_allowlist_reason(path: &Path) -> Option<&'static str> {
    let normalized = path.to_string_lossy().replace('\\', "/");
    UNPARSEABLE_ALLOWLIST
        .iter()
        .find(|(allow, _)| normalized.ends_with(allow))
        .map(|(_, reason)| *reason)
}

/// The receipt schemas sealed with the `receipt_id` self-digest convention,
/// sourced from each family's own schema constant so the set cannot silently
/// drift from what the emitters produce.
///
/// KEEP IN SYNC: when a new receipt family starts sealing with `receipt_id`, add
/// its schema constant here, or the gate will silently skip every receipt of
/// that family. There is no registry of "schemas that carry a receipt_id" to
/// derive this from.
pub fn sealed_schemas() -> Vec<&'static str> {
    let mut schemas = vec![
        super::RECEIPT_SCHEMA_V1,
        super::distributed::DISTRIBUTED_RECEIPT_SCHEMA_V1,
    ];
    schemas.extend(super::agent::RECOGNIZED_SCHEMAS);
    schemas
}

/// A committed sealed receipt whose stored digest does not match its body and is
/// not grandfathered by [`BASELINE`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestMismatch {
    pub path: PathBuf,
    pub stored: String,
    pub computed: String,
}

/// A mismatch grandfathered by [`BASELINE`] - reported as tracked debt, never a
/// failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowlistedMismatch {
    pub path: PathBuf,
    pub reason: &'static str,
}

/// The outcome of auditing a directory tree.
#[derive(Debug, Default)]
pub struct AuditReport {
    /// `.json` files read.
    pub scanned: usize,
    /// Sealed receipts whose digest matched.
    pub verified: usize,
    /// Sealed-family receipts carrying no (or an empty) `receipt_id`.
    pub unsealed: Vec<PathBuf>,
    /// `.json` files that could not be read (I/O error or non-UTF-8). A receipt
    /// corrupted into unreadable bytes lands here, so it is a failure.
    pub unreadable: Vec<PathBuf>,
    /// `.json` files that did not parse as JSON (e.g. a merge-conflict marker
    /// left in a receipt). A failure: the seal cannot even be reached, and this
    /// is exactly the "bad merge" shape the gate must not skip.
    pub unparseable: Vec<PathBuf>,
    /// Non-JSON `.json` files grandfathered by [`UNPARSEABLE_ALLOWLIST`]
    /// (misnamed artifacts) - reported, never a failure.
    pub unparseable_allowlisted: Vec<PathBuf>,
    /// Mismatches grandfathered by [`BASELINE`] (tracked debt, not failures).
    pub allowlisted: Vec<AllowlistedMismatch>,
    /// Sealed receipts whose digest did not match and are NOT grandfathered.
    pub mismatches: Vec<DigestMismatch>,
    /// Indices into the baseline matched this run (for stale detection).
    matched_baseline: HashSet<usize>,
}

impl AuditReport {
    /// True when no un-grandfathered corruption was found. An unreadable or
    /// unparseable `.json` (a plausible corrupted-receipt shape) is a failure
    /// too, so the gate does not fail open on a receipt broken into invalid JSON.
    pub fn ok(&self) -> bool {
        self.mismatches.is_empty() && self.unreadable.is_empty() && self.unparseable.is_empty()
    }

    /// Process exit code: 0 when every sealed receipt is intact or grandfathered,
    /// 1 otherwise.
    pub fn exit_code(&self) -> i32 {
        i32::from(!self.ok())
    }
}

/// Classify and check a single already-parsed receipt found at `path`, updating
/// `report`. No filesystem I/O, so it is exhaustively unit-testable.
fn audit_value(
    path: &Path,
    value: &Value,
    sealed: &[&str],
    baseline: &[BaselineEntry],
    report: &mut AuditReport,
) {
    let Some(obj) = value.as_object() else {
        return;
    };
    let Some(schema) = obj.get("schema").and_then(Value::as_str) else {
        return;
    };
    if !sealed.contains(&schema) {
        return; // not a sealed family - it never carried a digest to check
    }
    let stored = match obj.get("receipt_id").and_then(Value::as_str) {
        None | Some("") => {
            report.unsealed.push(path.to_path_buf());
            return;
        }
        Some(stored) => stored,
    };
    let computed = receipt_id_over(value);
    if computed == stored {
        report.verified += 1;
        return;
    }
    // A mismatch: grandfathered by its exact (stored, computed) pair, or a real
    // failure. Content-keyed, so any further edit drops the match.
    match baseline
        .iter()
        .position(|entry| entry.stored == stored && entry.computed == computed)
    {
        Some(index) => {
            report.matched_baseline.insert(index);
            report.allowlisted.push(AllowlistedMismatch {
                path: path.to_path_buf(),
                reason: baseline[index].reason,
            });
        }
        None => report.mismatches.push(DigestMismatch {
            path: path.to_path_buf(),
            stored: stored.to_string(),
            computed,
        }),
    }
}

/// Baseline entries not encountered in this run (receipt re-sealed, removed, or
/// edited further). Reported so the debt list can be pruned; never a failure.
pub fn stale_baseline<'a>(
    baseline: &'a [BaselineEntry],
    report: &AuditReport,
) -> Vec<&'a BaselineEntry> {
    baseline
        .iter()
        .enumerate()
        .filter(|(index, _)| !report.matched_baseline.contains(index))
        .map(|(_, entry)| entry)
        .collect()
}

/// Recursively audit every `*.json` file under `dir`. Files that do not parse as
/// JSON are counted as scanned and skipped (many `.json` under `qa/` are not
/// receipts).
pub fn audit_dir(dir: &Path) -> std::io::Result<AuditReport> {
    let sealed = sealed_schemas();
    let mut report = AuditReport::default();
    audit_dir_inner(dir, &sealed, BASELINE, &mut report)?;
    Ok(report)
}

fn audit_dir_inner(
    dir: &Path,
    sealed: &[&str],
    baseline: &[BaselineEntry],
    report: &mut AuditReport,
) -> std::io::Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    // Deterministic order so the printed summary is stable across runs/hosts.
    entries.sort();
    for path in entries {
        if path.is_dir() {
            audit_dir_inner(&path, sealed, baseline, report)?;
        } else if path.extension().is_some_and(|ext| ext == "json") {
            report.scanned += 1;
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(_) => {
                    report.unreadable.push(path);
                    continue;
                }
            };
            match serde_json::from_str::<Value>(&text) {
                Ok(value) => audit_value(&path, &value, sealed, baseline, report),
                Err(_) => {
                    if unparseable_allowlist_reason(&path).is_some() {
                        report.unparseable_allowlisted.push(path);
                    } else {
                        report.unparseable.push(path);
                    }
                }
            }
        }
    }
    Ok(())
}

/// CLI entry: audit `dir`, print a summary, and return the process exit code.
pub fn run(dir: &Path) -> i32 {
    let report = match audit_dir(dir) {
        Ok(report) => report,
        Err(err) => {
            println!("FAIL: could not read {}: {err}", dir.display());
            return 1;
        }
    };
    for mismatch in &report.mismatches {
        println!(
            "FAIL digest: {} (stored {}, computed {})",
            mismatch.path.display(),
            mismatch.stored,
            mismatch.computed
        );
    }
    for path in &report.unparseable {
        println!(
            "FAIL unparseable: {} (not valid JSON - corrupted receipt or merge markers?)",
            path.display()
        );
    }
    for path in &report.unreadable {
        println!(
            "FAIL unreadable: {} (I/O error or non-UTF-8)",
            path.display()
        );
    }
    for path in &report.unparseable_allowlisted {
        println!("NOTE non-JSON (allowlisted): {}", path.display());
    }
    for entry in &report.allowlisted {
        println!(
            "NOTE tracked-debt (allowlisted): {} - {}",
            entry.path.display(),
            entry.reason
        );
    }
    for path in &report.unsealed {
        println!(
            "NOTE unsealed (no receipt_id, not verifiable): {}",
            path.display()
        );
    }
    for entry in stale_baseline(BASELINE, &report) {
        println!(
            "NOTE stale baseline entry (not encountered - prune it?): {}",
            entry.path
        );
    }
    let failed = report.mismatches.len() + report.unparseable.len() + report.unreadable.len();
    println!(
        "scanned {} json file(s): {} verified, {} allowlisted debt, {} unsealed, {} failed",
        report.scanned,
        report.verified,
        report.allowlisted.len(),
        report.unsealed.len(),
        failed
    );
    if report.ok() {
        println!(
            "ALL SEALED RECEIPTS INTACT (excluding {} allowlisted)",
            report.allowlisted.len()
        );
    } else {
        println!("RECEIPT INTEGRITY CHECK FAILED ({failed} problem(s), not allowlisted)");
    }
    report.exit_code()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn seal(mut body: Value) -> Value {
        let id = crate::receipt::receipt_id_over(&body);
        body.as_object_mut()
            .expect("object")
            .insert("receipt_id".to_string(), Value::String(id));
        body
    }

    /// A minimal object stamped with a real sealed-family schema. The audit is
    /// generic over the body, so a full typed receipt is unnecessary.
    fn sealed_body() -> Value {
        json!({
            "schema": crate::receipt::RECEIPT_SCHEMA_V1,
            "receipt_id": "",
            "created_utc": "2026-01-01T00:00:00Z",
            "nested": { "b": 2, "a": [1, 2, 3] }
        })
    }

    fn report_of(value: &Value) -> AuditReport {
        report_with_baseline(value, &[])
    }

    fn report_with_baseline(value: &Value, baseline: &[BaselineEntry]) -> AuditReport {
        let sealed = sealed_schemas();
        let mut report = AuditReport::default();
        audit_value(Path::new("t.json"), value, &sealed, baseline, &mut report);
        report
    }

    #[test]
    fn sealed_schemas_covers_parity_distributed_and_agent() {
        let s = sealed_schemas();
        assert!(s.contains(&crate::receipt::RECEIPT_SCHEMA_V1));
        assert!(s.contains(&crate::receipt::distributed::DISTRIBUTED_RECEIPT_SCHEMA_V1));
        for schema in crate::receipt::agent::RECOGNIZED_SCHEMAS {
            assert!(s.contains(&schema), "missing agent schema {schema}");
        }
    }

    #[test]
    fn a_sealed_receipt_verifies() {
        let report = report_of(&seal(sealed_body()));
        assert_eq!(report.verified, 1);
        assert!(report.mismatches.is_empty());
        assert!(report.allowlisted.is_empty());
        assert!(report.unsealed.is_empty());
    }

    #[test]
    fn a_tampered_sealed_receipt_is_a_mismatch() {
        let mut receipt = seal(sealed_body());
        receipt["created_utc"] = json!("2099-12-31T23:59:59Z");
        let report = report_of(&receipt);
        assert_eq!(report.verified, 0);
        assert_eq!(report.mismatches.len(), 1);
        assert!(!report.ok());
        assert_eq!(report.exit_code(), 1);
    }

    #[test]
    fn a_baselined_mismatch_is_tracked_debt_not_a_failure() {
        let sealed = seal(sealed_body());
        let stored = sealed["receipt_id"].as_str().unwrap().to_string();
        let mut tampered = sealed;
        tampered["created_utc"] = json!("2099-12-31T23:59:59Z");
        let computed = receipt_id_over(&tampered);
        let baseline = [BaselineEntry {
            path: "t.json",
            // Deliberate test-scoped leak to get the `&'static str` the field
            // wants; the test process is short-lived.
            stored: stored.leak(),
            computed: computed.leak(),
            reason: "test debt",
        }];

        let report = report_with_baseline(&tampered, &baseline);
        assert!(report.mismatches.is_empty());
        assert_eq!(report.allowlisted.len(), 1);
        assert_eq!(report.allowlisted[0].reason, "test debt");
        assert!(report.ok());
        assert_eq!(report.exit_code(), 0);
        assert!(stale_baseline(&baseline, &report).is_empty());
    }

    #[test]
    fn a_baseline_entry_that_no_longer_matches_is_stale() {
        let baseline = [BaselineEntry {
            path: "gone.json",
            stored: "a".repeat(64).leak(),
            computed: "b".repeat(64).leak(),
            reason: "removed",
        }];
        let report = report_with_baseline(&seal(sealed_body()), &baseline);
        assert_eq!(report.verified, 1);
        assert!(report.ok());
        assert_eq!(stale_baseline(&baseline, &report).len(), 1);
    }

    #[test]
    fn a_further_edit_to_a_baselined_receipt_fails_again() {
        // Grandfathering is keyed on the exact (stored, computed); a further edit
        // changes computed, drops the match, and becomes a real failure.
        let sealed = seal(sealed_body());
        let stored = sealed["receipt_id"].as_str().unwrap().to_string();
        let mut tampered = sealed;
        tampered["created_utc"] = json!("2099-12-31T23:59:59Z");
        let old_computed = receipt_id_over(&tampered);
        let baseline = [BaselineEntry {
            path: "t.json",
            stored: stored.leak(),
            computed: old_computed.leak(),
            reason: "test debt",
        }];
        // Edit it further: computed changes, no longer matches the baseline.
        tampered["nested"]["b"] = json!(999);
        let report = report_with_baseline(&tampered, &baseline);
        assert_eq!(report.mismatches.len(), 1, "further edit must fail");
        assert!(report.allowlisted.is_empty());
    }

    #[test]
    fn a_sealed_family_receipt_without_id_is_unsealed_not_failed() {
        let mut body = sealed_body();
        body.as_object_mut().unwrap().remove("receipt_id");
        let report = report_of(&body);
        assert_eq!(report.unsealed.len(), 1);
        assert!(report.ok());
    }

    #[test]
    fn an_unsealed_schema_is_skipped_even_with_a_receipt_id() {
        let other = json!({
            "schema": "camelid.speed-receipt/v1",
            "receipt_id": "not-our-digest",
            "x": 1
        });
        let report = report_of(&other);
        assert_eq!(report.verified, 0);
        assert!(report.mismatches.is_empty());
        assert!(report.unsealed.is_empty());
    }

    #[test]
    fn a_non_object_or_schemaless_value_is_ignored() {
        assert!(report_of(&json!([1, 2, 3])).ok());
        assert_eq!(report_of(&json!([1, 2, 3])).verified, 0);
        let no_schema = json!({ "receipt_id": "x", "y": 1 });
        assert_eq!(report_of(&no_schema).verified, 0);
        assert!(report_of(&no_schema).unsealed.is_empty());
    }

    #[test]
    fn unparseable_allowlist_matches_only_the_documented_non_json_artifacts() {
        assert!(unparseable_allowlist_reason(Path::new(
            "qa/evidence-bundles/mixtral-8x7b-v0.1-q8-current-head-sev1-20260514T124203Z-head-61e6e972a294/api/unload.json"
        ))
        .is_some());
        // Windows separators and a leading dir still match (suffix, normalized).
        assert!(unparseable_allowlist_reason(Path::new(
            r"C:\repo\qa\evidence-bundles\backend-q8-stream-diagnostics-loop-20260518T2222Z-head-7bfba9ac68e1\artifacts\same-host-plan.json"
        ))
        .is_some());
        assert!(
            unparseable_allowlist_reason(Path::new("qa/receipts/anything-else.json")).is_none()
        );
    }

    #[test]
    fn audit_dir_walks_recursively_and_reports() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();

        std::fs::write(
            root.join("good.json"),
            serde_json::to_vec(&seal(sealed_body())).unwrap(),
        )
        .unwrap();
        let mut bad = seal(sealed_body());
        bad["created_utc"] = json!("2099-01-01T00:00:00Z");
        std::fs::write(nested.join("bad.json"), serde_json::to_vec(&bad).unwrap()).unwrap();
        std::fs::write(
            root.join("other.json"),
            b"{\"schema\":\"camelid.speed-receipt/v1\"}",
        )
        .unwrap();
        std::fs::write(root.join("notes.txt"), b"not json").unwrap();
        // A merge-conflict marker in a .json: must FAIL as unparseable, not skip.
        std::fs::write(root.join("conflict.json"), b"{\n<<<<<<< HEAD\n}\n").unwrap();

        let report = audit_dir(root).unwrap();
        assert_eq!(report.verified, 1);
        assert_eq!(report.mismatches.len(), 1);
        assert!(report.mismatches[0].path.ends_with("bad.json"));
        assert_eq!(
            report.unparseable.len(),
            1,
            "a conflict-marker .json must fail, not be silently skipped"
        );
        assert!(report.unparseable[0].ends_with("conflict.json"));
        assert!(!report.ok());
        assert_eq!(report.scanned, 4); // good, bad, other, conflict (not the .txt)
        assert_eq!(run(root), 1);
        // tempdir cleans up on drop, even on panic.
    }
}
